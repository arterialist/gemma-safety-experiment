"""Gemma Scope 2 JumpReLU Sparse Autoencoder analysis for Gemma 3.

Supports loading `google/gemma-scope-2-1b-it` (or 4b-it) JumpReLU SAEs directly from HuggingFace,
verifying reconstruction fidelity (FVU / cosine), encoding residual activations into sparse
monosemantic features, computing differential feature activations (Harmful vs Harmless vs OR-Bench),
and decoding feature semantics via cosine alignment with refusal directions and logit lens.
"""

import json
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

from .mech_interp import get_lm_head, get_text_backbone


class JumpReLUSAE:
    """Exact Gemma Scope 2 JumpReLU Sparse Autoencoder."""

    def __init__(self, params: Dict[str, torch.Tensor], config: Dict[str, Any]):
        self.w_enc = params["w_enc"].float()       # [d_model, d_sae]
        self.b_enc = params["b_enc"].float()       # [d_sae]
        self.threshold = params["threshold"].float()  # [d_sae]
        self.w_dec = params["w_dec"].float()       # [d_sae, d_model]
        self.b_dec = params["b_dec"].float()       # [d_model]
        self.config = config
        self.d_model, self.d_sae = self.w_enc.shape

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "google/gemma-scope-2-1b-it",
        sae_path: str = "resid_post/layer_17_width_16k_l0_medium"
    ) -> "JumpReLUSAE":
        cfg_file = hf_hub_download(repo_id, f"{sae_path}/config.json")
        params_file = hf_hub_download(repo_id, f"{sae_path}/params.safetensors")
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        params = load_file(params_file)
        return cls(params, cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations x [..., d_model] -> sparse latents [..., d_sae]."""
        x_f = x.float()
        pre_acts = x_f @ self.w_enc + self.b_enc
        return pre_acts * (pre_acts > self.threshold).float()

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode sparse latents f [..., d_sae] -> reconstructed activations [..., d_model]."""
        return f.float() @ self.w_dec + self.b_dec

    def reconstruction_metrics(self, x: torch.Tensor) -> Dict[str, float]:
        """Measure L0 sparsity, Fraction of Variance Unexplained (FVU), and cosine similarity."""
        x_f = x.float()
        f = self.encode(x_f)
        x_hat = self.decode(f)
        l0 = float((f > 0).float().sum(dim=-1).mean().item())
        mse = ((x_f - x_hat) ** 2).sum()
        var = ((x_f - x_f.mean(dim=0, keepdim=True)) ** 2).sum().clamp(min=1e-8)
        fvu = float((mse / var).item())
        cos = float(torch.nn.functional.cosine_similarity(x_f, x_hat, dim=-1).mean().item())
        return {"l0": l0, "fvu": fvu, "explained_variance": 1.0 - fvu, "cosine": cos}


def analyze_sae_refusal_features(
    sae: JumpReLUSAE,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_acts_layer: torch.Tensor,
    harmless_acts_layer: torch.Tensor,
    refusal_dir: torch.Tensor,
    over_refusal_acts_layer: torch.Tensor = None,
    top_k_features: int = 12
) -> Dict[str, Any]:
    """Decompose refusal activations & direction into Gemma Scope 2 JumpReLU features."""
    f_harm = sae.encode(harmful_acts_layer)       # [N_h, d_sae]
    f_benign = sae.encode(harmless_acts_layer)    # [N_b, d_sae]

    recon_harm = sae.reconstruction_metrics(harmful_acts_layer)
    recon_benign = sae.reconstruction_metrics(harmless_acts_layer)

    mean_diff = f_harm.mean(dim=0) - f_benign.mean(dim=0)  # [d_sae]
    fire_rate_harm = (f_harm > 0).float().mean(dim=0)
    fire_rate_benign = (f_benign > 0).float().mean(dim=0)

    if over_refusal_acts_layer is not None:
        f_or = sae.encode(over_refusal_acts_layer)
        fire_rate_or = (f_or > 0).float().mean(dim=0)
        mean_or = f_or.mean(dim=0)
    else:
        f_or = None
        fire_rate_or = None
        mean_or = None

    # Normalize refusal_dir and w_dec rows for cosine alignment
    r_hat = (refusal_dir.float() / refusal_dir.float().norm().clamp(min=1e-8))
    w_dec_norm = sae.w_dec / sae.w_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dec_cos_with_refusal = w_dec_norm @ r_hat  # [d_sae]

    # Top positively differential features (more active on harmful than harmless)
    top_pos_vals, top_pos_idx = torch.topk(mean_diff, k=top_k_features)
    # Top features aligned by decoder cosine * mean_diff (causal contribution to refusal dir)
    refusal_contrib = mean_diff * dec_cos_with_refusal * sae.w_dec.norm(dim=-1)
    top_contrib_vals, top_contrib_idx = torch.topk(refusal_contrib, k=top_k_features)

    # Logit lens on decoder vectors of top features
    lm_head = get_lm_head(model)
    w_u = lm_head.weight.detach().float().cpu()

    def describe_features(indices: torch.Tensor) -> List[Dict[str, Any]]:
        entries = []
        for idx_t in indices:
            idx = int(idx_t.item())
            dec_vec = sae.w_dec[idx]
            logits = w_u @ dec_vec
            top_tok_vals, top_tok_ids = torch.topk(logits, k=6)
            top_tokens = [tokenizer.decode([int(t)]) for t in top_tok_ids]
            entry = {
                "feature_id": idx,
                "mean_act_harm": float(f_harm[:, idx].mean().item()),
                "mean_act_harmless": float(f_benign[:, idx].mean().item()),
                "mean_diff": float(mean_diff[idx].item()),
                "fire_rate_harm": float(fire_rate_harm[idx].item()),
                "fire_rate_harmless": float(fire_rate_benign[idx].item()),
                "cos_with_refusal_dir": float(dec_cos_with_refusal[idx].item()),
                "refusal_dir_contribution": float(refusal_contrib[idx].item()),
                "decoder_top_tokens": top_tokens,
            }
            if f_or is not None:
                entry["mean_act_or_bench"] = float(mean_or[idx].item())
                entry["fire_rate_or_bench"] = float(fire_rate_or[idx].item())
            entries.append(entry)
        return entries

    # Total cosine between SAE-reconstructed diff vector and true refusal direction
    sae_diff_in_resid = mean_diff @ sae.w_dec
    sae_vs_true_cos = float(torch.nn.functional.cosine_similarity(
        sae_diff_in_resid.unsqueeze(0), r_hat.unsqueeze(0)
    ).item())

    return {
        "reconstruction_harmful": recon_harm,
        "reconstruction_harmless": recon_benign,
        "sae_reconstructed_diff_cosine_with_refusal": sae_vs_true_cos,
        "top_differential_features": describe_features(top_pos_idx),
        "top_refusal_contributing_features": describe_features(top_contrib_idx),
    }
