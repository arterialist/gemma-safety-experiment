"""Permanent weight modification (abliteration) algorithms tailored for Gemma 2 / Gemma 3.

Implements:
1. naive_orthogonalize: Standard Arditi et al. projection ignoring Gemma's post_layernorms.
2. postnorm_exact_orthogonalize: Gemma-exact projection mapping target basis Q through
   diag(1 + w_post_layernorm) via QR decomposition so residual writes are strictly orthogonal to Q.
3. postnorm_col_mpoa: Gemma-exact projection + column-norm preservation (preserves exact orthogonality).
4. postnorm_row_mpoa: Gemma-exact projection + row-norm preservation (MPOA).

Supports single directions (k=1) or multi-direction concept subspaces (k>=2) and selective layer ranges.
"""

import os
import json
import copy
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from .mech_interp import get_text_backbone


def _to_orthonormal_basis(directions: torch.Tensor) -> torch.Tensor:
    """Convert [D] or [K, D] or [D, K] tensor into orthonormal columns Q of shape [D, K] in float32."""
    if directions.ndim == 1:
        q = directions.float().unsqueeze(1)  # [D, 1]
    elif directions.ndim == 2:
        # Assume shape [K, D] if rows < cols, else [D, K]
        if directions.shape[0] < directions.shape[1]:
            q = directions.float().t()
        else:
            q = directions.float()
    else:
        raise ValueError(f"Expected 1D or 2D direction tensor, got {directions.shape}")

    # QR factorization gives orthonormal columns Q of shape [D, K]
    q_orth, _ = torch.linalg.qr(q, mode="reduced")
    return q_orth


def compute_postnorm_target_basis(
    q_orth: torch.Tensor,
    post_norm_weight: Optional[torch.Tensor]
) -> torch.Tensor:
    """Map residual-stream orthonormal basis Q [D, K] into pre-postnorm space.
    
    In Gemma 2/3, RMSNorm scales each coordinate by g = (1.0 + w_post).
    For (g * z) to be orthogonal to Q, z must be orthogonal to diag(g) @ Q.
    Returns orthonormal basis Q_tilde [D, K] spanning col(diag(g) @ Q).
    """
    if post_norm_weight is None:
        return q_orth
    g = (1.0 + post_norm_weight.detach().float().cpu()).unsqueeze(1)  # [D, 1]
    scaled_q = g * q_orth
    q_tilde, _ = torch.linalg.qr(scaled_q, mode="reduced")
    return q_tilde


def orthogonalize_linear_weight(
    weight: torch.Tensor,
    basis_q: torch.Tensor,
    alpha: float = 1.0,
    norm_mode: str = "exact"
) -> torch.Tensor:
    """Orthogonalize an nn.Linear weight [d_out, d_in] against orthonormal basis_q [d_out, K].
    
    norm_mode:
      - 'exact': W' = W - alpha * Q @ (Q^T @ W)
      - 'col_mpoa': W' = W - alpha * Q @ (Q^T @ W), then rescale each column back to original L2 norm
                    (preserves exact orthogonality when alpha=1.0 AND preserves column weight energy)
      - 'row_mpoa': W' = W - alpha * Q @ (Q^T @ W), then rescale each row back to original L2 norm
    """
    orig_dtype = weight.dtype
    w = weight.detach().float().cpu()
    q = basis_q.float().cpu()  # [d_out, K]

    # Projected matrix
    proj = q @ (q.t() @ w)  # [d_out, d_in]
    w_new = w - alpha * proj

    if norm_mode == "exact":
        pass
    elif norm_mode == "col_mpoa":
        orig_col_norms = w.norm(dim=0, keepdim=True).clamp(min=1e-8)
        new_col_norms = w_new.norm(dim=0, keepdim=True).clamp(min=1e-8)
        w_new = w_new * (orig_col_norms / new_col_norms)
    elif norm_mode == "row_mpoa":
        orig_row_norms = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
        new_row_norms = w_new.norm(dim=1, keepdim=True).clamp(min=1e-8)
        w_new = w_new * (orig_row_norms / new_row_norms)
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    return w_new.to(dtype=orig_dtype)


def apply_weight_surgery(
    model: AutoModelForCausalLM,
    layer_directions: Dict[int, torch.Tensor],
    method: str = "postnorm_col_mpoa",
    alpha: float = 1.0,
    edit_attn: bool = True,
    edit_mlp: bool = True,
    layer_alphas: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Modify model weights in-place to permanently remove refusal direction(s).
    
    method options:
      - 'naive': Ignore Gemma post_layernorms, norm_mode='exact'
      - 'postnorm_exact': Map through post_*_layernorm, norm_mode='exact'
      - 'postnorm_col_mpoa': Map through post_*_layernorm, norm_mode='col_mpoa'
      - 'postnorm_row_mpoa': Map through post_*_layernorm, norm_mode='row_mpoa'
    """
    use_postnorm = method != "naive"
    if method == "naive":
        norm_mode = "exact"
    elif method == "postnorm_exact":
        norm_mode = "exact"
    elif method == "postnorm_col_mpoa":
        norm_mode = "col_mpoa"
    elif method == "postnorm_row_mpoa":
        norm_mode = "row_mpoa"
    else:
        raise ValueError(f"Unknown surgery method: {method}")

    _, layers = get_text_backbone(model)
    edited_layers = []
    max_residual_leak = 0.0

    with torch.no_grad():
        for l_idx, raw_dir in sorted(layer_directions.items()):
            layer = layers[l_idx]
            a_l = layer_alphas.get(l_idx, alpha) if layer_alphas else alpha
            if a_l == 0.0:
                continue

            q_orth = _to_orthonormal_basis(raw_dir)  # [D, K]

            if edit_attn and hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                post_attn_w = (
                    layer.post_attention_layernorm.weight
                    if (use_postnorm and hasattr(layer, "post_attention_layernorm"))
                    else None
                )
                q_attn = compute_postnorm_target_basis(q_orth, post_attn_w)
                w_o = layer.self_attn.o_proj.weight
                w_o_new = orthogonalize_linear_weight(w_o, q_attn, alpha=a_l, norm_mode=norm_mode)
                w_o.copy_(w_o_new.to(device=w_o.device, dtype=w_o.dtype))

                # Measure post-norm residual leak: || Q^T diag(1+w) W_new || / || W_new ||
                if hasattr(layer, "post_attention_layernorm"):
                    g = (1.0 + layer.post_attention_layernorm.weight.detach().float().cpu()).unsqueeze(1)
                    scaled_w = g * w_o.detach().float().cpu()
                    leak = float((q_orth.t() @ scaled_w).norm() / scaled_w.norm().clamp(min=1e-8))
                    max_residual_leak = max(max_residual_leak, leak)

            if edit_mlp and hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                post_ffn_w = (
                    layer.post_feedforward_layernorm.weight
                    if (use_postnorm and hasattr(layer, "post_feedforward_layernorm"))
                    else None
                )
                q_ffn = compute_postnorm_target_basis(q_orth, post_ffn_w)
                w_d = layer.mlp.down_proj.weight
                w_d_new = orthogonalize_linear_weight(w_d, q_ffn, alpha=a_l, norm_mode=norm_mode)
                w_d.copy_(w_d_new.to(device=w_d.device, dtype=w_d.dtype))

                if hasattr(layer, "post_feedforward_layernorm"):
                    g = (1.0 + layer.post_feedforward_layernorm.weight.detach().float().cpu()).unsqueeze(1)
                    scaled_w = g * w_d.detach().float().cpu()
                    leak = float((q_orth.t() @ scaled_w).norm() / scaled_w.norm().clamp(min=1e-8))
                    max_residual_leak = max(max_residual_leak, leak)

            edited_layers.append(l_idx)

    return {
        "method": method,
        "alpha": alpha,
        "edited_layers": edited_layers,
        "max_postnorm_leak_ratio": max_residual_leak,
    }


def extract_subspace_directions(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
    top_k: int = 2,
    biproject_harmless: bool = True
) -> Dict[int, torch.Tensor]:
    """Compute top-K orthogonal directions per layer from contrastive activations.
    
    Uses difference-in-means as v1, plus top singular vectors of centered harmful-vs-harmless
    residuals (orthogonalized against harmless mean if biproject_harmless=True).
    """
    num_layers = harmful_acts.shape[1]
    out: Dict[int, torch.Tensor] = {}

    for l in range(num_layers):
        h = harmful_acts[:, l, :].float()
        b = harmless_acts[:, l, :].float()
        mu_h = h.mean(dim=0)
        mu_b = b.mean(dim=0)
        diff = mu_h - mu_b

        if top_k == 1:
            vecs = diff.unsqueeze(0)  # [1, D]
        else:
            # First direction is mean difference; remaining k-1 capture orthogonal variance modes
            diff_hat = diff / diff.norm().clamp(min=1e-8)
            centered = h - mu_b.unsqueeze(0)
            centered_perp = centered - (centered @ diff_hat).unsqueeze(1) * diff_hat.unsqueeze(0)
            _, _, vh = torch.linalg.svd(centered_perp, full_matrices=False)
            vecs = torch.cat([diff.unsqueeze(0), vh[:top_k - 1]], dim=0)

        if biproject_harmless:
            b_hat = mu_b / mu_b.norm().clamp(min=1e-8)
            vecs = vecs - (vecs @ b_hat).unsqueeze(1) * b_hat.unsqueeze(0)

        # Orthonormalize [D, K]
        q, _ = torch.linalg.qr(vecs.t(), mode="reduced")
        out[l] = q  # [D, K]

    return out


def save_modified_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    save_dir: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Save weight-modified Gemma model and tokenizer cleanly to disk."""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    if metadata is not None:
        with open(os.path.join(save_dir, "surgery_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    return save_dir
