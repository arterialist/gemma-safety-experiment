"""Mechanistic interpretability tools for refusal analysis in Gemma models.

Covers:
- Residual stream extraction at user-turn end (t_inst) and assistant-turn start (t_post)
- Contrastive difference-in-means direction discovery across all layers
- Linear probe and projection separability (AUROC on harmful vs harmless vs over-refusal)
- Logit lens projection of refusal directions onto vocabulary
- Inference-time causal activation steering (addition and ablation)
"""

import contextlib
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM

from .model_utils import format_user_prompt


def _unwrap_peft(model: AutoModelForCausalLM):
    """Unwrap PEFT wrapper if present to access underlying HuggingFace Gemma module."""
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    return model


def get_text_backbone(model: AutoModelForCausalLM):
    """Return text model backbone and decoder layers list for Gemma 2 / Gemma 3 / PEFT wrappers."""
    m = _unwrap_peft(model)
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model, m.model.layers
    if hasattr(m, "language_model") and hasattr(m.language_model, "model"):
        return m.language_model.model, m.language_model.model.layers
    if hasattr(m, "model") and hasattr(m.model, "language_model"):
        return m.model.language_model, m.model.language_model.layers
    raise ValueError(f"Unsupported Gemma model structure: {type(model)}")


def get_lm_head(model: AutoModelForCausalLM):
    m = _unwrap_peft(model)
    if hasattr(m, "lm_head"):
        return m.lm_head
    if hasattr(m, "language_model") and hasattr(m.language_model, "lm_head"):
        return m.language_model.lm_head
    raise ValueError("Cannot locate lm_head")


def extract_residual_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    positions: Tuple[int, ...] = (-6, -5, -1),
    batch_size: int = 16
) -> Dict[int, torch.Tensor]:
    """Extract post-layer residual activations at specified negative token offsets from sequence end.
    
    For Gemma 3 chat template:
      pos -6: final user instruction token (t_inst)
      pos -5: <end_of_turn>
      pos -1: trailing newline after <start_of_turn>model (t_post)
    
    Returns dict mapping token_offset -> Tensor of shape [num_prompts, num_layers, d_model] in float32 on CPU.
    """
    device = next(model.parameters()).device
    _, layers = get_text_backbone(model)
    num_layers = len(layers)

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    all_pos_batches: Dict[int, List[torch.Tensor]] = {p: [] for p in positions}

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        formatted = [format_user_prompt(tokenizer, p) for p in batch]
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        # hidden_states has length num_layers + 1: index 0 is embeddings, 1..num_layers are post-layer outputs
        hidden_states = outputs.hidden_states
        for pos in positions:
            # Stack layers 0..num_layers-1 -> shape [batch, num_layers, d_model]
            layer_vecs = torch.stack(
                [hidden_states[l + 1][:, pos, :].float().cpu() for l in range(num_layers)],
                dim=1
            )
            all_pos_batches[pos].append(layer_vecs)

    tokenizer.padding_side = orig_padding_side
    return {pos: torch.cat(batches, dim=0) for pos, batches in all_pos_batches.items()}


def compute_refusal_directions(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Compute layer-wise mean activations, raw difference-in-means, and unit refusal directions.
    
    Inputs have shape [N, num_layers, d_model].
    """
    mu_harm = harmful_acts.mean(dim=0)        # [L, D]
    mu_benign = harmless_acts.mean(dim=0)     # [L, D]
    diff = mu_harm - mu_benign                # [L, D]
    norms = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit_dirs = diff / norms                  # [L, D]

    # Also compute orthogonalized (biprojected) direction removing component parallel to harmless mean
    mu_b_norm_sq = (mu_benign * mu_benign).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    proj_coeff = (diff * mu_benign).sum(dim=-1, keepdim=True) / mu_b_norm_sq
    biproj = diff - proj_coeff * mu_benign
    biproj_unit = biproj / biproj.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return {
        "mu_harm": mu_harm,
        "mu_harmless": mu_benign,
        "diff": diff,
        "unit_dirs": unit_dirs,
        "biproj_unit_dirs": biproj_unit,
        "diff_norms": norms.squeeze(-1),
    }


def analyze_layer_separability(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
    over_refusal_acts: Optional[torch.Tensor] = None
) -> List[Dict[str, Any]]:
    """Compute per-layer refusal separability metrics (diff-in-means AUROC, CV linear probe AUROC, cosine)."""
    num_layers = harmful_acts.shape[1]
    dirs = compute_refusal_directions(harmful_acts, harmless_acts)
    unit_dirs = dirs["unit_dirs"]

    metrics = []
    for l in range(num_layers):
        r = unit_dirs[l]
        h_proj = (harmful_acts[:, l, :] @ r).numpy()
        b_proj = (harmless_acts[:, l, :] @ r).numpy()

        y_true = np.concatenate([np.ones(len(h_proj)), np.zeros(len(b_proj))])
        scores = np.concatenate([h_proj, b_proj])
        dim_auroc = float(roc_auc_score(y_true, scores))

        # 5-fold cross-validated logistic regression probe AUROC
        X = np.vstack([harmful_acts[:, l, :].numpy(), harmless_acts[:, l, :].numpy()])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_preds = np.zeros_like(y_true)
        for tr, va in skf.split(X, y_true):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=2000)
            )
            clf.fit(X[tr], y_true[tr])
            cv_preds[va] = clf.predict_proba(X[va])[:, 1]
        probe_auroc = float(roc_auc_score(y_true, cv_preds))

        cos_prev = float(torch.dot(unit_dirs[l], unit_dirs[l - 1]).item()) if l > 0 else 1.0

        row = {
            "layer": l,
            "diff_norm": float(dirs["diff_norms"][l].item()),
            "dim_auroc": dim_auroc,
            "probe_auroc": probe_auroc,
            "harm_proj_mean": float(h_proj.mean()),
            "harm_proj_std": float(h_proj.std()),
            "harmless_proj_mean": float(b_proj.mean()),
            "harmless_proj_std": float(b_proj.std()),
            "cos_with_prev_layer": cos_prev,
        }

        if over_refusal_acts is not None:
            or_proj = (over_refusal_acts[:, l, :] @ r).numpy()
            y_or = np.concatenate([np.ones(len(h_proj)), np.zeros(len(or_proj))])
            s_or = np.concatenate([h_proj, or_proj])
            row["or_proj_mean"] = float(or_proj.mean())
            row["or_proj_std"] = float(or_proj.std())
            row["harm_vs_or_dim_auroc"] = float(roc_auc_score(y_or, s_or))

        metrics.append(row)

    return metrics


def logit_lens_on_direction(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    direction: torch.Tensor,
    top_k: int = 15,
    apply_final_norm: bool = False
) -> Dict[str, List[Tuple[str, float]]]:
    """Project a residual-stream direction onto vocabulary logits via lm_head."""
    lm_head = get_lm_head(model)
    backbone, _ = get_text_backbone(model)
    w_u = lm_head.weight.detach().float().cpu()  # [vocab, d_model]

    vec = direction.detach().float().cpu()
    if apply_final_norm and hasattr(backbone, "norm"):
        norm_w = backbone.norm.weight.detach().float().cpu()
        rms = torch.sqrt(torch.mean(vec ** 2) + 1e-6)
        vec = (vec / rms) * (1.0 + norm_w)

    logits = w_u @ vec  # [vocab]
    top_pos_vals, top_pos_idx = torch.topk(logits, k=top_k)
    top_neg_vals, top_neg_idx = torch.topk(-logits, k=top_k)

    promoted = [(tokenizer.decode([int(idx)]), float(val)) for idx, val in zip(top_pos_idx, top_pos_vals)]
    suppressed = [(tokenizer.decode([int(idx)]), float(-val)) for idx, val in zip(top_neg_idx, top_neg_vals)]

    return {"promoted": promoted, "suppressed": suppressed}


@contextlib.contextmanager
def activation_steering_context(
    model: AutoModelForCausalLM,
    direction: torch.Tensor | Dict[int, torch.Tensor],
    layers_to_steer: List[int],
    mode: str = "ablate",
    alpha: float = 1.0,
    prefill_only: bool = False,
):
    """Context manager applying forward hooks on residual stream layer outputs.
    
    direction can be a single [D] tensor or a dict {layer_idx: [D] tensor}.
    mode='ablate': x <- x - alpha * (x . r_hat) r_hat
    mode='add':    x <- x + alpha * direction (or r_hat if unit)
    prefill_only:  apply steering only during prefill at the final prompt token ([-1]),
                   preventing token-by-token degeneration loops during autoregressive decoding.
    """
    _, layers = get_text_backbone(model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    handles = []

    def make_hook(l_idx: int):
        raw_d = direction[l_idx] if isinstance(direction, dict) else direction
        vec = raw_d.to(device=device, dtype=dtype)
        r_hat = (vec / vec.norm().clamp(min=1e-8))

        def hook_fn(module, inputs, outputs):
            is_tuple = isinstance(outputs, tuple)
            h = outputs[0] if is_tuple else outputs
            if prefill_only and h.shape[1] == 1:
                return outputs

            if mode == "ablate":
                if prefill_only:
                    h_new = h.clone()
                    proj = torch.einsum("bd,d->b", h[:, -1, :], r_hat).unsqueeze(-1)
                    h_new[:, -1, :] = h[:, -1, :] - alpha * proj * r_hat
                else:
                    proj = torch.einsum("...d,d->...", h, r_hat).unsqueeze(-1)
                    h_new = h - alpha * proj * r_hat
            elif mode == "add":
                delta = alpha * vec
                if prefill_only:
                    h_new = h.clone()
                    h_new[:, -1, :] = h_new[:, -1, :] + delta
                else:
                    h_new = h + delta
            else:
                raise ValueError(f"Unknown mode: {mode}")
            if is_tuple:
                return (h_new,) + outputs[1:]
            return h_new
        return hook_fn

    try:
        for l_idx in layers_to_steer:
            handles.append(layers[l_idx].register_forward_hook(make_hook(l_idx)))
        yield
    finally:
        for h in handles:
            h.remove()
