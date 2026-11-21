"""Unit tests verifying Gemma-exact weight orthogonalization and norm-preservation algebra."""

import torch
import pytest

from gemma_safety.weight_surgery import (
    _to_orthonormal_basis,
    compute_postnorm_target_basis,
    orthogonalize_linear_weight,
)


def test_postnorm_exact_zero_leak():
    torch.manual_seed(0)
    d_model, d_in = 64, 128
    w = torch.randn(d_model, d_in)
    post_norm_w = torch.randn(d_model) * 0.5  # Gemma RMSNorm weight deviation from 1
    q = _to_orthonormal_basis(torch.randn(d_model, 2))

    # Naive orthogonalization leaves non-zero leak after post-norm scaling
    w_naive = orthogonalize_linear_weight(w, q, alpha=1.0, norm_mode="exact")
    g = (1.0 + post_norm_w).unsqueeze(1)
    leak_naive = (q.t() @ (g * w_naive)).norm() / (g * w_naive).norm()
    assert leak_naive > 0.01

    # Post-norm aware exact & col_mpoa orthogonalization give zero post-norm leak
    q_tilde = compute_postnorm_target_basis(q, post_norm_w)
    for mode in ("exact", "col_mpoa"):
        w_fixed = orthogonalize_linear_weight(w, q_tilde, alpha=1.0, norm_mode=mode)
        leak_fixed = (q.t() @ (g * w_fixed)).norm() / (g * w_fixed).norm()
        assert leak_fixed < 1e-5, f"Mode {mode} leaked {leak_fixed}"


def test_mpoa_norm_preservation():
    torch.manual_seed(1)
    d_model, d_in = 64, 128
    w = torch.randn(d_model, d_in)
    q = _to_orthonormal_basis(torch.randn(d_model))

    w_col = orthogonalize_linear_weight(w, q, alpha=1.0, norm_mode="col_mpoa")
    assert torch.allclose(w_col.norm(dim=0), w.norm(dim=0), atol=1e-5)

    w_row = orthogonalize_linear_weight(w, q, alpha=1.0, norm_mode="row_mpoa")
    assert torch.allclose(w_row.norm(dim=1), w.norm(dim=1), atol=1e-5)
