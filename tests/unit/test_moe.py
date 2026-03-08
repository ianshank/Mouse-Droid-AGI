from __future__ import annotations

import pytest
import torch

from mousedroid.scaling.moe import SparseMoELayer


def test_constructor() -> None:
    layer = SparseMoELayer(input_dim=16, n_experts=4, expert_dim=8, top_k=2)
    assert layer._n_experts == 4
    assert layer._top_k == 2


def test_top_k_exceeds_n_experts_raises() -> None:
    with pytest.raises(ValueError, match=r"top_k.*must not exceed"):
        SparseMoELayer(input_dim=16, n_experts=4, expert_dim=8, top_k=5)


def test_top_k_equals_n_experts_ok() -> None:
    layer = SparseMoELayer(input_dim=16, n_experts=4, expert_dim=8, top_k=4)
    assert layer._top_k == 4


def test_forward_output_shape() -> None:
    layer = SparseMoELayer(input_dim=16, n_experts=4, expert_dim=8, top_k=2)
    x = torch.randn(3, 16)
    out = layer(x)
    assert out.shape == (3, 16)


def test_forward_single_batch() -> None:
    layer = SparseMoELayer(input_dim=8, n_experts=3, expert_dim=4, top_k=1)
    x = torch.randn(1, 8)
    out = layer(x)
    assert out.shape == (1, 8)


def test_sparse_activation() -> None:
    layer = SparseMoELayer(input_dim=16, n_experts=8, expert_dim=8, top_k=2)
    x = torch.randn(4, 16)
    logits = layer.gate(x)
    _, top_k_idx = torch.topk(logits, 2, dim=-1)
    # Each token uses only 2 out of 8 experts
    assert top_k_idx.shape == (4, 2)
    for row in top_k_idx:
        assert len(set(row.tolist())) <= 2


def test_forward_is_differentiable() -> None:
    layer = SparseMoELayer(input_dim=16, n_experts=4, expert_dim=8, top_k=2)
    x = torch.randn(2, 16, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None


def test_vectorized_dispatch_matches_shape() -> None:
    layer = SparseMoELayer(input_dim=32, n_experts=6, expert_dim=16, top_k=3)
    x = torch.randn(5, 32)
    out = layer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
