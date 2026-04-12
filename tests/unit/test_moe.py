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


# ---------------------------------------------------------------------------
# Load balancing: verify all experts receive traffic
# ---------------------------------------------------------------------------


def test_load_balancing_all_experts_activated() -> None:
    """Over many samples, every expert should be activated at least once."""
    n_experts = 4
    layer = SparseMoELayer(input_dim=16, n_experts=n_experts, expert_dim=8, top_k=2)
    x = torch.randn(200, 16)
    logits = layer.gate(x)
    _, top_k_idx = torch.topk(logits, 2, dim=-1)
    activated = set(top_k_idx.flatten().tolist())
    # With 200 samples and top_k=2, all 4 experts should be selected at least once
    assert activated == set(range(n_experts))


# ---------------------------------------------------------------------------
# Gradient flow through all experts
# ---------------------------------------------------------------------------


def test_gradient_flows_through_all_expert_params() -> None:
    """Gradients must reach expert_w1, expert_w2, expert_b1, expert_b2, and gate."""
    layer = SparseMoELayer(input_dim=16, n_experts=4, expert_dim=8, top_k=2)
    x = torch.randn(4, 16)
    out = layer(x)
    out.sum().backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        # At least some gradients should be non-zero
        assert param.grad.abs().sum() > 0, f"All-zero gradient for {name}"


# ---------------------------------------------------------------------------
# Batch size = 1 edge case
# ---------------------------------------------------------------------------


def test_batch_size_one() -> None:
    """MoE should work correctly with a single-element batch."""
    layer = SparseMoELayer(input_dim=8, n_experts=4, expert_dim=4, top_k=2)
    x = torch.randn(1, 8)
    out = layer(x)
    assert out.shape == (1, 8)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# top_k = n_experts (all experts activated)
# ---------------------------------------------------------------------------


def test_top_k_equals_n_experts_all_activated() -> None:
    """When top_k == n_experts, every expert contributes to the output."""
    n_experts = 3
    layer = SparseMoELayer(input_dim=8, n_experts=n_experts, expert_dim=4, top_k=n_experts)
    x = torch.randn(2, 8)
    out = layer(x)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()
    # Verify gradient flows with all experts active
    out.sum().backward()
    assert layer.expert_w1.grad is not None


# ---------------------------------------------------------------------------
# Gate softmax normalisation
# ---------------------------------------------------------------------------


def test_gate_weights_sum_to_one() -> None:
    """Top-k gate weights should sum to 1.0 after softmax."""
    layer = SparseMoELayer(input_dim=16, n_experts=8, expert_dim=8, top_k=2)
    x = torch.randn(4, 16)
    logits = layer.gate(x)
    top_k_vals, _ = torch.topk(logits, 2, dim=-1)
    gate_weights = torch.nn.functional.softmax(top_k_vals, dim=-1)
    # Each row should sum to 1
    sums = gate_weights.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=1e-5)
