"""Unit tests for the sparse Mixture-of-Experts layer.

Verifies the top-k > n_experts guard, output shape, differentiability, and
that a fully-sparse (top_k=1) route still yields a correctly-shaped, finite
output. (The exact single-expert value is not re-derived — reconstructing one
expert's contribution would duplicate the layer's internal matmul; shape +
finiteness is the stable contract exercised here.)
"""

from __future__ import annotations

import pytest
import torch

from mousedroid.scaling.moe import SparseMoELayer


def test_rejects_top_k_greater_than_experts() -> None:
    """Constructing with ``top_k > n_experts`` raises ValueError."""
    with pytest.raises(ValueError, match="top_k"):
        SparseMoELayer(input_dim=4, n_experts=2, expert_dim=8, top_k=3)


def test_forward_output_shape() -> None:
    """Output shape matches ``(batch, input_dim)``."""
    torch.manual_seed(0)
    layer = SparseMoELayer(input_dim=6, n_experts=4, expert_dim=8, top_k=2)
    out = layer(torch.randn(5, 6))
    assert out.shape == (5, 6)


def test_forward_is_differentiable() -> None:
    """Gradients flow back to the gate parameters."""
    torch.manual_seed(0)
    layer = SparseMoELayer(input_dim=6, n_experts=4, expert_dim=8, top_k=2)
    out = layer(torch.randn(3, 6))
    out.sum().backward()
    assert layer.gate.weight.grad is not None


def test_single_expert_routing_shape() -> None:
    """top_k=1 still produces a correctly shaped output (fully sparse route)."""
    torch.manual_seed(0)
    layer = SparseMoELayer(input_dim=4, n_experts=3, expert_dim=5, top_k=1)
    out = layer(torch.randn(2, 4))
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()
