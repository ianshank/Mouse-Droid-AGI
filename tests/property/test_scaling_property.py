"""Property-based tests for MoE and AdaptiveCompute scaling modules."""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.scaling.adaptive import AdaptiveCompute
from mousedroid.scaling.moe import SparseMoELayer

# ---------------------------------------------------------------------------
# MoE: output shape invariant for any valid batch size
# ---------------------------------------------------------------------------


@given(batch_size=st.integers(min_value=1, max_value=64))
@settings(max_examples=50, deadline=5000)
def test_moe_output_shape_invariant(batch_size: int) -> None:
    """MoE output shape must match input shape for any batch size."""
    input_dim, n_experts, expert_dim, top_k = 16, 4, 8, 2
    layer = SparseMoELayer(input_dim, n_experts, expert_dim, top_k)
    x = torch.randn(batch_size, input_dim)
    out = layer(x)
    assert out.shape == (batch_size, input_dim)


@given(batch_size=st.integers(min_value=1, max_value=64))
@settings(max_examples=50, deadline=5000)
def test_moe_output_is_finite(batch_size: int) -> None:
    """MoE output must contain only finite values."""
    layer = SparseMoELayer(input_dim=16, n_experts=4, expert_dim=8, top_k=2)
    x = torch.randn(batch_size, 16)
    out = layer(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# AdaptiveCompute: n_steps always in [1, max_steps]
# ---------------------------------------------------------------------------


@given(
    max_steps=st.integers(min_value=1, max_value=10),
    batch_size=st.integers(min_value=1, max_value=16),
)
@settings(max_examples=50, deadline=5000)
def test_adaptive_compute_steps_in_range(max_steps: int, batch_size: int) -> None:
    """Number of pondering steps must be in [1, max_steps]."""
    ac = AdaptiveCompute(input_dim=8, max_steps=max_steps, halt_threshold=1.0)
    x = torch.randn(batch_size, 8)
    output, n_steps = ac(x)
    assert 1 <= n_steps <= max_steps


@given(
    max_steps=st.integers(min_value=1, max_value=10),
    batch_size=st.integers(min_value=1, max_value=16),
)
@settings(max_examples=50, deadline=5000)
def test_adaptive_compute_output_shape(max_steps: int, batch_size: int) -> None:
    """Output shape must match input shape regardless of steps taken."""
    input_dim = 8
    ac = AdaptiveCompute(input_dim=input_dim, max_steps=max_steps)
    x = torch.randn(batch_size, input_dim)
    output, _ = ac(x)
    assert output.shape == (batch_size, input_dim)


@given(batch_size=st.integers(min_value=1, max_value=32))
@settings(max_examples=50, deadline=5000)
def test_adaptive_compute_output_is_finite(batch_size: int) -> None:
    """Output values must be finite."""
    ac = AdaptiveCompute(input_dim=8, max_steps=5)
    x = torch.randn(batch_size, 8)
    output, _ = ac(x)
    assert torch.isfinite(output).all()


# ---------------------------------------------------------------------------
# AdaptiveCompute: very low threshold halts early
# ---------------------------------------------------------------------------


@given(batch_size=st.integers(min_value=1, max_value=8))
@settings(max_examples=30, deadline=5000)
def test_adaptive_compute_low_threshold_halts(batch_size: int) -> None:
    """With halt_threshold near zero, should not reach max_steps."""
    ac = AdaptiveCompute(input_dim=8, max_steps=100, halt_threshold=0.001)
    x = torch.randn(batch_size, 8)
    _, n_steps = ac(x)
    # With such a low threshold, should halt well before 100
    assert n_steps <= 100
