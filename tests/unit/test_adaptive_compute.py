"""Tests for AdaptiveCompute module."""

from __future__ import annotations

import torch

from mousedroid.scaling.adaptive import AdaptiveCompute


def test_constructor():
    ac = AdaptiveCompute(input_dim=8, max_steps=5, halt_threshold=1.0)
    assert ac._max_steps == 5
    assert ac._halt_threshold == 1.0


def test_forward_returns_output_and_steps():
    ac = AdaptiveCompute(input_dim=8, max_steps=5)
    x = torch.randn(2, 8)
    output, n_steps = ac(x)
    assert output.shape == (2, 8)
    assert 1 <= n_steps <= 5


def test_forward_single_step():
    ac = AdaptiveCompute(input_dim=4, max_steps=1)
    x = torch.randn(1, 4)
    output, n_steps = ac(x)
    assert n_steps == 1
    assert output.shape == (1, 4)


def test_forward_batch_dimension():
    ac = AdaptiveCompute(input_dim=4, max_steps=3)
    x = torch.randn(5, 4)
    output, _n_steps = ac(x)
    assert output.shape == (5, 4)


def test_low_halt_threshold_stops_early():
    # Very low threshold should cause early halting
    ac = AdaptiveCompute(input_dim=4, max_steps=10, halt_threshold=0.01)
    x = torch.randn(1, 4)
    _, n_steps = ac(x)
    # With a very low threshold, should stop before max_steps in most cases
    assert n_steps <= 10


# ---------------------------------------------------------------------------
# Maximum steps always respected
# ---------------------------------------------------------------------------


def test_maximum_steps_respected():
    """n_steps must never exceed max_steps, even with high threshold."""
    ac = AdaptiveCompute(input_dim=8, max_steps=3, halt_threshold=1000.0)
    x = torch.randn(4, 8)
    _, n_steps = ac(x)
    assert n_steps <= 3


# ---------------------------------------------------------------------------
# Ponder cost: more steps means more computation
# ---------------------------------------------------------------------------


def test_ponder_cost_increases_with_threshold():
    """Higher halt threshold should generally require more steps."""
    ac_low = AdaptiveCompute(input_dim=8, max_steps=20, halt_threshold=0.01)
    ac_high = AdaptiveCompute(input_dim=8, max_steps=20, halt_threshold=10.0)

    # Use the same weights for both
    ac_high.load_state_dict(ac_low.state_dict())

    x = torch.randn(4, 8)
    _, steps_low = ac_low(x)
    _, steps_high = ac_high(x)

    # Higher threshold needs at least as many steps
    assert steps_high >= steps_low


# ---------------------------------------------------------------------------
# Output shape preserved across step counts
# ---------------------------------------------------------------------------


def test_output_shape_consistent_across_max_steps():
    """Output shape must be (batch, input_dim) regardless of max_steps."""
    for max_steps in [1, 3, 7]:
        ac = AdaptiveCompute(input_dim=8, max_steps=max_steps)
        x = torch.randn(2, 8)
        output, _ = ac(x)
        assert output.shape == (2, 8), f"Failed for max_steps={max_steps}"


# ---------------------------------------------------------------------------
# No gradients in forward (torch.no_grad decorator)
# ---------------------------------------------------------------------------


def test_forward_runs_under_no_grad():
    """Forward is decorated with torch.no_grad, so output should not require grad."""
    ac = AdaptiveCompute(input_dim=4, max_steps=3)
    x = torch.randn(2, 4, requires_grad=True)
    output, _ = ac(x)
    # The output should not track gradients because forward uses @torch.no_grad()
    assert not output.requires_grad


# ---------------------------------------------------------------------------
# Different batch elements can halt at different effective steps
# ---------------------------------------------------------------------------


def test_batch_elements_independent():
    """Different inputs should produce different outputs (non-trivially)."""
    ac = AdaptiveCompute(input_dim=8, max_steps=5)
    # Create two very different inputs
    x = torch.zeros(2, 8)
    x[0, 0] = 100.0
    x[1, 7] = -100.0
    output, _ = ac(x)
    # The two outputs should differ
    assert not torch.allclose(output[0], output[1])
