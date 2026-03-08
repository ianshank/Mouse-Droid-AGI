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
