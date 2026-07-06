"""Unit tests for the adaptive-compute (ponder) module.

Verifies output shape, that the actual step count never exceeds the configured
maximum, and that a very low halt threshold causes early stopping.
"""

from __future__ import annotations

import torch

from mousedroid.scaling.adaptive import AdaptiveCompute


def test_forward_output_shape_and_step_bound() -> None:
    """Output preserves input shape and step count is within ``max_steps``."""
    torch.manual_seed(0)
    module = AdaptiveCompute(input_dim=8, max_steps=5, halt_threshold=1.0)
    x = torch.randn(4, 8)

    output, n_steps = module(x)

    assert output.shape == x.shape
    assert 1 <= n_steps <= 5


def test_low_threshold_halts_early() -> None:
    """A near-zero halt threshold stops after a single pondering step."""
    torch.manual_seed(0)
    module = AdaptiveCompute(input_dim=8, max_steps=6, halt_threshold=1e-6)
    output, n_steps = module(torch.randn(2, 8))
    assert n_steps == 1
    assert output.shape == (2, 8)


def test_forward_runs_under_no_grad() -> None:
    """The ``@torch.no_grad()`` forward detaches output even from a grad input.

    The input requires grad AND the module's ``nn.Linear`` params require grad,
    so without the decorator the output would land on the autograd graph
    (``requires_grad=True``). Asserting it is ``False`` therefore genuinely
    guards the decorator — remove ``@torch.no_grad()`` and this fails.
    """
    module = AdaptiveCompute(input_dim=4, max_steps=3)
    x = torch.randn(1, 4, requires_grad=True)
    output, _ = module(x)
    assert not output.requires_grad
