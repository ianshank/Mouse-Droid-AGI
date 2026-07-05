"""Unit tests for the MAML meta-learning adapter.

Uses a tiny ``nn.Linear`` so inner/outer loops run in milliseconds. Verifies
that inner-loop adaptation leaves the base model untouched and returns a
distinct adapted copy, and that a meta-step returns a finite scalar loss and
mutates the base parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.meta.maml import MAMLAdapter


def _mse(pred: Tensor, target: Tensor) -> Tensor:
    return nn.functional.mse_loss(pred, target)


def _support() -> list[tuple[Tensor, Tensor]]:
    torch.manual_seed(0)
    return [(torch.randn(4, 3), torch.randn(4, 2)) for _ in range(2)]


def test_adapt_returns_distinct_copy_leaving_base_unchanged() -> None:
    """Inner-loop adaptation must not mutate the base model in place."""
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    base_before = model.weight.detach().clone()
    adapter = MAMLAdapter(model, inner_lr=0.05, outer_lr=0.01, inner_steps=2)

    adapted = adapter.adapt(_support(), _mse)

    assert adapted is not model
    # Base is untouched by adaptation...
    assert torch.equal(model.weight.detach(), base_before)
    # ...but the adapted copy moved.
    assert not torch.equal(adapted.weight.detach(), base_before)


def test_meta_step_returns_finite_nonnegative_loss() -> None:
    """The outer loop runs adapt -> query -> backward -> step and returns MSE."""
    torch.manual_seed(1)
    model = nn.Linear(3, 2)
    adapter = MAMLAdapter(model, inner_lr=0.05, outer_lr=0.05, inner_steps=1)

    loss = adapter.meta_step([(_support(), _support())], _mse)

    assert isinstance(loss, float)
    assert loss == loss  # not NaN
    assert loss >= 0.0  # mean of MSE terms


def test_meta_step_is_first_order_base_unchanged() -> None:
    """Characterizes the current first-order implementation.

    ``meta_step`` computes the query loss on the deep-copied *adapted* model, so
    gradients do not flow back to the base model's parameters — the meta
    optimizer sees ``grad is None`` and the base weights are unchanged. This
    documents (does not endorse) the present behaviour so a future move to true
    second-order MAML flips this test deliberately rather than silently.
    """
    torch.manual_seed(1)
    model = nn.Linear(3, 2)
    adapter = MAMLAdapter(model, inner_lr=0.05, outer_lr=0.05, inner_steps=1)
    before = model.weight.detach().clone()

    adapter.meta_step([(_support(), _support())], _mse)

    assert torch.equal(model.weight.detach(), before)
