"""Tests for MAML adapter."""

from __future__ import annotations

import torch
import torch.nn as nn

from mousedroid.meta.maml import MAMLAdapter


def _make_model() -> nn.Module:
    return nn.Linear(4, 2)


def test_constructor():
    model = _make_model()
    adapter = MAMLAdapter(model, inner_lr=0.01, outer_lr=0.001, inner_steps=3)
    assert adapter._inner_lr == 0.01
    assert adapter._inner_steps == 3


def test_adapt_returns_different_model():
    model = _make_model()
    adapter = MAMLAdapter(model, inner_lr=0.01, outer_lr=0.001, inner_steps=1)

    x = torch.randn(2, 4)
    y = torch.randn(2, 2)
    support = [(x, y)]

    adapted = adapter.adapt(support, nn.MSELoss())
    assert adapted is not model


def test_adapt_modifies_copy_not_original():
    model = _make_model()
    original_weight = model.weight.data.clone()
    adapter = MAMLAdapter(model, inner_lr=0.1, outer_lr=0.001, inner_steps=3)

    x = torch.randn(4, 4)
    y = torch.randn(4, 2)
    support = [(x, y)]

    adapter.adapt(support, nn.MSELoss())
    # Original model should be unchanged
    assert torch.allclose(model.weight.data, original_weight)


def test_meta_step_returns_float():
    model = _make_model()
    adapter = MAMLAdapter(model, inner_lr=0.01, outer_lr=0.001, inner_steps=1)

    x_s = torch.randn(2, 4)
    y_s = torch.randn(2, 2)
    x_q = torch.randn(2, 4)
    y_q = torch.randn(2, 2)

    tasks = [([(x_s, y_s)], [(x_q, y_q)])]
    loss = adapter.meta_step(tasks, nn.MSELoss())
    assert isinstance(loss, float)
    assert loss >= 0.0
