"""Tests for TensorRTOptimizer."""

from __future__ import annotations

import torch
import torch.nn as nn

from mousedroid.config.schema import JetsonConfig
from mousedroid.efficiency.tensorrt import TensorRTOptimizer


def test_constructor_default():
    cfg = JetsonConfig()
    opt = TensorRTOptimizer(cfg)
    assert opt.enabled is True
    assert opt.precision == "fp16"


def test_constructor_disabled():
    cfg = JetsonConfig(tensorrt_enabled=False)
    opt = TensorRTOptimizer(cfg)
    assert opt.enabled is False


def test_optimize_disabled_returns_original_model():
    cfg = JetsonConfig(tensorrt_enabled=False)
    opt = TensorRTOptimizer(cfg)
    model = nn.Linear(4, 2)
    sample = torch.randn(1, 4)
    result = opt.optimize(model, sample)
    assert result is model


def test_optimizer_stores_config():
    cfg = JetsonConfig(
        tensorrt_enabled=True,
        precision="int8",
        workspace_gb=2.0,
        dla_enabled=True,
    )
    opt = TensorRTOptimizer(cfg)
    assert opt.precision == "int8"
    assert opt.workspace_gb == 2.0
    assert opt.dla_enabled is True
