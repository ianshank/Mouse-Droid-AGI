from __future__ import annotations

import pytest
import torch

from mousedroid.config.schema import CuriosityConfig, ModelConfig
from mousedroid.curiosity.icm import IntrinsicCuriosityModule


@pytest.fixture
def icm() -> IntrinsicCuriosityModule:
    return IntrinsicCuriosityModule(ModelConfig(), CuriosityConfig())


def test_constructor(icm: IntrinsicCuriosityModule) -> None:
    assert hasattr(icm, "forward_model")
    assert hasattr(icm, "inverse_model")


def test_forward_returns_three_tensors(icm: IntrinsicCuriosityModule) -> None:
    s = torch.randn(2, 256)
    a = torch.randn(2, 3)
    s_next = torch.randn(2, 256)
    fwd_loss, inv_loss, pred = icm(s, a, s_next)
    assert fwd_loss.shape == ()
    assert inv_loss.shape == ()
    assert pred.shape == (2, 256)


def test_intrinsic_reward_output_shape(icm: IntrinsicCuriosityModule) -> None:
    s = torch.randn(4, 256)
    a = torch.randn(4, 3)
    s_next = torch.randn(4, 256)
    reward = icm.intrinsic_reward(s, a, s_next)
    assert reward.shape == (4,)


def test_intrinsic_reward_non_negative(icm: IntrinsicCuriosityModule) -> None:
    s = torch.randn(3, 256)
    a = torch.randn(3, 3)
    s_next = torch.randn(3, 256)
    reward = icm.intrinsic_reward(s, a, s_next)
    assert (reward >= 0.0).all()


def test_scale_factor_applied() -> None:
    cfg_c = CuriosityConfig(intrinsic_reward_scale=10.0)
    icm_scaled = IntrinsicCuriosityModule(ModelConfig(), cfg_c)
    cfg_c2 = CuriosityConfig(intrinsic_reward_scale=1.0)
    icm_base = IntrinsicCuriosityModule(ModelConfig(), cfg_c2)

    # Can't directly compare different models, but verify scaled version has _scale=10
    assert icm_scaled._scale == 10.0
    assert icm_base._scale == 1.0


def test_is_nn_module(icm: IntrinsicCuriosityModule) -> None:
    assert isinstance(icm, torch.nn.Module)
