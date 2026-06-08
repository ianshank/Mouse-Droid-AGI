"""Vision branch is optional and gated on vision_dim, mirroring audio/lidar."""

from __future__ import annotations

import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.encoder import MultimodalEncoder


def _cfg(**over: object) -> ModelConfig:
    return ModelConfig(**over)  # type: ignore[arg-type]


def test_vision_enabled_by_default() -> None:
    enc = MultimodalEncoder(_cfg())
    assert enc.vision_enabled is True
    assert hasattr(enc, "vision_proj")


def test_vision_disabled_when_vision_dim_zero() -> None:
    enc = MultimodalEncoder(_cfg(vision_dim=0, vision_proj_dim=0))
    assert enc.vision_enabled is False
    assert not hasattr(enc, "vision_proj")


def test_forward_without_vision_runs() -> None:
    cfg = _cfg(vision_dim=0, vision_proj_dim=0)
    model = MultimodalEncoder(cfg)
    motor = torch.zeros(2, cfg.motor_state_dim)
    mask = torch.ones(2, 5)
    out = model(None, None, motor, mask)
    assert out.shape == (2, cfg.obs_dim)


def test_vision_enabled_but_none_raises() -> None:
    import pytest

    cfg = _cfg()  # vision enabled by default
    enc = MultimodalEncoder(cfg)
    motor = torch.zeros(1, cfg.motor_state_dim)
    mask = torch.ones(1, 5)
    with pytest.raises(ValueError, match="vision tensor"):
        enc(None, None, motor, mask)


def test_default_forward_byte_identical_path() -> None:
    cfg = _cfg()
    enc = MultimodalEncoder(cfg)
    vision = torch.zeros(1, cfg.vision_dim)
    motor = torch.zeros(1, cfg.motor_state_dim)
    mask = torch.ones(1, 5)
    out = enc(vision, None, motor, mask)
    assert out.shape == (1, cfg.obs_dim)
