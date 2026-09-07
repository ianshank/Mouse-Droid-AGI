"""F-040 backwards-compat: imu_dim=0 keeps fused_dim and motor_state_dim."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from mousedroid.config.schema import ModelConfig, Settings
from mousedroid.world_model.encoder import MultimodalEncoder

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_model_config_imu_off() -> None:
    cfg = ModelConfig()
    assert cfg.imu_dim == 0
    assert cfg.imu_proj_dim == 32
    assert cfg.motor_state_dim == 4
    assert cfg.lidar_dim == 0


def test_default_encoder_fusion_in_features_unchanged() -> None:
    """imu_dim=0 must not resize encoder.fusion (checkpoint-compatible)."""
    cfg = ModelConfig()
    enc = MultimodalEncoder(cfg)
    # vision_proj 128 + ultrasonic 32 + motor 32 = 192 when defaults apply.
    assert enc.fusion.in_features == (
        cfg.vision_proj_dim + cfg.ultrasonic_proj_dim + cfg.motor_proj_dim
    )


def test_enabled_imu_grows_fusion_by_proj_dim() -> None:
    cfg = ModelConfig(imu_dim=3, imu_proj_dim=8)
    enc = MultimodalEncoder(cfg)
    baseline = ModelConfig()
    base_enc = MultimodalEncoder(baseline)
    assert enc.fusion.in_features == base_enc.fusion.in_features + 8
    batch = 2
    out = enc(
        torch.randn(batch, cfg.vision_dim),
        torch.randn(batch, cfg.ultrasonic_dim),
        torch.randn(batch, cfg.motor_state_dim),
        torch.ones(batch, 6),
        imu=torch.randn(batch, 3),
    )
    assert out.shape == (batch, cfg.obs_dim)


@pytest.mark.parametrize(
    "overlay",
    sorted(p.name for p in (_REPO_ROOT / "config").glob("*.yaml")),
)
def test_shipped_overlays_still_load_with_imu_off(overlay: str) -> None:
    path = _REPO_ROOT / "config" / overlay
    text = path.read_text(encoding="utf-8")
    if "# config-validator: skip" in text:
        pytest.skip(f"{overlay} carries skip marker (not a Settings overlay)")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        return
    cfg = Settings.model_validate({**raw, "mock_hardware": True})
    assert cfg.model.imu_dim == 0
    assert cfg.model.motor_state_dim == 4
