"""Regression tests — RSSM checkpoint migration across modality configurations.

Three migration scenarios are covered:

1. ``u+m → l+m``   — drop ultrasonic, add LiDAR (most common retraining path)
2. ``u+a+m → l+a+m`` — drop ultrasonic, keep audio, add LiDAR
3. Same-config   — identity / no-op migration preserves all weights exactly
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.checkpoint_migration import (
    load_rssm_with_migration,
    migrate_state_dict,
)
from mousedroid.world_model.rssm import RSSM

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _small_cfg(**overrides: int | float | str) -> ModelConfig:
    """Minimal ``ModelConfig`` with small dims for fast tests."""
    defaults: dict[str, int | float | str] = {
        "vision_dim": 8,
        "vision_proj_dim": 16,
        "ultrasonic_dim": 1,
        "ultrasonic_proj_dim": 8,
        "motor_state_dim": 4,
        "motor_proj_dim": 8,
        "hidden_dim": 32,
        "latent_dim": 16,
        "action_dim": 3,
        "obs_dim": 32,
        "audio_dim": 0,
        "audio_proj_dim": 8,
        "lidar_dim": 0,
        "lidar_proj_dim": 8,
        "belief_dim": 16,
        "desire_dim": 8,
        "intention_classes": 4,
        "affect_dim": 2,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


def _rssm_sd(cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """State dict from a freshly initialised RSSM."""
    return RSSM(cfg).state_dict()


# ---------------------------------------------------------------------------
# Scenario 1: ultrasonic + motor  →  lidar + motor
# ---------------------------------------------------------------------------


class TestUltrasonicToLidar:
    """Drop ultrasonic, add LiDAR — the primary Cut 2 migration path."""

    def setup_method(self) -> None:
        torch.manual_seed(42)
        self.old_cfg = _small_cfg(
            ultrasonic_dim=1,
            ultrasonic_proj_dim=8,
            lidar_dim=0,
            lidar_proj_dim=8,
        )
        self.new_cfg = _small_cfg(
            ultrasonic_dim=0,
            ultrasonic_proj_dim=0,
            lidar_dim=6,
            lidar_proj_dim=8,
        )
        self.old_sd = _rssm_sd(self.old_cfg)

    def test_ultrasonic_keys_removed(self) -> None:
        migrated, _ = migrate_state_dict(self.old_sd, self.new_cfg)
        assert "encoder.ultrasonic_proj.weight" not in migrated
        assert "encoder.ultrasonic_proj.bias" not in migrated

    def test_lidar_keys_added(self) -> None:
        migrated, _ = migrate_state_dict(self.old_sd, self.new_cfg)
        assert "encoder.lidar_proj.weight" in migrated
        assert "encoder.lidar_proj.bias" in migrated

    def test_lidar_proj_weight_shape(self) -> None:
        migrated, _ = migrate_state_dict(self.old_sd, self.new_cfg)
        # nn.Linear(lidar_dim=6, lidar_proj_dim=8) → weight shape (8, 6)
        assert migrated["encoder.lidar_proj.weight"].shape == (8, 6)

    def test_fusion_weight_shape(self) -> None:
        migrated, _ = migrate_state_dict(self.old_sd, self.new_cfg)
        # Old: vision(16) + ultrasonic(8) + motor(8) = 32
        # New: vision(16) + motor(8) + lidar(8) = 32
        assert self.old_sd["encoder.fusion.weight"].shape[1] == 32
        assert migrated["encoder.fusion.weight"].shape[1] == 32

    def test_vision_motor_columns_preserved(self) -> None:
        """Vision and motor column slices must be copied verbatim."""
        migrated, _ = migrate_state_dict(self.old_sd, self.new_cfg)
        old_fw = self.old_sd["encoder.fusion.weight"]
        new_fw = migrated["encoder.fusion.weight"]

        # Old column layout: [vision(16) | ultrasonic(8) | motor(8)]
        old_vision = old_fw[:, :16]
        old_motor = old_fw[:, 24:32]  # after vision+ultrasonic

        # New column layout: [vision(16) | motor(8) | lidar(8)]
        new_vision = new_fw[:, :16]
        new_motor = new_fw[:, 16:24]  # after vision

        assert torch.allclose(old_vision, new_vision)
        assert torch.allclose(old_motor, new_motor)

    def test_report_fields(self) -> None:
        _, report = migrate_state_dict(self.old_sd, self.new_cfg)
        assert "ultrasonic" in report["dropped_modalities"]  # type: ignore[operator]
        assert "lidar" in report["added_modalities"]  # type: ignore[operator]
        assert report["old_fusion_shape"] == [32, 32]
        assert report["new_fusion_shape"] == [32, 32]

    def test_load_into_new_rssm_no_error(self) -> None:
        migrated, _ = migrate_state_dict(self.old_sd, self.new_cfg)
        rssm = RSSM(self.new_cfg)
        rssm.load_state_dict(migrated)  # must not raise

    def test_migrated_rssm_forward_pass(self) -> None:
        migrated, _ = migrate_state_dict(self.old_sd, self.new_cfg)
        rssm = RSSM(self.new_cfg)
        rssm.load_state_dict(migrated)
        rssm.eval()
        batch = 2
        with torch.no_grad():
            vision = torch.randn(batch, self.new_cfg.vision_dim)
            motor = torch.randn(batch, self.new_cfg.motor_state_dim)
            lidar = torch.randn(batch, self.new_cfg.lidar_dim)
            valid = torch.ones(batch, 5)
            obs = rssm.encoder(vision, None, motor, valid, lidar=lidar)
        assert obs.shape == (batch, self.new_cfg.obs_dim)
        assert torch.isfinite(obs).all()

    def test_load_rssm_with_migration_bare_state_dict(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "rssm_bare.pt"
        torch.save(self.old_sd, ckpt)
        rssm = load_rssm_with_migration(ckpt, self.new_cfg)
        assert isinstance(rssm, RSSM)
        assert not rssm.encoder.ultrasonic_enabled
        assert rssm.encoder.lidar_enabled

    def test_load_rssm_with_migration_full_checkpoint(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "rssm_full.pt"
        full_ckpt = {"model_state_dict": self.old_sd, "epoch": 10, "best_loss": 0.42}
        torch.save(full_ckpt, ckpt)
        rssm = load_rssm_with_migration(ckpt, self.new_cfg)
        assert isinstance(rssm, RSSM)
        assert rssm.encoder.lidar_enabled


# ---------------------------------------------------------------------------
# Scenario 2: ultrasonic + audio + motor  →  lidar + audio + motor
# ---------------------------------------------------------------------------


class TestUltrasonicAudioToLidarAudio:
    """Drop ultrasonic, keep audio, add LiDAR.  Audio columns must be preserved."""

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.old_cfg = _small_cfg(
            ultrasonic_dim=1,
            ultrasonic_proj_dim=8,
            audio_dim=4,
            audio_proj_dim=8,
            lidar_dim=0,
            lidar_proj_dim=8,
        )
        self.new_cfg = _small_cfg(
            ultrasonic_dim=0,
            ultrasonic_proj_dim=0,
            audio_dim=4,
            audio_proj_dim=8,
            lidar_dim=6,
            lidar_proj_dim=8,
        )
        self.old_sd = _rssm_sd(self.old_cfg)

    def test_audio_columns_preserved_in_fusion(self) -> None:
        """Audio projection columns must be identical in old and new weights."""
        migrated, _ = migrate_state_dict(self.old_sd, self.new_cfg)
        old_fw = self.old_sd["encoder.fusion.weight"]
        new_fw = migrated["encoder.fusion.weight"]

        # Old layout: [vision(16) | ultrasonic(8) | motor(8) | audio(8)]
        audio_old_start = 16 + 8 + 8  # = 32
        old_audio_cols = old_fw[:, audio_old_start : audio_old_start + 8]

        # New layout: [vision(16) | motor(8) | audio(8) | lidar(8)]
        audio_new_start = 16 + 8  # = 24
        new_audio_cols = new_fw[:, audio_new_start : audio_new_start + 8]

        assert torch.allclose(old_audio_cols, new_audio_cols)

    def test_new_fusion_weight_shape(self) -> None:
        migrated, _ = migrate_state_dict(self.old_sd, self.new_cfg)
        # New: vision(16) + motor(8) + audio(8) + lidar(8) = 40
        assert migrated["encoder.fusion.weight"].shape[1] == 16 + 8 + 8 + 8

    def test_old_fusion_weight_shape(self) -> None:
        # Old: vision(16) + ultrasonic(8) + motor(8) + audio(8) = 40
        assert self.old_sd["encoder.fusion.weight"].shape[1] == 16 + 8 + 8 + 8

    def test_load_and_forward(self) -> None:
        migrated, _ = migrate_state_dict(self.old_sd, self.new_cfg)
        rssm = RSSM(self.new_cfg)
        rssm.load_state_dict(migrated)
        rssm.eval()
        batch = 3
        with torch.no_grad():
            vision = torch.randn(batch, self.new_cfg.vision_dim)
            motor = torch.randn(batch, self.new_cfg.motor_state_dim)
            audio = torch.randn(batch, self.new_cfg.audio_dim)
            lidar = torch.randn(batch, self.new_cfg.lidar_dim)
            valid = torch.ones(batch, 5)
            obs = rssm.encoder(vision, None, motor, valid, audio=audio, lidar=lidar)
        assert obs.shape == (batch, self.new_cfg.obs_dim)
        assert torch.isfinite(obs).all()

    def test_report_lists_both_changes(self) -> None:
        _, report = migrate_state_dict(self.old_sd, self.new_cfg)
        assert "ultrasonic" in report["dropped_modalities"]  # type: ignore[operator]
        assert "lidar" in report["added_modalities"]  # type: ignore[operator]
        # Audio must NOT appear in either list.
        assert "audio" not in report["dropped_modalities"]  # type: ignore[operator]
        assert "audio" not in report["added_modalities"]  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Scenario 3: same config  →  identity / no-op migration
# ---------------------------------------------------------------------------


class TestNoOpMigration:
    """Migrating to the identical config must leave all weights untouched."""

    def setup_method(self) -> None:
        torch.manual_seed(7)
        # Use lidar-only to satisfy the at-least-one-distance-modality constraint.
        self.cfg = _small_cfg(
            ultrasonic_dim=0,
            ultrasonic_proj_dim=0,
            lidar_dim=6,
            lidar_proj_dim=8,
        )
        self.old_sd = _rssm_sd(self.cfg)

    def test_fusion_weight_unchanged(self) -> None:
        migrated, _ = migrate_state_dict(self.old_sd, self.cfg)
        assert torch.allclose(
            self.old_sd["encoder.fusion.weight"],
            migrated["encoder.fusion.weight"],
        )

    def test_all_keys_identical(self) -> None:
        migrated, _ = migrate_state_dict(self.old_sd, self.cfg)
        assert set(migrated.keys()) == set(self.old_sd.keys())

    def test_report_has_no_changes(self) -> None:
        _, report = migrate_state_dict(self.old_sd, self.cfg)
        assert report["dropped_modalities"] == []
        assert report["added_modalities"] == []
        assert report["old_fusion_shape"] == report["new_fusion_shape"]


# ---------------------------------------------------------------------------
# Edge-case: incompatible projection dim raises ValueError
# ---------------------------------------------------------------------------


def test_incompatible_proj_dim_raises() -> None:
    """Migration must raise if a retained modality has changed proj_dim."""
    old_cfg = _small_cfg(
        ultrasonic_dim=0,
        ultrasonic_proj_dim=0,
        lidar_dim=6,
        lidar_proj_dim=8,
    )
    # New config shrinks motor_proj_dim from 8 → 4, which is incompatible.
    new_cfg = _small_cfg(
        ultrasonic_dim=0,
        ultrasonic_proj_dim=0,
        lidar_dim=6,
        lidar_proj_dim=8,
        motor_proj_dim=4,
        # obs_dim must accommodate smaller fusion (adjust to keep RSSM valid)
        obs_dim=32,
    )
    old_sd = _rssm_sd(old_cfg)
    with pytest.raises(ValueError, match="Projection dim mismatch"):
        migrate_state_dict(old_sd, new_cfg)


# ---------------------------------------------------------------------------
# Edge-case: bad checkpoint type raises TypeError
# ---------------------------------------------------------------------------


def test_load_rssm_bad_checkpoint_type(tmp_path: Path) -> None:
    ckpt = tmp_path / "bad.pt"
    torch.save([1, 2, 3], ckpt)  # list, not dict
    cfg = _small_cfg(
        ultrasonic_dim=0,
        ultrasonic_proj_dim=0,
        lidar_dim=6,
        lidar_proj_dim=8,
    )
    with pytest.raises(TypeError, match="expected dict"):
        load_rssm_with_migration(ckpt, cfg)


# ---------------------------------------------------------------------------
# Coverage gap: _build_new_parts ultrasonic branch (line 85)
# ---------------------------------------------------------------------------


def test_build_new_parts_includes_ultrasonic() -> None:
    """migrate_state_dict must handle new_cfg with ultrasonic enabled (adds ultrasonic columns)."""
    old_cfg = _small_cfg(
        ultrasonic_dim=0,
        ultrasonic_proj_dim=0,
        lidar_dim=6,
        lidar_proj_dim=8,
    )
    new_cfg = _small_cfg(
        ultrasonic_dim=1,
        ultrasonic_proj_dim=8,
        lidar_dim=0,
        lidar_proj_dim=8,
    )
    old_sd = _rssm_sd(old_cfg)
    migrated, report = migrate_state_dict(old_sd, new_cfg)
    assert "ultrasonic" in report["added_modalities"]
    assert "encoder.ultrasonic_proj.weight" in migrated
    assert "encoder.ultrasonic_proj.bias" in migrated


# ---------------------------------------------------------------------------
# Coverage gap: _new_proj_tensors unknown modality raises ValueError (line 170)
# ---------------------------------------------------------------------------


def test_new_proj_tensors_unknown_modality_raises() -> None:
    """_new_proj_tensors must raise ValueError for unrecognised modality names."""
    from mousedroid.world_model.checkpoint_migration import _new_proj_tensors

    cfg = _small_cfg(
        ultrasonic_dim=0,
        ultrasonic_proj_dim=0,
        lidar_dim=6,
        lidar_proj_dim=8,
    )
    with pytest.raises(ValueError, match="Unknown modality"):
        _new_proj_tensors(cfg, "depth_camera")
