"""build_rssm_vision_finetune migrates a vision-OFF checkpoint to a vision-ON RSSM."""

from __future__ import annotations

from pathlib import Path

import torch

from mousedroid.config.schema import RoverConfig, RoverSimConfig, Settings
from mousedroid.factory import build_rssm_trainable, build_rssm_vision_finetune


def _cfg() -> Settings:
    return Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")),
    )


def test_migrates_and_enables_vision(tmp_path: Path) -> None:
    cfg = _cfg()
    pretrained = build_rssm_trainable(cfg)  # vision OFF (lidar on)
    assert pretrained.encoder.vision_enabled is False
    ckpt = tmp_path / "pre.pt"
    torch.save(pretrained.state_dict(), ckpt)

    fine = build_rssm_vision_finetune(cfg, ckpt)
    assert fine.encoder.vision_enabled is True
    assert fine.cfg.vision_dim == cfg.camera.feature_dim
    assert hasattr(fine.encoder, "vision_proj")


def test_no_rover_yields_lidar_off_vision_on(tmp_path: Path) -> None:
    """build_rssm_vision_finetune works with rover=None (lidar stays default-off)."""
    cfg = Settings(mock_hardware=True)  # rover is None
    pretrained = build_rssm_trainable(cfg)
    ckpt = tmp_path / "pre.pt"
    torch.save(pretrained.state_dict(), ckpt)
    fine = build_rssm_vision_finetune(cfg, ckpt)
    assert fine.encoder.vision_enabled is True
    assert fine.cfg.lidar_dim == cfg.model.lidar_dim  # unchanged when no rover


def test_dynamics_core_transferred_verbatim(tmp_path: Path) -> None:
    cfg = _cfg()
    pretrained = build_rssm_trainable(cfg)
    ckpt = tmp_path / "pre.pt"
    torch.save(pretrained.state_dict(), ckpt)
    fine = build_rssm_vision_finetune(cfg, ckpt)
    # gru / posterior / prior are shape-invariant -> copied verbatim from the ckpt.
    assert torch.allclose(fine.gru.weight_ih.cpu(), pretrained.gru.weight_ih.cpu())
    assert torch.allclose(fine.posterior.weight.cpu(), pretrained.posterior.weight.cpu())
    assert torch.allclose(fine.prior.weight.cpu(), pretrained.prior.weight.cpu())


def test_vision_off_to_vision_off_migration_is_identity(tmp_path: Path) -> None:
    """Loading a vision-OFF checkpoint into the SAME config adds no vision modality."""
    from mousedroid.world_model.checkpoint_migration import load_rssm_with_migration

    cfg = _cfg()
    pretrained = build_rssm_trainable(cfg)  # vision off
    ckpt = tmp_path / "pre.pt"
    torch.save(pretrained.state_dict(), ckpt)
    reloaded = load_rssm_with_migration(ckpt, pretrained.cfg, torch.device("cpu"))
    assert reloaded.encoder.vision_enabled is False  # no vision spuriously added
    assert torch.allclose(reloaded.gru.weight_ih.cpu(), pretrained.gru.weight_ih.cpu())
