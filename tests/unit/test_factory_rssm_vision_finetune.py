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
