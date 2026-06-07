"""RSSM.train_sequence: grad-enabled raw-modality reconstruction + KL."""

from __future__ import annotations

import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.rssm import RSSM


def _model() -> RSSM:
    # vision off (pretraining variant); lidar on so we exercise the lidar head.
    cfg = ModelConfig(vision_dim=0, vision_proj_dim=0, lidar_dim=16, lidar_proj_dim=32)
    torch.manual_seed(0)
    return RSSM(cfg)


def _batch(model: RSSM, b: int = 4, t: int = 6) -> dict[str, torch.Tensor]:
    cfg = model.cfg
    return {
        "motor": torch.randn(b, t, cfg.motor_state_dim),
        "ultrasonic": torch.rand(b, t, 1),
        "lidar": torch.rand(b, t, cfg.lidar_dim),
        "valid_mask": torch.ones(b, t, 5),
        "action": torch.randn(b, t, cfg.action_dim),
    }


def test_train_sequence_returns_finite_losses() -> None:
    model = _model()
    out = model.train_sequence(_batch(model))
    for key in ("loss", "recon", "kl"):
        assert torch.isfinite(out[key])
    assert out["loss"].requires_grad


def test_train_sequence_backward_populates_grads() -> None:
    model = _model()
    out = model.train_sequence(_batch(model))
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_overfits_single_batch_loss_decreases() -> None:
    model = _model()
    batch = _batch(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first: float | None = None
    out: dict[str, torch.Tensor] = {}
    for _ in range(40):
        opt.zero_grad()
        out = model.train_sequence(batch)
        out["loss"].backward()
        opt.step()
        if first is None:
            first = out["loss"].detach().item()
    assert first is not None
    assert out["loss"].detach().item() < first  # learns something


def test_no_posterior_collapse_probe() -> None:
    """posterior_std stays above a floor — guards the obs_embed-collapse failure."""
    model = _model()
    out = model.train_sequence(_batch(model))
    assert float(out["posterior_std"]) > 1e-3
