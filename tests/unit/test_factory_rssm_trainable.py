"""build_rssm_trainable returns a concrete trainable RSSM with vision off."""

from __future__ import annotations

from mousedroid.config.schema import Settings
from mousedroid.factory import build_rssm_trainable
from mousedroid.world_model.rssm import RSSM


def test_returns_trainable_rssm_vision_off() -> None:
    cfg = Settings(mock_hardware=True)
    model = build_rssm_trainable(cfg)
    assert isinstance(model, RSSM)
    assert model.encoder.vision_enabled is False  # pretraining drops vision
    assert any(p.requires_grad for p in model.parameters())


def test_overrides_pretrain_knobs_from_training_config() -> None:
    cfg = Settings(mock_hardware=True)
    model = build_rssm_trainable(cfg)
    assert model.cfg.kl_free_nats == cfg.training.rssm_free_nats
    assert model.cfg.kl_balance_alpha == cfg.training.rssm_kl_balance_alpha
