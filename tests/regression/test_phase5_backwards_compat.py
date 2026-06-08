"""Phase 5 Layer-1 additions are byte-identical by default."""

from __future__ import annotations

import torch

from mousedroid.config.schema import ModelConfig, Settings
from mousedroid.world_model.encoder import MultimodalEncoder


def test_default_model_has_vision_enabled() -> None:
    assert MultimodalEncoder(ModelConfig()).vision_enabled is True


def test_default_training_config_pretrain_disabled() -> None:
    cfg = Settings(mock_hardware=True)
    assert cfg.training.rssm_pretrain_enabled is False


def test_default_encoder_output_shape_unchanged() -> None:
    cfg = ModelConfig()
    enc = MultimodalEncoder(cfg)
    out = enc(
        torch.zeros(1, cfg.vision_dim),
        None,
        torch.zeros(1, cfg.motor_state_dim),
        torch.ones(1, 5),
    )
    assert out.shape == (1, cfg.obs_dim)


def test_default_model_kl_knobs_have_dreamer_defaults() -> None:
    cfg = ModelConfig()
    assert cfg.kl_beta == 1.0
    assert cfg.kl_balance_alpha == 0.8
    assert cfg.kl_free_nats == 1.0
