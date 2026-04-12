"""Integration test: audio flows end-to-end from observation bundle through RSSM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.rssm import RSSM


@dataclass
class _AudioObservation:
    """Minimal observation bundle for audio integration testing."""

    timestamp: float = 0.0
    vision_features: np.ndarray = None  # type: ignore[assignment]
    distance_m: float = 2.0
    motor_state: np.ndarray = None  # type: ignore[assignment]
    audio_chunk: np.ndarray = None  # type: ignore[assignment]
    valid_mask: np.ndarray = None  # type: ignore[assignment]
    n_modalities: int = 4

    def __post_init__(self) -> None:
        if self.vision_features is None:
            self.vision_features = np.zeros(256, dtype=np.float32)
        if self.motor_state is None:
            self.motor_state = np.zeros(4, dtype=np.float32)
        if self.audio_chunk is None:
            self.audio_chunk = np.zeros(1024, dtype=np.float32)
        if self.valid_mask is None:
            self.valid_mask = np.ones(4, dtype=np.float32)


def test_audio_end_to_end_observe_step() -> None:
    """Full pipeline: config -> RSSM -> encoder with audio -> valid output."""
    cfg = ModelConfig(audio_dim=1024, audio_proj_dim=32)
    rssm = RSSM(cfg)

    obs = _AudioObservation(
        vision_features=np.random.randn(256).astype(np.float32),
        distance_m=1.5,
        motor_state=np.array([0.1, 0.0, 0.2, 11.5], dtype=np.float32),
        audio_chunk=np.random.randn(1024).astype(np.float32),
        valid_mask=np.ones(4, dtype=np.float32),
    )

    prev_action = torch.zeros(1, cfg.action_dim)
    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)

    new_h, new_z, obs_embed, surprise = rssm.observe_step(obs, prev_action, h, z)

    assert new_h.shape == (1, cfg.hidden_dim)
    assert new_z.shape == (1, cfg.latent_dim)
    assert obs_embed.shape == (1, cfg.obs_dim)
    assert np.isfinite(surprise)
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_z).all()
    assert torch.isfinite(obs_embed).all()


def test_audio_disabled_pipeline_unchanged() -> None:
    """Default config (audio_dim=0) still works with 4-element valid_mask."""
    cfg = ModelConfig()  # audio_dim=0
    rssm = RSSM(cfg)

    obs = _AudioObservation(
        audio_chunk=np.random.randn(1024).astype(np.float32),
    )

    prev_action = torch.zeros(1, cfg.action_dim)
    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)

    new_h, new_z, _obs_embed, surprise = rssm.observe_step(obs, prev_action, h, z)

    assert new_h.shape == (1, cfg.hidden_dim)
    assert new_z.shape == (1, cfg.latent_dim)
    assert np.isfinite(surprise)


def test_multi_step_audio_sequence() -> None:
    """Multiple observe steps with varying audio produce finite state."""
    cfg = ModelConfig(audio_dim=1024, audio_proj_dim=32)
    rssm = RSSM(cfg)

    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    prev_action = torch.zeros(1, cfg.action_dim)

    rng = np.random.default_rng(42)
    for _ in range(5):
        obs = _AudioObservation(
            vision_features=rng.standard_normal(256).astype(np.float32),
            motor_state=rng.standard_normal(4).astype(np.float32),
            audio_chunk=rng.standard_normal(1024).astype(np.float32),
        )
        h, z, _, surprise = rssm.observe_step(obs, prev_action, h, z)
        assert torch.isfinite(h).all()
        assert torch.isfinite(z).all()
        assert np.isfinite(surprise)
        prev_action = torch.randn(1, cfg.action_dim)
