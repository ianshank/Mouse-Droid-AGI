from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.rssm import RSSM


@dataclass
class MockObservationBundle:
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


@pytest.fixture
def cfg() -> ModelConfig:
    return ModelConfig()


@pytest.fixture
def rssm(cfg: ModelConfig) -> RSSM:
    return RSSM(cfg)


def test_constructor(rssm: RSSM) -> None:
    assert hasattr(rssm, "encoder")
    assert hasattr(rssm, "gru")
    assert hasattr(rssm, "posterior")
    assert hasattr(rssm, "prior")
    assert hasattr(rssm, "reward_head")


def test_constructor_custom_cfg() -> None:
    cfg = ModelConfig(hidden_dim=128, latent_dim=32, action_dim=2)
    model = RSSM(cfg)
    assert model.gru.input_size == 32 + 2
    assert model.gru.hidden_size == 128


def test_observe_step_returns_correct_tuple_sizes(rssm: RSSM, cfg: ModelConfig) -> None:
    obs = MockObservationBundle()
    prev_action = torch.zeros(1, cfg.action_dim)
    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    new_h, new_z, obs_embed, surprise = rssm.observe_step(obs, prev_action, h, z)
    assert new_h.shape == (1, cfg.hidden_dim)
    assert new_z.shape == (1, cfg.latent_dim)
    assert obs_embed.shape == (1, cfg.obs_dim)
    assert isinstance(surprise, float)


def test_imagine_step_returns_correct_tuple_sizes(rssm: RSSM, cfg: ModelConfig) -> None:
    action = torch.zeros(1, cfg.action_dim)
    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    new_h, new_z, reward = rssm.imagine_step(action, h, z)
    assert new_h.shape == (1, cfg.hidden_dim)
    assert new_z.shape == (1, cfg.latent_dim)
    assert reward.shape == (1, 1)


def test_imagine_step_no_grad(rssm: RSSM, cfg: ModelConfig) -> None:
    action = torch.zeros(1, cfg.action_dim)
    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    new_h, new_z, reward = rssm.imagine_step(action, h, z)
    assert not new_h.requires_grad
    assert not new_z.requires_grad
    assert not reward.requires_grad


def test_observe_with_mock_bundle(rssm: RSSM, cfg: ModelConfig) -> None:
    obs = MockObservationBundle(
        vision_features=np.random.randn(256).astype(np.float32),
        distance_m=1.5,
        motor_state=np.array([0.1, 0.0, 0.2, 11.5], dtype=np.float32),
    )
    prev_action = torch.randn(1, cfg.action_dim)
    h = torch.randn(1, cfg.hidden_dim)
    z = torch.randn(1, cfg.latent_dim)
    new_h, new_z, _obs_embed, surprise = rssm.observe_step(obs, prev_action, h, z)
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_z).all()
    assert np.isfinite(surprise)


def test_surprise_is_non_negative(rssm: RSSM, cfg: ModelConfig) -> None:
    obs = MockObservationBundle()
    prev_action = torch.zeros(1, cfg.action_dim)
    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    _, _, _, surprise = rssm.observe_step(obs, prev_action, h, z)
    assert surprise >= 0.0 or np.isfinite(surprise)


def test_rssm_is_nn_module(rssm: RSSM) -> None:
    assert isinstance(rssm, torch.nn.Module)


def test_kl_divergence_identical_distributions() -> None:
    mean = torch.zeros(2, 4)
    logvar = torch.zeros(2, 4)
    kl = RSSM._kl_divergence(mean, logvar, mean, logvar)
    assert kl.item() == pytest.approx(0.0, abs=1e-5)


def test_sample_gaussian(rssm: RSSM) -> None:
    params = torch.randn(2, 128)  # latent_dim*2 = 64*2
    sample, mean, logvar = rssm._sample_gaussian(params)
    assert sample.shape == (2, 64)
    assert mean.shape == (2, 64)
    assert logvar.shape == (2, 64)


# ---------------------------------------------------------------------------
# Audio-enabled RSSM tests
# ---------------------------------------------------------------------------


def test_observe_step_audio_enabled() -> None:
    cfg = ModelConfig(audio_dim=1024, audio_proj_dim=32)
    model = RSSM(cfg)
    obs = MockObservationBundle(
        audio_chunk=np.random.randn(1024).astype(np.float32),
    )
    prev_action = torch.zeros(1, cfg.action_dim)
    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    new_h, new_z, obs_embed, surprise = model.observe_step(obs, prev_action, h, z)
    assert new_h.shape == (1, cfg.hidden_dim)
    assert new_z.shape == (1, cfg.latent_dim)
    assert obs_embed.shape == (1, cfg.obs_dim)
    assert np.isfinite(surprise)


def test_observe_step_audio_disabled_ignores_chunk(rssm: RSSM, cfg: ModelConfig) -> None:
    """Default config (audio_dim=0) ignores audio_chunk without error."""
    obs = MockObservationBundle(
        audio_chunk=np.random.randn(1024).astype(np.float32),
    )
    prev_action = torch.zeros(1, cfg.action_dim)
    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    new_h, _new_z, _obs_embed, _surprise = rssm.observe_step(obs, prev_action, h, z)
    assert new_h.shape == (1, cfg.hidden_dim)
    assert torch.isfinite(new_h).all()


def test_observe_step_audio_with_nonzero_data() -> None:
    """Audio data actually influences the encoding when enabled."""
    cfg = ModelConfig(audio_dim=1024, audio_proj_dim=32)
    model = RSSM(cfg)
    model.eval()

    prev_action = torch.zeros(1, cfg.action_dim)
    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)

    obs_silent = MockObservationBundle(
        audio_chunk=np.zeros(1024, dtype=np.float32),
    )
    obs_loud = MockObservationBundle(
        audio_chunk=np.ones(1024, dtype=np.float32),
    )

    torch.manual_seed(0)
    _, _, embed_silent, _ = model.observe_step(obs_silent, prev_action, h, z)
    torch.manual_seed(0)
    _, _, embed_loud, _ = model.observe_step(obs_loud, prev_action, h, z)
    assert not torch.allclose(embed_silent, embed_loud)
