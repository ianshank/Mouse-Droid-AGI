"""Unit tests for DualStreamRSSM — dual-stream hybrid CfC/GRU RSSM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

pytest.importorskip("ncps")

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
from mousedroid.world_model.protocol import SafetyTraceProtocol, WorldModelProtocol

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    cfc_dim: int = 16,
    hidden_dim: int = 32,
    latent_dim: int = 8,
    action_dim: int = 2,
) -> ModelConfig:
    return ModelConfig(
        vision_dim=16,
        ultrasonic_dim=1,
        motor_state_dim=4,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
        obs_dim=16,
        vision_proj_dim=8,
        ultrasonic_proj_dim=4,
        motor_proj_dim=4,
        cfc_hidden_dim=cfc_dim,
        cfc_backbone_units=32,
        cfc_backbone_layers=1,
    )


@dataclass
class MockObservation:
    timestamp: float = 0.0
    vision_features: np.ndarray = None  # type: ignore[assignment]
    distance_m: float = 2.0
    motor_state: np.ndarray = None  # type: ignore[assignment]
    audio_chunk: np.ndarray = None  # type: ignore[assignment]
    valid_mask: np.ndarray = None  # type: ignore[assignment]
    n_modalities: int = 4
    lidar_features: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.vision_features is None:
            self.vision_features = np.zeros(16, dtype=np.float32)
        if self.motor_state is None:
            self.motor_state = np.zeros(4, dtype=np.float32)
        if self.audio_chunk is None:
            self.audio_chunk = np.zeros(0, dtype=np.float32)
        if self.valid_mask is None:
            self.valid_mask = np.ones(4, dtype=np.float32)


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestDualStreamRSSMConstructor:
    """Test DualStreamRSSM construction."""

    def test_constructor_with_valid_config(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        assert model is not None

    def test_is_nn_module(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        assert isinstance(model, torch.nn.Module)

    def test_conforms_to_world_model_protocol(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        assert isinstance(model, WorldModelProtocol)

    def test_conforms_to_safety_trace_protocol(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        assert isinstance(model, SafetyTraceProtocol)

    def test_has_expected_submodules(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        assert hasattr(model, "encoder")
        assert hasattr(model, "gru")
        assert hasattr(model, "cfc")
        assert hasattr(model, "fusion")
        assert hasattr(model, "posterior")
        assert hasattr(model, "prior")
        assert hasattr(model, "reward_head")
        assert hasattr(model, "observation_decoder")

    def test_various_cfc_dimensions(self) -> None:
        for cfc_dim in [8, 16, 32, 64]:
            cfg = _make_cfg(cfc_dim=cfc_dim)
            model = DualStreamRSSM(cfg)
            assert model.cfc.hidden_size == cfc_dim


# ---------------------------------------------------------------------------
# observe_step tests
# ---------------------------------------------------------------------------


class TestDualStreamRSSMObserveStep:
    """Test observe_step — real observation processing."""

    def test_observe_step_shapes(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        obs = MockObservation()
        h = torch.zeros(1, combined_dim)
        z = torch.zeros(1, cfg.latent_dim)
        action = torch.zeros(1, cfg.action_dim)

        new_h, new_z, obs_embed, surprise = model.observe_step(obs, action, h, z)
        assert new_h.shape == (1, combined_dim)
        assert new_z.shape == (1, cfg.latent_dim)
        assert obs_embed.shape == (1, cfg.obs_dim)
        assert isinstance(surprise, float)

    def test_observe_step_finite_outputs(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        obs = MockObservation(
            vision_features=np.random.randn(cfg.vision_dim).astype(np.float32),
            distance_m=1.5,
            motor_state=np.array([0.1, 0.0, 0.2, 11.5], dtype=np.float32),
        )
        h = torch.randn(1, combined_dim) * 0.1
        z = torch.randn(1, cfg.latent_dim) * 0.1
        action = torch.randn(1, cfg.action_dim)

        new_h, new_z, obs_embed, surprise = model.observe_step(obs, action, h, z)
        assert torch.isfinite(new_h).all()
        assert torch.isfinite(new_z).all()
        assert torch.isfinite(obs_embed).all()
        assert np.isfinite(surprise)

    def test_surprise_is_finite(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        obs = MockObservation()
        h = torch.zeros(1, combined_dim)
        z = torch.zeros(1, cfg.latent_dim)
        action = torch.zeros(1, cfg.action_dim)

        _, _, _, surprise = model.observe_step(obs, action, h, z)
        assert np.isfinite(surprise)

    def test_observe_step_with_audio(self) -> None:
        cfg = _make_cfg()
        cfg_with_audio = ModelConfig(**{**cfg.model_dump(), "audio_dim": 64, "audio_proj_dim": 8})
        model = DualStreamRSSM(cfg_with_audio)
        combined_dim = cfg_with_audio.hidden_dim + cfg_with_audio.cfc_hidden_dim

        obs = MockObservation(
            vision_features=np.zeros(cfg_with_audio.vision_dim, dtype=np.float32),
            audio_chunk=np.random.randn(64).astype(np.float32),
        )
        h = torch.zeros(1, combined_dim)
        z = torch.zeros(1, cfg_with_audio.latent_dim)
        action = torch.zeros(1, cfg_with_audio.action_dim)

        new_h, new_z, _, surprise = model.observe_step(obs, action, h, z)
        assert torch.isfinite(new_h).all()
        assert torch.isfinite(new_z).all()
        assert np.isfinite(surprise)


# ---------------------------------------------------------------------------
# imagine_step tests
# ---------------------------------------------------------------------------


class TestDualStreamRSSMImagineStep:
    """Test imagine_step — latent imagination."""

    def test_imagine_step_shapes(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        h = torch.zeros(1, combined_dim)
        z = torch.zeros(1, cfg.latent_dim)
        action = torch.zeros(1, cfg.action_dim)

        new_h, new_z, reward = model.imagine_step(action, h, z)
        assert new_h.shape == (1, combined_dim)
        assert new_z.shape == (1, cfg.latent_dim)
        assert reward.shape == (1, 1)

    def test_imagine_step_no_grad(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        h = torch.zeros(1, combined_dim)
        z = torch.zeros(1, cfg.latent_dim)
        action = torch.zeros(1, cfg.action_dim)

        new_h, new_z, reward = model.imagine_step(action, h, z)
        assert not new_h.requires_grad
        assert not new_z.requires_grad
        assert not reward.requires_grad

    def test_imagine_step_1d_action(self) -> None:
        """imagine_step handles 1D action input via unsqueeze."""
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        h = torch.zeros(1, combined_dim)
        z = torch.zeros(1, cfg.latent_dim)
        action = torch.zeros(cfg.action_dim)  # 1D, no batch dim

        new_h, _new_z, _reward = model.imagine_step(action, h, z)
        assert new_h.shape == (1, combined_dim)

    def test_imagine_step_finite_output(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        h = torch.randn(1, combined_dim) * 0.1
        z = torch.randn(1, cfg.latent_dim) * 0.1
        action = torch.randn(1, cfg.action_dim)

        new_h, new_z, reward = model.imagine_step(action, h, z)
        assert torch.isfinite(new_h).all()
        assert torch.isfinite(new_z).all()
        assert torch.isfinite(reward).all()


# ---------------------------------------------------------------------------
# decode tests
# ---------------------------------------------------------------------------


class TestDualStreamRSSMDecode:
    """Test observation decoder."""

    def test_decode_shape(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        h = torch.randn(1, combined_dim)
        z = torch.randn(1, cfg.latent_dim)
        recon = model.decode(h, z)
        assert recon.shape == (1, cfg.obs_dim)

    def test_decode_batch(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        h = torch.randn(4, combined_dim)
        z = torch.randn(4, cfg.latent_dim)
        recon = model.decode(h, z)
        assert recon.shape == (4, cfg.obs_dim)

    def test_decode_gradient_flows(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        h = torch.randn(1, combined_dim, requires_grad=True)
        z = torch.randn(1, cfg.latent_dim, requires_grad=True)
        recon = model.decode(h, z)
        loss = recon.sum()
        loss.backward()
        assert h.grad is not None
        assert z.grad is not None


# ---------------------------------------------------------------------------
# Safety trace tests
# ---------------------------------------------------------------------------


class TestDualStreamRSSMSafetyTrace:
    """Test safety trace extraction."""

    def test_safety_trace_shape(self) -> None:
        cfg = _make_cfg(cfc_dim=32)
        model = DualStreamRSSM(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        h = torch.randn(1, combined_dim)
        trace = model.get_safety_trace(h)
        assert trace.shape == (1, 32)

    def test_safety_trace_extracts_cfc_portion(self) -> None:
        cfg = _make_cfg(cfc_dim=8, hidden_dim=4)
        model = DualStreamRSSM(cfg)

        # Construct combined state with known values
        h_gru = torch.ones(1, 4)
        h_cfc = torch.full((1, 8), 2.0)
        h_combined = torch.cat([h_gru, h_cfc], dim=-1)

        trace = model.get_safety_trace(h_combined)
        assert torch.allclose(trace, h_cfc)


# ---------------------------------------------------------------------------
# Parameter iterator tests
# ---------------------------------------------------------------------------


class TestDualStreamRSSMParameters:
    """Test parameter group separation for dual optimizer."""

    def test_gru_parameters_non_empty(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        gru_params = list(model.gru_parameters())
        assert len(gru_params) > 0

    def test_cfc_parameters_non_empty(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        cfc_params = list(model.cfc_parameters())
        assert len(cfc_params) > 0

    def test_cfc_parameters_are_cfc_only(self) -> None:
        """CfC parameter iterator should only yield CfC module params."""
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        cfc_param_set = {id(p) for p in model.cfc_parameters()}
        cfc_module_param_set = {id(p) for p in model.cfc.parameters()}
        assert cfc_param_set == cfc_module_param_set

    def test_gru_parameters_include_shared_heads(self) -> None:
        """GRU parameter iterator includes encoder + heads."""
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        gru_param_ids = {id(p) for p in model.gru_parameters()}
        # Should include GRU cell params
        for p in model.gru.parameters():
            assert id(p) in gru_param_ids
        # Should include encoder params
        for p in model.encoder.parameters():
            assert id(p) in gru_param_ids
        # Should include posterior params
        for p in model.posterior.parameters():
            assert id(p) in gru_param_ids

    def test_no_overlap_between_gru_and_cfc_params(self) -> None:
        """GRU and CfC parameter sets should not overlap."""
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        gru_ids = {id(p) for p in model.gru_parameters()}
        cfc_ids = {id(p) for p in model.cfc_parameters()}
        assert gru_ids.isdisjoint(cfc_ids)


# ---------------------------------------------------------------------------
# KL divergence tests
# ---------------------------------------------------------------------------


class TestKLDivergence:
    """Test KL divergence computation."""

    def test_kl_identical_distributions(self) -> None:
        mean = torch.zeros(2, 4)
        logvar = torch.zeros(2, 4)
        kl = DualStreamRSSM._kl_divergence(mean, logvar, mean, logvar)
        assert kl.item() == 0.0

    def test_kl_non_negative(self) -> None:
        post_mean = torch.randn(4, 8)
        post_logvar = torch.randn(4, 8)
        prior_mean = torch.randn(4, 8)
        prior_logvar = torch.randn(4, 8)
        kl = DualStreamRSSM._kl_divergence(post_mean, post_logvar, prior_mean, prior_logvar)
        assert kl.item() >= -1e-5  # Allow small numerical error


# ---------------------------------------------------------------------------
# Backward pass / gradient tests
# ---------------------------------------------------------------------------


class TestDualStreamRSSMGradients:
    """Test gradient flow through the full model."""

    def test_backward_pass_finite_gradients(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        h = torch.zeros(1, combined_dim)
        z = torch.randn(1, cfg.latent_dim)
        action = torch.randn(1, cfg.action_dim)

        obs = MockObservation()
        new_h, new_z, obs_embed, _ = model.observe_step(obs, action, h, z)

        # Compute a loss and backward
        recon = model.decode(new_h, new_z)
        loss = (recon - obs_embed).pow(2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"


# ---------------------------------------------------------------------------
# Stability tests
# ---------------------------------------------------------------------------


class TestDualStreamRSSMStability:
    """Test numerical stability over long rollouts."""

    def test_long_imagine_rollout_stable(self) -> None:
        """Hidden state remains finite over 100 sequential imagine steps."""
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        model.eval()
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        h = torch.zeros(1, combined_dim)
        z = torch.zeros(1, cfg.latent_dim)

        for _ in range(100):
            action = torch.randn(1, cfg.action_dim)
            h, z, _ = model.imagine_step(action, h, z)

        assert torch.isfinite(h).all(), "Hidden state diverged after 100 steps"
        assert torch.isfinite(z).all(), "Latent state diverged after 100 steps"

    def test_sequential_observe_steps_stable(self) -> None:
        """Multiple observe steps don't produce NaN/Inf."""
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        h = torch.zeros(1, combined_dim)
        z = torch.zeros(1, cfg.latent_dim)
        action = torch.zeros(1, cfg.action_dim)

        for _ in range(10):
            obs = MockObservation(
                vision_features=np.random.randn(cfg.vision_dim).astype(np.float32),
            )
            h, z, _, surprise = model.observe_step(obs, action, h, z)
            assert torch.isfinite(h).all()
            assert torch.isfinite(z).all()
            assert np.isfinite(surprise)
