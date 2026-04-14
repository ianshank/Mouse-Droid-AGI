"""Regression tests for dual-stream RSSM backwards compatibility.

Verifies that:
- Existing ``Settings`` and ``ModelConfig`` defaults still load without changes.
- ``cfc_hidden_dim=0`` continues to build a classic RSSM (not DualStreamRSSM).
- ``cfc_hidden_dim>0`` now builds a DualStreamRSSM as expected.
- ``DualStreamTrainingConfig`` defaults exist and are sensible.
- Classic RSSM observe_step/imagine_step still pass unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from mousedroid.config.schema import DualStreamTrainingConfig, ModelConfig, Settings
from mousedroid.factory import build_world_model
from mousedroid.world_model.protocol import WorldModelProtocol
from mousedroid.world_model.rssm import RSSM

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _settings_with_cfc_dim(cfc_hidden_dim: int) -> Settings:
    """Build a Settings instance with the given CfC hidden dimension.

    Args:
        cfc_hidden_dim: Value to assign to ``model.cfc_hidden_dim``.
            Use 0 to disable the CfC stream (classic RSSM).

    Returns:
        Settings instance with mock_hardware=True and the specified CfC dim.
    """
    return Settings(
        mock_hardware=True,
        model=ModelConfig(
            vision_dim=16,
            ultrasonic_dim=1,
            motor_state_dim=4,
            hidden_dim=32,
            latent_dim=8,
            action_dim=2,
            obs_dim=16,
            vision_proj_dim=8,
            ultrasonic_proj_dim=4,
            motor_proj_dim=4,
            cfc_hidden_dim=cfc_hidden_dim,
            cfc_backbone_units=32,
            cfc_backbone_layers=1,
        ),
    )


@dataclass
class _MockObservation:
    """Minimal mock observation for regression test forward passes."""

    vision_features: np.ndarray
    distance_m: float = 2.0
    motor_state: np.ndarray = None  # type: ignore[assignment]
    audio_chunk: np.ndarray = None  # type: ignore[assignment]
    valid_mask: np.ndarray = None  # type: ignore[assignment]
    lidar_features: np.ndarray = None  # type: ignore[assignment]
    n_modalities: int = 4
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        """Fill default arrays for any optional modalities left as None."""
        if self.motor_state is None:
            self.motor_state = np.zeros(4, dtype=np.float32)
        if self.audio_chunk is None:
            self.audio_chunk = np.zeros(0, dtype=np.float32)
        if self.valid_mask is None:
            self.valid_mask = np.ones(4, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. test_classic_rssm_when_cfc_disabled
# ---------------------------------------------------------------------------


class TestClassicRssmWhenCfcDisabled:
    """build_world_model returns RSSM (not DualStreamRSSM) when cfc_hidden_dim=0."""

    def test_classic_rssm_when_cfc_disabled(self) -> None:
        """Settings with cfc_hidden_dim=0 must produce a classic RSSM instance.

        This guarantees that existing deployments that omit cfc_hidden_dim from
        their YAML configs continue to receive the original RSSM world model.
        Changing this behaviour would be a silent breaking change.
        """
        cfg = _settings_with_cfc_dim(0)
        world_model = build_world_model(cfg)

        assert isinstance(world_model, RSSM), (
            f"Expected RSSM but got {type(world_model).__name__} when cfc_hidden_dim=0"
        )

    def test_classic_rssm_not_dual_stream_type(self) -> None:
        """A classic RSSM must not be an instance of DualStreamRSSM.

        Running isinstance checks against both types guards against a future
        refactoring that accidentally makes RSSM a subclass of DualStreamRSSM.
        """
        pytest.importorskip("ncps", reason="ncps required to import DualStreamRSSM")
        from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

        cfg = _settings_with_cfc_dim(0)
        world_model = build_world_model(cfg)

        assert not isinstance(world_model, DualStreamRSSM), (
            "build_world_model with cfc_hidden_dim=0 must not produce DualStreamRSSM"
        )

    def test_classic_rssm_conforms_to_protocol(self) -> None:
        """The returned RSSM must still satisfy WorldModelProtocol.

        Protocol conformance must not be broken by adding the dual-stream path
        in build_world_model.
        """
        cfg = _settings_with_cfc_dim(0)
        world_model = build_world_model(cfg)

        assert isinstance(world_model, WorldModelProtocol), (
            "Classic RSSM must conform to WorldModelProtocol"
        )


# ---------------------------------------------------------------------------
# 2. test_dual_stream_when_cfc_enabled
# ---------------------------------------------------------------------------


class TestDualStreamWhenCfcEnabled:
    """build_world_model returns DualStreamRSSM when cfc_hidden_dim > 0."""

    def test_dual_stream_when_cfc_enabled(self) -> None:
        """Settings with cfc_hidden_dim=64 must produce a DualStreamRSSM.

        This confirms the factory routing introduced by the dual-stream feature
        is active and produces the expected model type for new configurations.
        """
        pytest.importorskip("ncps", reason="ncps required for DualStreamRSSM")
        from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

        cfg = _settings_with_cfc_dim(64)
        world_model = build_world_model(cfg)

        assert isinstance(world_model, DualStreamRSSM), (
            f"Expected DualStreamRSSM but got {type(world_model).__name__} when cfc_hidden_dim=64"
        )

    @pytest.mark.parametrize("cfc_dim", [8, 16, 32, 64, 128])
    def test_dual_stream_for_various_cfc_dims(self, cfc_dim: int) -> None:
        """Any positive cfc_hidden_dim triggers DualStreamRSSM construction.

        Tests a range of CfC dimensions to confirm the factory condition is
        ``> 0`` (not a specific magic value).

        Args:
            cfc_dim: CfC hidden dimension to test.
        """
        pytest.importorskip("ncps", reason="ncps required for DualStreamRSSM")
        from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

        cfg = _settings_with_cfc_dim(cfc_dim)
        world_model = build_world_model(cfg)

        assert isinstance(world_model, DualStreamRSSM), (
            f"Expected DualStreamRSSM for cfc_dim={cfc_dim}, got {type(world_model).__name__}"
        )

    def test_dual_stream_conforms_to_protocol(self) -> None:
        """DualStreamRSSM must conform to WorldModelProtocol.

        Protocol conformance is required for the dual-stream model to be a
        drop-in replacement for RSSM in the orchestrator.
        """
        pytest.importorskip("ncps", reason="ncps required for DualStreamRSSM")

        cfg = _settings_with_cfc_dim(64)
        world_model = build_world_model(cfg)

        assert isinstance(world_model, WorldModelProtocol), (
            "DualStreamRSSM must conform to WorldModelProtocol"
        )


# ---------------------------------------------------------------------------
# 3. test_config_defaults_backward_compatible
# ---------------------------------------------------------------------------


class TestConfigDefaultsBackwardCompatible:
    """Default Settings() must load without errors even without CfC fields in YAML."""

    def test_config_defaults_backward_compatible(self) -> None:
        """Default Settings(mock_hardware=True) succeeds without any YAML overrides.

        Any code that previously constructed ``Settings(mock_hardware=True)``
        must continue to work after the dual-stream fields were added.
        """
        cfg = Settings(mock_hardware=True)
        assert cfg is not None

    def test_cfc_hidden_dim_defaults_to_zero(self) -> None:
        """Default ModelConfig has cfc_hidden_dim=0 (CfC disabled).

        The zero default ensures that existing configs that do not mention
        cfc_hidden_dim continue to use the classic GRU-only RSSM.
        """
        cfg = Settings(mock_hardware=True)
        assert cfg.model.cfc_hidden_dim == 0, (
            f"Default cfc_hidden_dim should be 0, got {cfg.model.cfc_hidden_dim}"
        )

    def test_dual_stream_training_config_present_in_settings(self) -> None:
        """Settings must expose a dual_stream_training attribute after the feature branch.

        This attribute must exist on the default Settings so that training code
        can unconditionally access it without version-checking.
        """
        cfg = Settings(mock_hardware=True)
        assert hasattr(cfg, "dual_stream_training"), (
            "Settings must have a dual_stream_training attribute"
        )
        assert isinstance(cfg.dual_stream_training, DualStreamTrainingConfig)

    def test_minimal_yaml_style_settings_still_works(self) -> None:
        """A minimal dict equivalent to a legacy YAML file must produce a valid Settings.

        This mimics the yaml.safe_load path used by the training CLI and
        ensures legacy configs without dual-stream fields are still parseable.
        """
        legacy_overrides = {
            "mock_hardware": True,
            "platform": "mouse_droid",
        }
        cfg = Settings(**legacy_overrides)
        assert cfg.mock_hardware is True
        assert cfg.model.cfc_hidden_dim == 0

    def test_new_cfc_model_fields_have_defaults(self) -> None:
        """All CfC-specific ModelConfig fields introduced by this branch have defaults.

        Without defaults, any existing YAML config that omits these fields
        would fail to load, breaking backwards compatibility.
        """
        model_cfg = ModelConfig(
            vision_dim=16,
            ultrasonic_dim=1,
            motor_state_dim=4,
            hidden_dim=32,
            latent_dim=8,
            action_dim=3,
            obs_dim=16,
            vision_proj_dim=8,
            ultrasonic_proj_dim=4,
            motor_proj_dim=4,
            # Intentionally omit all cfc_* fields to test defaults
        )
        assert model_cfg.cfc_hidden_dim == 0
        assert model_cfg.cfc_backbone_units > 0
        assert model_cfg.cfc_backbone_layers > 0
        assert isinstance(model_cfg.cfc_mode, str)
        assert 0.0 <= model_cfg.cfc_sparsity_level <= 1.0


# ---------------------------------------------------------------------------
# 4. test_dual_stream_training_config_defaults
# ---------------------------------------------------------------------------


class TestDualStreamTrainingConfigDefaults:
    """DualStreamTrainingConfig() must construct successfully with all defaults."""

    def test_dual_stream_training_config_defaults(self) -> None:
        """DualStreamTrainingConfig() initialises with all required defaults.

        Construction without any arguments must not raise, confirming that
        every field has a valid default value and can be used out of the box.
        """
        train_cfg = DualStreamTrainingConfig()
        assert train_cfg is not None

    def test_default_gru_lr(self) -> None:
        """Default gru_lr is 3e-4 as documented.

        This is the standard Adam learning rate for the primary GRU stream.
        """
        train_cfg = DualStreamTrainingConfig()
        assert train_cfg.gru_lr == pytest.approx(3e-4, rel=1e-6)

    def test_default_cfc_lr(self) -> None:
        """Default cfc_lr is 1e-4 as documented.

        The CfC stream uses a lower learning rate than the GRU to give the
        primary stream time to stabilise before the liquid network adapts.
        """
        train_cfg = DualStreamTrainingConfig()
        assert train_cfg.cfc_lr == pytest.approx(1e-4, rel=1e-6)

    def test_default_gru_grad_clip(self) -> None:
        """Default gru_grad_clip is 10.0."""
        train_cfg = DualStreamTrainingConfig()
        assert train_cfg.gru_grad_clip == pytest.approx(10.0, rel=1e-6)

    def test_default_cfc_grad_clip(self) -> None:
        """Default cfc_grad_clip is 1.0 (tighter than GRU clip)."""
        train_cfg = DualStreamTrainingConfig()
        assert train_cfg.cfc_grad_clip == pytest.approx(1.0, rel=1e-6)

    def test_default_cfc_loss_weight_initial(self) -> None:
        """Default cfc_loss_weight_initial is 0.1."""
        train_cfg = DualStreamTrainingConfig()
        assert train_cfg.cfc_loss_weight_initial == pytest.approx(0.1, rel=1e-6)

    def test_default_cfc_loss_weight_final(self) -> None:
        """Default cfc_loss_weight_final is 1.0."""
        train_cfg = DualStreamTrainingConfig()
        assert train_cfg.cfc_loss_weight_final == pytest.approx(1.0, rel=1e-6)

    def test_default_cfc_loss_warmup_steps(self) -> None:
        """Default cfc_loss_warmup_steps is 10000."""
        train_cfg = DualStreamTrainingConfig()
        assert train_cfg.cfc_loss_warmup_steps == 10000

    def test_default_fallback_check_interval(self) -> None:
        """Default fallback_check_interval is 1000."""
        train_cfg = DualStreamTrainingConfig()
        assert train_cfg.fallback_check_interval == 1000

    def test_default_fallback_degradation_threshold(self) -> None:
        """Default fallback_degradation_threshold is 0.05."""
        train_cfg = DualStreamTrainingConfig()
        assert train_cfg.fallback_degradation_threshold == pytest.approx(0.05, rel=1e-6)

    def test_initial_weight_less_than_final(self) -> None:
        """cfc_loss_weight_initial must be less than cfc_loss_weight_final.

        The warmup schedule ramps from initial to final; if initial >= final
        the schedule is trivially flat or inverted, which is almost certainly
        a misconfiguration.
        """
        train_cfg = DualStreamTrainingConfig()
        assert train_cfg.cfc_loss_weight_initial < train_cfg.cfc_loss_weight_final, (
            "cfc_loss_weight_initial must be < cfc_loss_weight_final for a valid warmup schedule"
        )


# ---------------------------------------------------------------------------
# 5. test_existing_rssm_tests_unaffected
# ---------------------------------------------------------------------------


class TestExistingRssmTestsUnaffected:
    """Classic RSSM observe_step and imagine_step still pass after dual-stream changes."""

    def _make_classic_cfg(self) -> ModelConfig:
        """Build a ModelConfig with cfc_hidden_dim=0 (pure GRU).

        Returns:
            ModelConfig for classic RSSM with no CfC.
        """
        return ModelConfig(
            vision_dim=16,
            ultrasonic_dim=1,
            motor_state_dim=4,
            hidden_dim=32,
            latent_dim=8,
            action_dim=3,
            obs_dim=16,
            vision_proj_dim=8,
            ultrasonic_proj_dim=4,
            motor_proj_dim=4,
            cfc_hidden_dim=0,
        )

    def test_existing_rssm_observe_step_shape(self) -> None:
        """Classic RSSM observe_step still returns correctly shaped tensors.

        This test mirrors the existing observe_step shape contract to catch any
        regression introduced by adding dual-stream routing in factory.py.
        """
        cfg = self._make_classic_cfg()
        rssm = RSSM(cfg)

        obs = _MockObservation(vision_features=np.zeros(cfg.vision_dim, dtype=np.float32))
        h = torch.zeros(1, cfg.hidden_dim)
        z = torch.zeros(1, cfg.latent_dim)
        action = torch.zeros(1, cfg.action_dim)

        new_h, new_z, obs_embed, surprise = rssm.observe_step(obs, action, h, z)

        assert new_h.shape == (
            1,
            cfg.hidden_dim,
        ), f"observe_step new_h shape: expected (1, {cfg.hidden_dim}), got {new_h.shape}"
        assert new_z.shape == (
            1,
            cfg.latent_dim,
        ), f"observe_step new_z shape: expected (1, {cfg.latent_dim}), got {new_z.shape}"
        assert obs_embed.shape == (
            1,
            cfg.obs_dim,
        ), f"observe_step obs_embed shape: expected (1, {cfg.obs_dim}), got {obs_embed.shape}"
        assert isinstance(surprise, float), (
            f"observe_step surprise must be float, got {type(surprise)}"
        )

    def test_existing_rssm_observe_step_finite(self) -> None:
        """Classic RSSM observe_step outputs must be finite after dual-stream changes.

        Ensures that structural changes in the encoder or factory path did not
        introduce a NaN/Inf regression in the classic code path.
        """
        cfg = self._make_classic_cfg()
        rssm = RSSM(cfg)

        obs = _MockObservation(
            vision_features=np.random.randn(cfg.vision_dim).astype(np.float32),
            distance_m=1.5,
            motor_state=np.array([0.1, 0.0, 0.2, 11.5], dtype=np.float32),
        )
        h = torch.randn(1, cfg.hidden_dim) * 0.1
        z = torch.randn(1, cfg.latent_dim) * 0.1
        action = torch.randn(1, cfg.action_dim)

        new_h, new_z, obs_embed, surprise = rssm.observe_step(obs, action, h, z)

        assert torch.isfinite(new_h).all(), "Classic RSSM observe_step new_h is not finite"
        assert torch.isfinite(new_z).all(), "Classic RSSM observe_step new_z is not finite"
        assert torch.isfinite(obs_embed).all(), "Classic RSSM observe_step obs_embed is not finite"
        assert np.isfinite(surprise), (
            f"Classic RSSM observe_step surprise is not finite: {surprise}"
        )

    def test_existing_rssm_imagine_step_shape(self) -> None:
        """Classic RSSM imagine_step still returns correctly shaped tensors.

        The imagine_step contract (new_h, new_z, reward) must not have changed.
        """
        cfg = self._make_classic_cfg()
        rssm = RSSM(cfg)
        rssm.eval()

        h = torch.zeros(1, cfg.hidden_dim)
        z = torch.zeros(1, cfg.latent_dim)
        action = torch.zeros(1, cfg.action_dim)

        new_h, new_z, reward = rssm.imagine_step(action, h, z)

        assert new_h.shape == (
            1,
            cfg.hidden_dim,
        ), f"imagine_step new_h shape: expected (1, {cfg.hidden_dim}), got {new_h.shape}"
        assert new_z.shape == (
            1,
            cfg.latent_dim,
        ), f"imagine_step new_z shape: expected (1, {cfg.latent_dim}), got {new_z.shape}"
        assert reward.shape == (
            1,
            1,
        ), f"imagine_step reward shape: expected (1, 1), got {reward.shape}"

    def test_existing_rssm_imagine_step_no_grad(self) -> None:
        """Classic RSSM imagine_step outputs must not require gradients.

        imagine_step is decorated with @torch.no_grad(); that invariant must
        not have been removed by any dual-stream refactoring.
        """
        cfg = self._make_classic_cfg()
        rssm = RSSM(cfg)
        rssm.eval()

        h = torch.zeros(1, cfg.hidden_dim)
        z = torch.zeros(1, cfg.latent_dim)
        action = torch.zeros(1, cfg.action_dim)

        new_h, new_z, reward = rssm.imagine_step(action, h, z)

        assert not new_h.requires_grad, "imagine_step new_h must not require grad"
        assert not new_z.requires_grad, "imagine_step new_z must not require grad"
        assert not reward.requires_grad, "imagine_step reward must not require grad"

    def test_factory_produces_rssm_matching_settings_model(self) -> None:
        """build_world_model with cfc_hidden_dim=0 returns RSSM using cfg.model dims.

        The factory must thread the model config through correctly so that the
        built RSSM's hidden_dim matches the Settings value.
        """
        cfg = _settings_with_cfc_dim(0)
        world_model = build_world_model(cfg)

        assert isinstance(world_model, RSSM)
        # The GRU hidden dim must match the model config
        assert world_model.gru.hidden_size == cfg.model.hidden_dim, (
            f"RSSM hidden_dim mismatch: expected {cfg.model.hidden_dim}, "
            f"got {world_model.gru.hidden_size}"
        )
