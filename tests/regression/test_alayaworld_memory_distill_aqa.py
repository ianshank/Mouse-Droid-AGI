"""AQA regression for F-023 (AlayaWorld adaptation) — schema hygiene + contracts.

Pins the requirement-level scenarios so a future refactor cannot silently drop
them: config-field hygiene (defaults OFF, every field described), protocol
conformance, the S1a constant-footprint scenario end-to-end through the
factory, and the disabled==baseline gate.
"""

from __future__ import annotations

import pytest
import torch

from mousedroid.common.torch_device import resolve_device
from mousedroid.config.schema import (
    DriftTrainingConfig,
    ModelConfig,
    Settings,
    WorldModelMemoryConfig,
)
from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.factory import build_latent_context
from mousedroid.training.drift_metrics import measure_drift
from mousedroid.world_model.bounded_context import BoundedContextMemory
from mousedroid.world_model.protocol import LatentContextProtocol
from mousedroid.world_model.rssm import RSSM, RawModalityDecoders


class TestConfigHygiene:
    def test_memory_block_defaults_off_with_descriptions(self) -> None:
        cfg = WorldModelMemoryConfig()
        assert cfg.enabled is False
        for name, field in WorldModelMemoryConfig.model_fields.items():
            assert field.description, f"world_model_memory.{name} missing description"

    def test_drift_block_defaults_off_with_descriptions(self) -> None:
        cfg = DriftTrainingConfig()
        assert cfg.enabled is False
        for name, field in DriftTrainingConfig.model_fields.items():
            assert field.description, f"training.drift.{name} missing description"

    def test_settings_fields_are_optional_default_none(self) -> None:
        assert Settings.model_fields["world_model_memory"].default is None
        # training.drift nests under TrainingConfig (training.replay precedent).
        from mousedroid.config.schema import TrainingConfig

        assert TrainingConfig.model_fields["drift"].default is None


class TestScenarioS1aConstantFootprint:
    def test_factory_built_memory_is_bounded_over_long_rollout(self) -> None:
        """S1a end-to-end: factory-built memory stays constant-size."""
        cfg = Settings.model_validate(
            {
                "mock_hardware": True,
                "world_model_memory": {
                    "enabled": True,
                    "recent_size": 8,
                    "sink_warmup_ticks": 0,
                },
            }
        )
        ctx = build_latent_context(cfg)
        assert isinstance(ctx, BoundedContextMemory)
        h_dim = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
        gen = torch.Generator().manual_seed(7)
        cap = 8 + 2
        for _ in range(2_000):
            ctx.observe(
                torch.randn(1, h_dim, generator=gen),
                torch.randn(1, cfg.model.latent_dim, generator=gen),
            )
            assert len(ctx) <= cap
        assert len(ctx) == cap


class TestDisabledEqualsBaseline:
    def test_default_settings_build_no_memory(self) -> None:
        cfg = Settings.model_validate({"mock_hardware": True})
        assert build_latent_context(cfg) is None

    def test_protocol_conformance(self) -> None:
        cfg = Settings.model_validate(
            {"mock_hardware": True, "world_model_memory": {"enabled": True}}
        )
        ctx = build_latent_context(cfg)
        assert isinstance(ctx, LatentContextProtocol)


class TestHardeningContracts:
    """CPU-runnable pins for the shape-guard + device-agnostic contracts.

    The CUDA-specific harmonisation is covered by ``skipif``-gated unit tests
    that never run on CPU CI; these pin the parts of the contract that ARE
    testable without a GPU so a refactor can't silently drop them.
    """

    def test_memory_rejects_multi_row_state(self) -> None:
        """A B>1 state must fail loudly, not be silently flattened into the ring."""
        cfg = WorldModelMemoryConfig.model_validate({"enabled": True})
        mem = BoundedContextMemory(cfg, h_dim=8, z_dim=4)
        with pytest.raises(ValueError, match="single carried state"):
            mem.observe(torch.zeros(2, 8), torch.zeros(2, 4))
        with pytest.raises(ValueError, match="single carried state"):
            mem.contextualize(torch.zeros(3, 8), torch.zeros(3, 4))

    def test_resolve_device_default_and_explicit(self) -> None:
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert resolve_device(None).type == expected
        assert resolve_device("auto").type == expected
        assert resolve_device("cpu").type == "cpu"

    def test_measure_drift_harmonizes_cpu_batch_noop(self) -> None:
        """A CPU model + CPU batch: harmonisation is a no-op and scoring runs."""
        mcfg = ModelConfig.model_validate(
            {
                "vision_dim": 0,
                "vision_proj_dim": 0,
                "ultrasonic_dim": 1,
                "hidden_dim": 16,
                "latent_dim": 8,
                "obs_dim": 16,
                "ultrasonic_proj_dim": 4,
                "motor_proj_dim": 8,
            }
        )
        torch.manual_seed(0)
        model = RSSM(mcfg)
        torch.manual_seed(0)
        decoders = RawModalityDecoders(mcfg)
        gen = torch.Generator().manual_seed(1)
        t = 12
        valid = torch.zeros(1, t, len(SENSOR_SLOT_MAP))
        valid[..., SENSOR_SLOT_MAP["motor"]] = 1.0
        valid[..., SENSOR_SLOT_MAP["ultrasonic"]] = 1.0
        batch = {
            "motor": torch.randn(1, t, mcfg.motor_state_dim, generator=gen),
            "ultrasonic": torch.randn(1, t, mcfg.ultrasonic_dim, generator=gen),
            "valid_mask": valid,
            "action": torch.randn(1, t, mcfg.action_dim, generator=gen),
        }
        report = measure_drift(model, batch, decoders, context_steps=4, horizon=6, seed=42)
        assert all(v >= 0.0 for curve in report.per_step_mse.values() for v in curve)

    def test_measure_drift_rejects_nonpositive_horizon(self) -> None:
        mcfg = ModelConfig.model_validate(
            {
                "vision_dim": 0,
                "vision_proj_dim": 0,
                "ultrasonic_dim": 1,
                "hidden_dim": 16,
                "latent_dim": 8,
                "obs_dim": 16,
                "ultrasonic_proj_dim": 4,
                "motor_proj_dim": 8,
            }
        )
        torch.manual_seed(0)
        model = RSSM(mcfg)
        decoders = RawModalityDecoders(mcfg)
        gen = torch.Generator().manual_seed(1)
        t = 8
        valid = torch.zeros(1, t, len(SENSOR_SLOT_MAP))
        valid[..., SENSOR_SLOT_MAP["motor"]] = 1.0
        valid[..., SENSOR_SLOT_MAP["ultrasonic"]] = 1.0
        batch = {
            "motor": torch.randn(1, t, mcfg.motor_state_dim, generator=gen),
            "ultrasonic": torch.randn(1, t, mcfg.ultrasonic_dim, generator=gen),
            "valid_mask": valid,
            "action": torch.randn(1, t, mcfg.action_dim, generator=gen),
        }
        with pytest.raises(ValueError, match="must be positive"):
            measure_drift(model, batch, decoders, context_steps=4, horizon=0, seed=42)
