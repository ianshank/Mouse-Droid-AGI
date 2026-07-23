"""AQA regression for F-023 (AlayaWorld adaptation) — schema hygiene + contracts.

Pins the requirement-level scenarios so a future refactor cannot silently drop
them: config-field hygiene (defaults OFF, every field described), protocol
conformance, the S1a constant-footprint scenario end-to-end through the
factory, and the disabled==baseline gate.
"""

from __future__ import annotations

import torch

from mousedroid.config.schema import (
    DriftTrainingConfig,
    Settings,
    WorldModelMemoryConfig,
)
from mousedroid.factory import build_latent_context
from mousedroid.world_model.bounded_context import BoundedContextMemory
from mousedroid.world_model.protocol import LatentContextProtocol


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
