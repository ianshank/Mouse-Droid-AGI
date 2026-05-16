"""Backwards-compat tests for the C1.2 world_model_enabled toggle."""

from __future__ import annotations

from mousedroid.config.schema import Settings, WeightUpdatePollConfig


def test_world_model_enabled_defaults_false_for_backwards_compat() -> None:
    cfg = WeightUpdatePollConfig()
    assert cfg.world_model_enabled is False


def test_world_model_enabled_can_be_toggled_on() -> None:
    cfg = WeightUpdatePollConfig(world_model_enabled=True)
    assert cfg.world_model_enabled is True


def test_existing_yaml_loads_without_world_model_enabled() -> None:
    """Existing YAML files (pre-C1.2) must still load with the default value."""
    settings = Settings()
    assert settings.cloud.weight_update.world_model_enabled is False
