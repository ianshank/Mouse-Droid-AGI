"""Regression: Phase-6 on-device-learning config is purely additive (WS1).

Guards the CLAUDE.md invariant "New config fields MUST have defaults; existing
YAML files must load unchanged". A Settings built without an
``on_device_learning`` block must leave the field ``None`` (default-off,
byte-identical), and a legacy YAML without the block must validate.
"""

from __future__ import annotations

import yaml

from mousedroid.config.schema import OnDeviceLearningConfig, Settings


def test_settings_without_block_is_none() -> None:
    s = Settings.model_validate({"mock_hardware": True})
    assert s.on_device_learning is None


def test_legacy_yaml_loads_without_on_device_learning() -> None:
    legacy_yaml = """
    mock_hardware: true
    platform: mouse_droid
    experience:
      path: /home/jetson/mousedroid_experience
    """
    data = yaml.safe_load(legacy_yaml)
    s = Settings.model_validate(data)
    assert s.on_device_learning is None
    # The experience root the slot derives from is untouched.
    assert s.experience.path == "/home/jetson/mousedroid_experience"


def test_block_present_round_trips() -> None:
    s = Settings.model_validate({"mock_hardware": True, "on_device_learning": {"enabled": True}})
    assert s.on_device_learning is not None
    assert s.on_device_learning.enabled is True
    # Unspecified fields fall back to defaults.
    assert s.on_device_learning.trigger_min_new_records == 500


def test_model_validate_empty_block_uses_all_defaults() -> None:
    cfg = OnDeviceLearningConfig.model_validate({})
    assert cfg.enabled is False
    assert cfg.slot_dir == "on_device_slot"
