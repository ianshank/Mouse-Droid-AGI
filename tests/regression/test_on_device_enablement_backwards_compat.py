"""Regression: Phase-6 ENABLEMENT (WS-E0) is additive + default-OFF byte-identical.

Pins the WS-E0 backwards-compatibility invariants:

* a pre-enablement ``on_device_learning`` block (none of the four new keys) loads
  with the new fields at their defaults — existing YAML is unchanged;
* ``Settings(...).on_device_learning`` round-trips the new fields;
* a config with no ``on_device_learning`` key keeps the block ``None``;
* a DISABLED block ⇒ ``build_on_device_coordinator(cfg, world_model=<wm>)``
  returns ``None`` exactly as before the new ``world_model`` kwarg existed
  (the kwarg never resurrects a coordinator the master switch turned off).
"""

from __future__ import annotations

from pathlib import Path

from mousedroid.config.schema import OnDeviceLearningConfig, Settings
from mousedroid.factory import build_on_device_coordinator, build_world_model


def test_pre_enablement_block_loads_with_new_fields_at_defaults() -> None:
    """A pre-WS-E0 enabled block (no new keys) loads with defaults applied."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "on_device_learning": {"enabled": True, "trigger_min_new_records": 10},
        }
    )
    block = cfg.on_device_learning
    assert block is not None
    # The new enablement fields are present at their default values.
    assert block.enable_hot_swap is False
    assert block.refine_sequence_length == 16
    assert block.refine_batch_episodes == 4


def test_new_fields_round_trip() -> None:
    """The new fields round-trip through model_validate (set values are kept)."""
    block = Settings.model_validate(
        {
            "mock_hardware": True,
            "on_device_learning": {
                "enabled": True,
                "enable_hot_swap": True,
                "refine_sequence_length": 32,
                "refine_batch_episodes": 8,
            },
        }
    ).on_device_learning
    assert block is not None
    assert block.enable_hot_swap is True
    assert block.refine_sequence_length == 32
    assert block.refine_batch_episodes == 8


def test_existing_yaml_without_on_device_key_is_none(tmp_path: Path) -> None:
    """A config without the on-device key keeps the block ``None`` (pre-enablement)."""
    cfg = Settings.model_validate(
        {"mock_hardware": True, "experience": {"path": str(tmp_path / "exp")}}
    )
    assert cfg.on_device_learning is None


def test_bare_on_device_config_defaults() -> None:
    """An ``OnDeviceLearningConfig()`` carries the new fields at defaults."""
    block = OnDeviceLearningConfig()
    assert block.enable_hot_swap is False
    assert block.refine_sequence_length == 16
    assert block.refine_batch_episodes == 4


def test_disabled_block_returns_none_with_world_model_kwarg(tmp_path: Path) -> None:
    """The new ``world_model`` kwarg never resurrects a disabled coordinator."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": str(tmp_path / "exp"), "map_size_gb": 0.01},
            "on_device_learning": {"enabled": False},
        }
    )
    wm = build_world_model(cfg)
    assert build_on_device_coordinator(cfg, world_model=wm) is None
    # Absent block too.
    cfg_absent = Settings.model_validate({"mock_hardware": True})
    assert build_on_device_coordinator(cfg_absent, world_model=wm) is None
