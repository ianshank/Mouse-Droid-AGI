"""Regression: WS4 safety-gate wiring is default-OFF + byte-identical.

Pins the backwards-compatibility invariants for Phase-6 WS4:

* an orchestrator/coordinator built with ``on_device_learning`` absent or
  disabled wires NO gate (and no coordinator at all) — byte-identical to pre-WS4;
* the WS4 scoring seed (``scoring_seed``) has a default, so a pre-WS4 YAML that
  enables on-device learning WITHOUT it still loads + validates;
* existing YAML configs (no ``on_device_learning`` key) load unchanged;
* a coordinator built with on-device learning enabled but no slow cadence run
  never marks any slot active (no implicit promotion at build time).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.schema import OnDeviceLearningConfig, Settings
from mousedroid.factory import build_on_device_coordinator
from mousedroid.learning.on_device.replay_trigger import ReplayTriggerCoordinator
from mousedroid.learning.on_device.slot_store import OnDeviceSlotStore

# Pins the concrete ``OnDeviceLearningConfig.scoring_seed`` default declared in
# ``config/schema.py``. A backwards-compat regression test asserts the EXACT
# default (not merely the type) so silent default drift fails CI: changing a
# config default is a behaviour change (CLAUDE.md config convention). Update BOTH
# this constant and the schema Field together if the default is ever retuned.
_EXPECTED_SCORING_SEED_DEFAULT = 1234


def test_ws4_fields_have_defaults() -> None:
    """The WS4 scoring seed defaults so enabling without it validates."""
    cfg = OnDeviceLearningConfig(enabled=True)
    assert isinstance(cfg.scoring_seed, int)
    assert cfg.scoring_seed == _EXPECTED_SCORING_SEED_DEFAULT


def test_pre_ws4_enabled_yaml_loads_without_scoring_knobs() -> None:
    """A pre-WS4 enabled block (no scoring knobs) still loads byte-identically."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "on_device_learning": {"enabled": True, "trigger_min_new_records": 10},
        }
    )
    assert cfg.on_device_learning is not None
    assert isinstance(cfg.on_device_learning.scoring_seed, int)
    assert cfg.on_device_learning.scoring_seed == _EXPECTED_SCORING_SEED_DEFAULT


def test_existing_yaml_loads_without_on_device_key(tmp_path: Path) -> None:
    """A config with no on-device key keeps the block ``None`` (pre-WS4)."""
    cfg = Settings.model_validate(
        {"mock_hardware": True, "experience": {"path": str(tmp_path / "exp")}}
    )
    assert cfg.on_device_learning is None


def test_no_coordinator_when_block_absent() -> None:
    """No on-device block -> no coordinator (so no gate runs)."""
    cfg = Settings.model_validate({"mock_hardware": True})
    assert build_on_device_coordinator(cfg) is None
    assert build_on_device_coordinator(cfg, metrics=None) is None


def test_no_coordinator_when_disabled() -> None:
    """Disabled block -> no coordinator regardless of metrics arg."""
    cfg = Settings.model_validate({"mock_hardware": True, "on_device_learning": {"enabled": False}})
    assert build_on_device_coordinator(cfg) is None


def test_coordinator_has_gate_runner_when_enabled(tmp_path: Path) -> None:
    """When enabled, the coordinator is wired with a WS4 gate runner."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": str(tmp_path / "exp"), "map_size_gb": 0.01},
            "on_device_learning": {"enabled": True, "trigger_min_new_records": 5},
        }
    )
    coordinator = build_on_device_coordinator(cfg)
    assert isinstance(coordinator, ReplayTriggerCoordinator)
    assert coordinator._gate_runner is not None  # WS4 gate wired


def test_build_does_not_mark_any_slot_active(tmp_path: Path) -> None:
    """Building the coordinator never promotes a slot (no implicit activation)."""
    exp_path = str(tmp_path / "exp")
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": exp_path, "map_size_gb": 0.01},
            "on_device_learning": {"enabled": True, "trigger_min_new_records": 5},
        }
    )
    build_on_device_coordinator(cfg)
    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)
    assert store.load_active() is None


@pytest.mark.asyncio
async def test_disabled_coordinator_runs_no_scoring(tmp_path: Path) -> None:
    """A disabled on-device block spawns no coordinator -> no scoring at all."""
    cfg = Settings.model_validate({"mock_hardware": True, "on_device_learning": {"enabled": False}})
    assert build_on_device_coordinator(cfg, metrics=None) is None
