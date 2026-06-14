"""Regression: WS3 on-device wiring is default-OFF + byte-identical.

Pins the backwards-compatibility invariants for Phase-6 WS3:

* an orchestrator built with ``on_device_learning`` absent (default ``None``)
  wires NO coordinator and spawns NO slow background task in ``start()``;
* the same holds when the block is present but ``enabled=False``;
* ``build_on_device_coordinator`` returns ``None`` in both cases;
* existing YAML configs load unchanged (no ``on_device_learning`` key required);
* the new ``check_interval_s`` field has a default so pre-WS3 YAML that DOES
  enable on-device learning without it still loads.
"""

from __future__ import annotations

import pytest

from mousedroid.config.schema import OnDeviceLearningConfig, Settings
from mousedroid.factory import build_on_device_coordinator, build_orchestrator


def test_default_settings_have_no_on_device_block() -> None:
    """A default Settings keeps ``on_device_learning`` at ``None``."""
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.on_device_learning is None


def test_build_coordinator_none_when_block_absent() -> None:
    """No on-device block -> no coordinator."""
    cfg = Settings.model_validate({"mock_hardware": True})
    assert build_on_device_coordinator(cfg) is None


def test_build_coordinator_none_when_disabled() -> None:
    """Block present but disabled -> no coordinator."""
    cfg = Settings.model_validate({"mock_hardware": True, "on_device_learning": {"enabled": False}})
    assert build_on_device_coordinator(cfg) is None


@pytest.mark.asyncio
async def test_orchestrator_spawns_no_slow_task_when_absent() -> None:
    """Default-OFF orchestrator never spawns the on-device slow task."""
    cfg = Settings.model_validate({"mock_hardware": True})
    orchestrator = build_orchestrator(cfg)

    assert orchestrator._on_device_coordinator is None
    await orchestrator.start()
    try:
        assert orchestrator._on_device_task is None
    finally:
        await orchestrator.stop()
    assert orchestrator._on_device_task is None


@pytest.mark.asyncio
async def test_orchestrator_spawns_no_slow_task_when_disabled() -> None:
    """Disabled block still spawns no on-device slow task."""
    cfg = Settings.model_validate({"mock_hardware": True, "on_device_learning": {"enabled": False}})
    orchestrator = build_orchestrator(cfg)

    await orchestrator.start()
    try:
        assert orchestrator._on_device_task is None
    finally:
        await orchestrator.stop()


def test_check_interval_has_default() -> None:
    """``check_interval_s`` defaults so enabling without it still validates."""
    cfg = OnDeviceLearningConfig(enabled=True)
    assert cfg.check_interval_s > 0


def test_existing_yaml_loads_without_on_device_key() -> None:
    """A pre-WS3 config dict (no on-device key) loads byte-identically."""
    cfg = Settings.model_validate({"mock_hardware": True, "experience": {"path": "/tmp/x"}})
    assert cfg.on_device_learning is None
    assert cfg.experience.path == "/tmp/x"
