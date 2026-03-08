"""Tests for MouseDroidOrchestrator health_check."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mousedroid.config.schema import Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator


def _make_orchestrator() -> MouseDroidOrchestrator:
    cfg = Settings(mock_hardware=True)

    world_model = MagicMock()
    agent = MagicMock()
    agent.name = "test_agent"
    safety_monitor = MagicMock()
    esp32 = AsyncMock()
    camera = AsyncMock()
    distance_sensor = MagicMock()
    distance_sensor.max_range_m = 4.0

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        camera=camera,
        distance_sensor=distance_sensor,
        cfg=cfg,
    )


def test_constructor():
    orch = _make_orchestrator()
    assert orch._running is False


async def test_health_check():
    orch = _make_orchestrator()
    result = await orch.health_check()
    assert result["status"] == "ok"
    assert result["mock_hardware"] is True
    assert "test_agent" in result["agents"]


async def test_start_sets_running():
    orch = _make_orchestrator()
    await orch.start()
    assert orch._running is True


async def test_stop_clears_running():
    orch = _make_orchestrator()
    await orch.start()
    await orch.stop()
    assert orch._running is False
