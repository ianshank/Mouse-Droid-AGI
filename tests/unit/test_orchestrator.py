"""Tests for MouseDroidOrchestrator — full coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import torch

from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.schema import Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext


def _make_orchestrator(
    *,
    emergency: bool = False,
) -> MouseDroidOrchestrator:
    cfg = Settings(mock_hardware=True)

    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )

    agent = MagicMock()
    agent.name = "test_agent"
    agent.act.return_value = torch.tensor([0.1, 0.0, 0.0])

    safety_ctx = SafetyContext(is_emergency=emergency)
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = safety_ctx

    esp32 = AsyncMock()
    esp32.read_encoders.return_value = EncoderReading()
    esp32.get_battery_voltage.return_value = 12.0

    camera = AsyncMock()
    camera.capture_features.return_value = np.zeros(cfg.camera.feature_dim, dtype=np.float32)

    distance_sensor = MagicMock()
    distance_sensor.max_range_m = 4.0
    distance_sensor.read_distance_m = AsyncMock(return_value=1.5)

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


async def test_tick_full_cycle():
    orch = _make_orchestrator()
    await orch.tick()
    orch._esp32.send_velocity.assert_awaited_once()


async def test_tick_emergency_stop():
    orch = _make_orchestrator(emergency=True)
    await orch.tick()
    orch._esp32.emergency_stop.assert_awaited_once()
    orch._esp32.send_velocity.assert_not_awaited()


async def test_sense_vision_failure():
    orch = _make_orchestrator()
    orch._camera.capture_features.side_effect = RuntimeError("camera fail")
    obs = await orch._sense()
    np.testing.assert_array_equal(
        obs.vision_features,
        np.zeros(orch._cfg.camera.feature_dim, dtype=np.float32),
    )
    assert obs.valid_mask[0] == 0.0


async def test_sense_distance_failure():
    orch = _make_orchestrator()
    orch._distance_sensor.read_distance_m = AsyncMock(
        side_effect=RuntimeError("sensor fail"),
    )
    obs = await orch._sense()
    assert obs.distance_m == orch._distance_sensor.max_range_m
    assert obs.valid_mask[1] == 0.0


async def test_sense_motor_failure():
    orch = _make_orchestrator()
    orch._esp32.read_encoders.side_effect = RuntimeError("motor fail")
    obs = await orch._sense()
    np.testing.assert_array_equal(obs.motor_state, np.zeros(4, dtype=np.float32))
    assert obs.valid_mask[2] == 0.0


async def test_run_loop_single_iteration():
    orch = _make_orchestrator()
    orch._running = True
    tick_count = 0
    original_tick = orch.tick

    async def counting_tick():
        nonlocal tick_count
        await original_tick()
        tick_count += 1
        orch._running = False

    orch.tick = counting_tick  # type: ignore[assignment]
    await orch.run()
    assert tick_count == 1


async def test_run_loop_tick_exception_continues():
    orch = _make_orchestrator()
    orch._running = True
    call_count = 0

    async def failing_tick():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("tick error")
        orch._running = False

    orch.tick = failing_tick  # type: ignore[assignment]
    await orch.run()
    assert call_count == 2


async def test_tick_action_1d():
    orch = _make_orchestrator()
    orch._agents[0].act.return_value = torch.tensor([0.5])
    await orch.tick()
    orch._esp32.send_velocity.assert_awaited_once()


async def test_tick_action_2d():
    orch = _make_orchestrator()
    orch._agents[0].act.return_value = torch.tensor([0.5, 0.3])
    await orch.tick()
    orch._esp32.send_velocity.assert_awaited_once()
