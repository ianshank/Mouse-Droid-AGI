"""Tests for orchestrator telemetry integration."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import torch

from mousedroid.config.schema import Settings
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.telemetry.protocol import TelemetryFrame, TelemetryPublisherProtocol


def _make_settings() -> Settings:
    os.environ["MOUSEDROID_MOCK_HARDWARE"] = "true"
    return Settings(mock_hardware=True)


def _make_orchestrator(telemetry_publisher=None, telemetry_server=None):
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    cfg = _make_settings()
    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        None,
        None,
    )

    agent = MagicMock()
    agent.name = "test_agent"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)

    safety = MagicMock()
    safety_ctx = MagicMock()
    safety_ctx.is_emergency = False
    safety_ctx.law_violations = ()
    safety_ctx.forward_clearance_ok = True
    safety.evaluate.return_value = safety_ctx

    esp32 = AsyncMock()
    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = MouseDroidObservationBundle()

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        telemetry_publisher=telemetry_publisher,
        telemetry_server=telemetry_server,
    )


async def test_tick_without_publisher():
    """Orchestrator works normally without telemetry publisher."""
    orch = _make_orchestrator()
    await orch.tick()
    assert orch._tick_count == 1


async def test_tick_publishes_frame():
    """Orchestrator publishes a TelemetryFrame when publisher is present."""
    publisher = AsyncMock(spec=TelemetryPublisherProtocol)
    orch = _make_orchestrator(telemetry_publisher=publisher)
    await orch.tick()
    publisher.publish.assert_called_once()
    frame = publisher.publish.call_args[0][0]
    assert isinstance(frame, TelemetryFrame)
    assert frame.tick_count == 1


async def test_tick_count_increments():
    publisher = AsyncMock(spec=TelemetryPublisherProtocol)
    orch = _make_orchestrator(telemetry_publisher=publisher)
    await orch.tick()
    await orch.tick()
    assert orch._tick_count == 2
    second_frame = publisher.publish.call_args_list[1][0][0]
    assert second_frame.tick_count == 2


async def test_frame_contains_sensor_data():
    publisher = AsyncMock(spec=TelemetryPublisherProtocol)
    orch = _make_orchestrator(telemetry_publisher=publisher)
    await orch.tick()
    frame = publisher.publish.call_args[0][0]
    assert isinstance(frame.motor_state, list)
    assert isinstance(frame.valid_mask, list)
    assert isinstance(frame.vision_norm, float)
    assert isinstance(frame.audio_rms, float)


async def test_frame_contains_safety_data():
    publisher = AsyncMock(spec=TelemetryPublisherProtocol)
    orch = _make_orchestrator(telemetry_publisher=publisher)
    await orch.tick()
    frame = publisher.publish.call_args[0][0]
    assert "is_emergency" in frame.safety
    assert "violations" in frame.safety


async def test_publisher_error_does_not_crash_tick():
    """A failing publisher must not break the control loop."""
    publisher = AsyncMock(spec=TelemetryPublisherProtocol)
    publisher.publish.side_effect = RuntimeError("boom")
    orch = _make_orchestrator(telemetry_publisher=publisher)
    # Should not raise
    await orch.tick()
    assert orch._tick_count == 1


async def test_start_starts_telemetry_server():
    server = AsyncMock()
    orch = _make_orchestrator(telemetry_server=server)
    await orch.start()
    server.start.assert_called_once()
    await orch.stop()


async def test_stop_stops_telemetry_server():
    server = AsyncMock()
    orch = _make_orchestrator(telemetry_server=server)
    await orch.start()
    await orch.stop()
    server.stop.assert_called_once()
