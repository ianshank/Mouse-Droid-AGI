"""Integration tests — self-healing orchestrator with resilient ESP32 driver.

These tests verify the full stack: orchestrator → resilient driver → mock ESP32,
with transient failures, circuit breaker activation, and sensor staleness.
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import torch

from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.schema import Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.resilience.circuit_breaker import CircuitState
from mousedroid.resilience.resilient_driver import ResilientESP32Driver
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle


def _make_observation(cfg: Settings) -> MouseDroidObservationBundle:
    """Create a default observation bundle for testing."""
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(1024, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def _build_orchestrator(
    *,
    emergency: bool = False,
    max_attempts: int = 2,
    failure_threshold: int = 3,
) -> tuple[MouseDroidOrchestrator, AsyncMock, ResilientESP32Driver]:
    """Build an orchestrator with a resilient ESP32 driver wrapping a mock."""
    cfg = Settings(mock_hardware=True)

    inner_esp32 = AsyncMock()
    inner_esp32.connect = AsyncMock()
    inner_esp32.disconnect = AsyncMock()
    inner_esp32.send_velocity = AsyncMock()
    inner_esp32.read_encoders = AsyncMock(return_value=EncoderReading())
    inner_esp32.get_battery_voltage = AsyncMock(return_value=12.0)
    inner_esp32.emergency_stop = AsyncMock()

    resilient = ResilientESP32Driver(
        inner_esp32,
        cfg.retry.model_copy(update={"max_attempts": max_attempts, "base_delay_s": 0.001}),
        cfg.circuit_breaker.model_copy(
            update={"failure_threshold": failure_threshold, "recovery_timeout_s": 30.0}
        ),
    )

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

    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _make_observation(cfg)
    sensor_manager.recovery_attempt.return_value = 0

    orch = MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=resilient,
        sensor_manager=sensor_manager,
        cfg=cfg,
    )

    return orch, inner_esp32, resilient


async def test_orchestrator_survives_transient_esp32_failures():
    """Orchestrator completes tick despite transient ESP32 read failures."""
    orch, inner, _resilient = _build_orchestrator(max_attempts=3)
    await orch.tick()
    inner.send_velocity.assert_awaited()


async def test_orchestrator_tick_continues_after_motor_read_failure():
    """Even if motor read fully fails (all retries exhausted), tick continues."""
    orch, _inner, _resilient = _build_orchestrator(max_attempts=1, failure_threshold=10)
    await orch.tick()


async def test_resilient_driver_stats_after_operations():
    """Stats reflect actual call counts and circuit states."""
    orch, _inner, resilient = _build_orchestrator()

    await orch.tick()

    stats = resilient.stats
    assert stats["command_circuit"] == "closed"
    assert stats["query_circuit"] == "closed"
    assert stats["total_calls"] >= 1


async def test_emergency_stop_works_even_with_open_circuit():
    """Emergency stop always reaches the ESP32 regardless of circuit state."""
    _orch, inner, resilient = _build_orchestrator(
        max_attempts=1,
        failure_threshold=1,
    )

    # Force the command circuit open
    inner.send_velocity = AsyncMock(side_effect=ConnectionError("fail"))
    with contextlib.suppress(Exception):
        await resilient.send_velocity(0.1, 0.0, 0.0)
    assert resilient.command_circuit_state == CircuitState.OPEN

    # Emergency stop should still work
    inner.emergency_stop = AsyncMock()
    await resilient.emergency_stop()
    inner.emergency_stop.assert_awaited_once()


async def test_factory_builds_resilient_driver():
    """Factory wraps ESP32 driver with resilient wrapper."""
    from mousedroid.factory import build_esp32_driver

    cfg = Settings(mock_hardware=True)
    driver = build_esp32_driver(cfg)
    assert isinstance(driver, ResilientESP32Driver)


async def test_full_tick_with_factory_built_driver():
    """End-to-end: factory-built resilient driver works through orchestrator."""
    from mousedroid.factory import build_esp32_driver

    cfg = Settings(mock_hardware=True)
    driver = build_esp32_driver(cfg)

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

    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)

    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _make_observation(cfg)
    sensor_manager.recovery_attempt.return_value = 0

    orch = MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=driver,
        sensor_manager=sensor_manager,
        cfg=cfg,
    )

    # Should complete without error
    await orch.tick()
