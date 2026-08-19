"""Unit tests covering edge cases and error conditions in AutonomousOrchestrator."""

from __future__ import annotations

import asyncio

import pytest

from mousedroid.config.schema.root import Settings
from mousedroid.factory import build_autonomous_orchestrator
from mousedroid.hardware.lidar_driver import MockLiDAR
from mousedroid.hardware.motor_controller import MockMotorController
from mousedroid.telemetry.metrics_registry import PrometheusMetricsRegistry


@pytest.mark.asyncio
async def test_orchestrator_preflight_sensor_validation() -> None:
    """Verify preflight sensor validation passes when healthy and fails when a sensor is closed."""
    cfg = Settings(mock_hardware=True)
    metrics = PrometheusMetricsRegistry()
    orch = build_autonomous_orchestrator(cfg, metrics=metrics)

    # Initial state: all healthy
    assert orch.validate_sensors() is True

    # Close camera: preflight should fail
    await orch._camera.close()
    assert orch.validate_sensors() is False

    await orch.stop()


@pytest.mark.asyncio
async def test_orchestrator_task_cancellation_triggers_estop() -> None:
    """Verify that cancelling an active mission task immediately executes emergency stop."""
    cfg = Settings(mock_hardware=True)
    metrics = PrometheusMetricsRegistry()
    orch = build_autonomous_orchestrator(cfg, metrics=metrics)

    # Create task running the loop
    task = asyncio.create_task(orch.run_loop("patrol sector", iterations=100, interval_s=0.1))
    await asyncio.sleep(0.01)

    # Cancel the loop task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Motor must be stopped at 0.0 velocity
    assert isinstance(orch._motor, MockMotorController)
    assert orch._motor.last_linear == 0.0
    assert orch._motor.last_angular == 0.0

    await orch.stop()


@pytest.mark.asyncio
async def test_orchestrator_empty_and_corrupt_scans() -> None:
    """Verify orchestrator handles empty scan arrays gracefully."""
    cfg = Settings(mock_hardware=True)
    metrics = PrometheusMetricsRegistry()
    orch = build_autonomous_orchestrator(cfg, metrics=metrics)

    assert isinstance(orch._lidar, MockLiDAR)
    orch._lidar.set_scan([])

    ok = await orch.execute_mission_step("drive forward")
    assert ok is True

    await orch.stop()
