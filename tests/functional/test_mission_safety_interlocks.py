"""Functional tests for mission dispatch and safety interlock triggers."""

from __future__ import annotations

import pytest

from mousedroid.config.schema.root import Settings
from mousedroid.factory import build_autonomous_orchestrator
from mousedroid.hardware.lidar_driver import MockLiDAR
from mousedroid.hardware.motor_controller import MockMotorController
from mousedroid.telemetry.metrics_registry import PrometheusMetricsRegistry


@pytest.mark.asyncio
async def test_functional_clear_path_mission() -> None:
    """Functional test: Clear path allows smooth traversal and command dispatch."""
    cfg = Settings(mock_hardware=True)
    metrics = PrometheusMetricsRegistry()
    orch = build_autonomous_orchestrator(cfg, metrics=metrics)

    ok = await orch.execute_mission_step("drive forward carefully")
    assert ok is True
    assert isinstance(orch._motor, MockMotorController)
    assert orch._motor.last_linear > 0.0

    await orch.stop()


@pytest.mark.asyncio
async def test_functional_blocked_path_interlock() -> None:
    """Functional test: Obstacle in proximity triggers immediate e-stop."""
    cfg = Settings(mock_hardware=True)
    metrics = PrometheusMetricsRegistry()
    orch = build_autonomous_orchestrator(cfg, metrics=metrics)

    assert isinstance(orch._lidar, MockLiDAR)
    orch._lidar.set_scan([0.05] * 360)

    ok = await orch.execute_mission_step("drive forward fast")
    assert ok is False
    assert isinstance(orch._motor, MockMotorController)
    assert orch._motor.last_linear == 0.0

    prom = metrics.render_prometheus()
    assert "mousedroid_safety_interventions_total" in prom
    assert 'cause="proximity"' in prom

    await orch.stop()
