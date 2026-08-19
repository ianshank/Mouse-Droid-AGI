"""End-to-end tests for the complete Autonomous Orchestration pipeline."""

from __future__ import annotations

import pytest

from mousedroid.config.schema.root import Settings
from mousedroid.factory import build_autonomous_orchestrator
from mousedroid.hardware.motor_controller import MockMotorController
from mousedroid.telemetry.metrics_registry import PrometheusMetricsRegistry


@pytest.mark.asyncio
async def test_e2e_autonomous_mission_run() -> None:
    """E2E test: Execute full mission multi-step cycle with simulated perception and actuation."""
    cfg = Settings(mock_hardware=True)
    metrics = PrometheusMetricsRegistry()
    orch = build_autonomous_orchestrator(cfg, metrics=metrics)

    # 1. Forward step
    ok_step1 = await orch.execute_mission_step("drive forward to waypoint 1")
    assert ok_step1 is True

    # 2. Turn step
    ok_step2 = await orch.execute_mission_step("turn left toward dock")
    assert ok_step2 is True

    # 3. Stop step (triggers intentional e-stop / neutral stop)
    ok_step3 = await orch.execute_mission_step("halt and stand by")
    assert ok_step3 is False
    assert isinstance(orch._motor, MockMotorController)
    assert orch._motor.last_linear == 0.0

    # 4. Multi-iteration mission run
    safe_steps = await orch.run_loop("patrol sector A", iterations=5, interval_s=0.001)
    assert safe_steps == 5

    # 5. Teardown and metric verification
    await orch.stop()
    prom = metrics.render_prometheus()
    assert "mousedroid_motor_commands_total" in prom
