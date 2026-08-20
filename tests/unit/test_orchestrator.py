"""Comprehensive unit tests for AutonomousOrchestrator."""

from __future__ import annotations

import pytest

from mousedroid.config.schema.hardware import LidarConfig
from mousedroid.config.schema.root import Settings
from mousedroid.hardware.camera_csi import MockCamera
from mousedroid.hardware.lidar_driver import MockLiDAR
from mousedroid.hardware.motor_controller import MockMotorController
from mousedroid.interfaces.protocols import GoalVector
from mousedroid.llm_gateway.composite_gateway import CompositeLLMGateway
from mousedroid.orchestrator.autonomous import AutonomousOrchestrator
from mousedroid.telemetry.metrics_registry import PrometheusMetricsRegistry


@pytest.mark.asyncio
async def test_orchestrator_full_mission_step_success() -> None:
    """Verify normal mission step execution."""
    cfg = Settings(mock_hardware=True)
    metrics = PrometheusMetricsRegistry()
    motor = MockMotorController(metrics=metrics)
    cam = MockCamera()
    lidar = MockLiDAR()
    llm = CompositeLLMGateway(cfg=cfg.llm, mock_mode=True, metrics=metrics)

    orch = AutonomousOrchestrator(cfg, motor, cam, lidar, llm, metrics)

    success = await orch.execute_mission_step("Navigate to sector B")
    assert success is True
    assert orch.is_running is False
    await orch.stop()


@pytest.mark.asyncio
async def test_orchestrator_proximity_safety_stop() -> None:
    """Verify immediate e-stop when obstacle violates safety clearance."""
    cfg = Settings(mock_hardware=True, lidar=LidarConfig(min_range_m=0.5))
    metrics = PrometheusMetricsRegistry()
    motor = MockMotorController(metrics=metrics)
    cam = MockCamera()
    lidar = MockLiDAR()
    lidar.set_scan([0.1] * 360)
    llm = CompositeLLMGateway(cfg=cfg.llm, mock_mode=True, metrics=metrics)

    orch = AutonomousOrchestrator(cfg, motor, cam, lidar, llm, metrics)

    success = await orch.execute_mission_step("Drive forward fast")
    assert success is False
    prom_metrics = metrics.render_prometheus()
    assert "mousedroid_safety_interventions_total" in prom_metrics
    assert 'cause="proximity"' in prom_metrics
    await orch.stop()


@pytest.mark.asyncio
async def test_orchestrator_unsafe_goal_trigger() -> None:
    """Verify e-stop when LLM returns an unsafe goal vector."""
    cfg = Settings(mock_hardware=True)
    metrics = PrometheusMetricsRegistry()
    motor = MockMotorController(metrics=metrics)
    cam = MockCamera()
    lidar = MockLiDAR()
    llm = CompositeLLMGateway(cfg=cfg.llm, mock_mode=True, metrics=metrics)

    async def _mock_unsafe_translate(cmd: str) -> GoalVector:
        return GoalVector(
            linear_velocity=1.0,
            angular_velocity=0.0,
            arm_action="e_stop",
            is_safe=False,
        )

    llm.translate_mission = _mock_unsafe_translate  # type: ignore[assignment]

    orch = AutonomousOrchestrator(cfg, motor, cam, lidar, llm, metrics)
    success = await orch.execute_mission_step("Trigger e-stop")
    assert success is False
    prom_metrics = metrics.render_prometheus()
    assert 'cause="unsafe_goal"' in prom_metrics
    await orch.stop()


@pytest.mark.asyncio
async def test_orchestrator_run_loop() -> None:
    """Verify loop execution over multiple iterations."""
    cfg = Settings(mock_hardware=True)
    metrics = PrometheusMetricsRegistry()
    motor = MockMotorController(metrics=metrics)
    cam = MockCamera()
    lidar = MockLiDAR()
    llm = CompositeLLMGateway(cfg=cfg.llm, mock_mode=True, metrics=metrics)

    orch = AutonomousOrchestrator(cfg, motor, cam, lidar, llm, metrics)
    safe_steps = await orch.run_loop("patrol sector", iterations=3, interval_s=0.001)
    assert safe_steps == 3
    await orch.stop()
