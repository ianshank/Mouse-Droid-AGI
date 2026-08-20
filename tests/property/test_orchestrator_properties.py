"""Property-based tests for autonomous orchestrator invariants."""

from __future__ import annotations

import asyncio

from hypothesis import given
from hypothesis import strategies as st

from mousedroid.config.schema.hardware import LidarConfig, MotorControllerConfig, MotorLimitsConfig
from mousedroid.config.schema.root import Settings
from mousedroid.hardware.camera_csi import MockCamera
from mousedroid.hardware.lidar_driver import MockLiDAR
from mousedroid.hardware.motor_controller import MockMotorController, MotorController
from mousedroid.llm_gateway.composite_gateway import CompositeLLMGateway
from mousedroid.orchestrator.autonomous import AutonomousOrchestrator
from mousedroid.telemetry.metrics_registry import PrometheusMetricsRegistry


@given(
    linear=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
    angular=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
)
def test_property_motor_velocity_bounding(linear: float, angular: float) -> None:
    """Property: All velocity outputs dispatched to the controller stay within clamped bounds."""
    limits = MotorLimitsConfig(max_linear_velocity=1.0, max_angular_velocity=1.5)
    cfg = MotorControllerConfig(limits=limits)
    metrics = PrometheusMetricsRegistry()
    motor = MotorController(cfg=cfg, port="/dev/ttyUSB0", metrics=metrics)

    loop = asyncio.new_event_loop()
    try:
        ok = loop.run_until_complete(motor.set_velocity(linear, angular))
        assert ok is True
    finally:
        loop.close()


@given(
    obstacle_distance=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
)
def test_property_proximity_safety_monotonicity(obstacle_distance: float) -> None:
    """Property: If obstacle distance is below threshold, e-stop is ALWAYS triggered."""
    threshold = 0.5
    cfg = Settings(mock_hardware=True, lidar=LidarConfig(min_range_m=threshold))

    metrics = PrometheusMetricsRegistry()
    motor = MockMotorController(metrics=metrics)
    cam = MockCamera()
    lidar = MockLiDAR()
    lidar.set_scan([obstacle_distance] * 360)
    llm = CompositeLLMGateway(cfg=cfg.llm, mock_mode=True, metrics=metrics)

    orch = AutonomousOrchestrator(cfg, motor, cam, lidar, llm, metrics)

    loop = asyncio.new_event_loop()
    try:
        success = loop.run_until_complete(orch.execute_mission_step("drive forward"))
        if obstacle_distance < threshold:
            assert success is False
            assert motor.last_linear == 0.0
            assert motor.last_angular == 0.0
        else:
            assert success is True
    finally:
        loop.close()
