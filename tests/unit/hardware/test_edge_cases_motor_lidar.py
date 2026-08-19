"""Unit tests covering edge cases in MotorController and LiDARScanner."""

from __future__ import annotations

import pytest

from mousedroid.config.schema.hardware import LidarConfig, MotorControllerConfig, MotorLimitsConfig
from mousedroid.hardware.lidar_driver import LiDARScanner, MockLiDAR
from mousedroid.hardware.motor_controller import MockMotorController, MotorController
from mousedroid.telemetry.metrics_registry import PrometheusMetricsRegistry


@pytest.mark.asyncio
async def test_motor_controller_nan_inf_inputs() -> None:
    """Verify that NaN and Inf velocity inputs are safely sanitized to 0.0."""
    limits = MotorLimitsConfig(max_linear_velocity=1.0, max_angular_velocity=2.0)
    cfg = MotorControllerConfig(limits=limits)
    metrics = PrometheusMetricsRegistry()
    motor = MotorController(cfg=cfg, port="/dev/ttyUSB0", metrics=metrics)

    # NaN inputs
    ok_nan = await motor.set_velocity(float("nan"), float("nan"))
    assert ok_nan is True

    # Inf inputs
    ok_inf = await motor.set_velocity(float("inf"), float("-inf"))
    assert ok_inf is True

    # MockMotorController with NaN
    mock_motor = MockMotorController(metrics=metrics)
    await mock_motor.set_velocity(float("nan"), float("inf"))
    assert mock_motor.last_linear == 0.0
    assert mock_motor.last_angular == 0.0

    await motor.close()
    await mock_motor.close()


@pytest.mark.asyncio
async def test_lidar_scanner_nan_inf_and_negative_filtering() -> None:
    """Verify that NaN, Inf, and negative range values in LiDAR scans are sanitized."""
    cfg = LidarConfig(min_range_m=0.2, max_range_m=10.0)
    lidar = LiDARScanner(cfg=cfg, port="/dev/ttyUSB1", buffer_size=50)

    raw_noisy = [float("nan"), float("inf"), -1.5, 3.0, 15.0]
    sanitized = lidar._sanitize_scan(raw_noisy)

    assert sanitized[0] == 10.0  # NaN replaced by max_dist
    assert sanitized[1] == 10.0  # Inf replaced by max_dist
    assert sanitized[2] == 10.0  # Negative replaced by max_dist
    assert sanitized[3] == 3.0  # Normal value preserved
    assert sanitized[4] == 10.0  # Out-of-bounds clamped to max_dist

    # Empty raw scan test
    empty_sanitized = lidar._sanitize_scan([])
    assert len(empty_sanitized) == 360
    assert empty_sanitized[0] == 10.0

    await lidar.close()
    assert lidar.is_healthy() is False


@pytest.mark.asyncio
async def test_mock_lidar_edge_cases() -> None:
    """Verify MockLiDAR closed state and custom scan injection."""
    mock_lidar = MockLiDAR(default_distance=4.0)
    assert mock_lidar.is_healthy() is True

    scan1 = await mock_lidar.get_latest_scan()
    assert len(scan1) == 360
    assert scan1[0] == 4.0

    mock_lidar.set_scan([1.0] * 360)
    scan2 = await mock_lidar.get_latest_scan()
    assert scan2[0] == 1.0

    await mock_lidar.close()
    assert mock_lidar.is_healthy() is False
    assert await mock_lidar.get_latest_scan() == []
