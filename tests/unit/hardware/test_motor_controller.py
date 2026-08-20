"""Unit tests for MotorController and MockMotorController."""

from __future__ import annotations

import pytest

from mousedroid.config.schema.hardware import MotorControllerConfig, MotorLimitsConfig
from mousedroid.hardware.motor_controller import MockMotorController, MotorController
from mousedroid.telemetry.metrics_registry import PrometheusMetricsRegistry


@pytest.mark.asyncio
async def test_motor_controller_clamping() -> None:
    """Verify velocity commands are strictly clamped to safety limits."""
    limits = MotorLimitsConfig(max_linear_velocity=1.0, max_angular_velocity=1.5)
    cfg = MotorControllerConfig(limits=limits)
    metrics = PrometheusMetricsRegistry()
    controller = MotorController(cfg=cfg, port="/dev/ttyUSB0", metrics=metrics)

    assert controller.is_healthy() is True

    # Command exceeding limits
    ok = await controller.set_velocity(5.0, -10.0)
    assert ok is True

    # Emergency stop
    await controller.emergency_stop()
    assert controller.is_healthy() is True

    # Teardown
    await controller.close()
    assert controller.is_healthy() is False

    prom = metrics.render_prometheus()
    assert "mousedroid_motor_commands_total" in prom


@pytest.mark.asyncio
async def test_mock_motor_controller() -> None:
    """Verify MockMotorController registers commands accurately."""
    mock_ctrl = MockMotorController()
    assert mock_ctrl.is_healthy() is True

    await mock_ctrl.set_velocity(0.5, 0.2)
    assert mock_ctrl.last_linear == 0.5
    assert mock_ctrl.last_angular == 0.2

    await mock_ctrl.emergency_stop()
    assert mock_ctrl.last_linear == 0.0
    assert mock_ctrl.last_angular == 0.0

    await mock_ctrl.close()
    assert mock_ctrl.is_healthy() is False
    assert mock_ctrl.closed is True
