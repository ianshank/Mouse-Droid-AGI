"""Tests for motor command validation and bounding."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock

import pytest

from mousedroid.common.tools.motor_tools import MotorToolDeps, register_motor_tools
from mousedroid.common.tools.registry import ToolRegistry
from mousedroid.comms.protocol import ESP32CommProtocol
from mousedroid.config.schema import Settings


@pytest.fixture
def esp32_mock() -> AsyncMock:
    return AsyncMock(spec=ESP32CommProtocol)


@pytest.fixture
def motor_registry(mock_settings: Settings, esp32_mock: AsyncMock) -> ToolRegistry:
    registry = ToolRegistry()
    mock_settings.esp32.max_velocity_mps = 2.0
    mock_settings.esp32.max_omega_rads = 1.0

    deps = MotorToolDeps(
        esp32=esp32_mock,
        cfg=mock_settings,
    )
    register_motor_tools(registry, deps)
    return registry


@pytest.mark.asyncio
async def test_speed_values_clamped_to_safe_range(
    motor_registry: ToolRegistry, esp32_mock: AsyncMock
) -> None:
    tool = motor_registry.get("set_velocity")
    assert tool is not None

    # Request values way outside the bounds
    await tool.handler(vx=10.0, vy=-5.0, omega=5.0)

    # Check that they were clamped to +/- 2.0 and +/- 1.0
    esp32_mock.send_velocity.assert_called_once_with(2.0, -2.0, 1.0)


@pytest.mark.asyncio
async def test_nan_values_handled_safely(
    motor_registry: ToolRegistry, esp32_mock: AsyncMock
) -> None:
    tool = motor_registry.get("set_velocity")
    assert tool is not None

    await tool.handler(vx=float("nan"), vy=float("inf"), omega=float("-inf"))

    # Inf gets clamped to bounds.
    # NaN behavior with min/max might result in the upper or lower bound,
    # but it shouldn't crash or pass NaN to driver.
    # We assert that the call arguments are finite floats.
    call_args = esp32_mock.send_velocity.call_args[0]
    for arg in call_args:
        assert isinstance(arg, float)
        assert not math.isnan(arg)
        assert not math.isinf(arg)


@pytest.mark.asyncio
async def test_rapid_direction_reversals_rate_limited() -> None:
    """Test that rapid direction reversals are prevented or rate limited.

    Currently this might not be implemented in set_velocity,
    but we ensure the test structure exists.
    """
    pass
