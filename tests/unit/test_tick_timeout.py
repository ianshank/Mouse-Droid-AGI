"""Tests for tick timeout triggering emergency stop.

Validates Phase 1B: ``asyncio.wait_for`` wrapping in ``orchestrator.run()``
fires ``esp32.emergency_stop()`` when a tick exceeds ``tick_timeout_s``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mousedroid.config.schema import Settings


def _build_mock_orchestrator(
    *,
    tick_timeout_s: float = 0.1,
    tick_side_effect: object | None = None,
) -> tuple[object, AsyncMock]:
    """Build a MockOrchestrator with injectable tick behaviour.

    Returns:
        (orchestrator, esp32_mock) tuple.
    """
    from mousedroid.factory import build_orchestrator

    cfg = Settings(mock_hardware=True)
    # Override tick timeout for test speed
    cfg.loop.tick_timeout_s = tick_timeout_s  # type: ignore[misc]

    orch = build_orchestrator(cfg)

    # Replace ESP32 with a mock so we can assert on emergency_stop
    esp32_mock = AsyncMock()
    esp32_mock.emergency_stop = AsyncMock()
    esp32_mock.send_velocity = AsyncMock()
    esp32_mock.connect = AsyncMock()
    esp32_mock.disconnect = AsyncMock()
    esp32_mock.read_encoders = AsyncMock(
        return_value=MagicMock(
            left_velocity_mps=0.0,
            right_velocity_mps=0.0,
            heading_rad=0.0,
        )
    )
    esp32_mock.get_battery_voltage = AsyncMock(return_value=12.0)
    orch._esp32 = esp32_mock

    # Replace sensor_manager to avoid hardware
    orch._sensor_manager = AsyncMock()
    orch._sensor_manager.read_all = AsyncMock(
        return_value=MagicMock(valid_mask=0xFF, valid_sensor_count=4)
    )
    orch._sensor_manager.start = AsyncMock()
    orch._sensor_manager.stop = AsyncMock()
    orch._sensor_manager.recovery_attempt = AsyncMock(return_value=0)

    # Inject tick side effect
    if tick_side_effect is not None:
        orch.tick = AsyncMock(side_effect=tick_side_effect)

    return orch, esp32_mock


async def test_tick_timeout_triggers_emergency_stop() -> None:
    """A tick that exceeds tick_timeout_s triggers emergency_stop."""

    async def slow_tick() -> None:
        await asyncio.sleep(5.0)  # Much longer than 0.1s timeout

    orch, esp32 = _build_mock_orchestrator(
        tick_timeout_s=0.05,
        tick_side_effect=slow_tick,
    )

    orch._running = True

    # Run one iteration of the loop then stop
    async def stop_after_one() -> None:
        await asyncio.sleep(0.2)
        orch._running = False

    await asyncio.gather(orch.run(), stop_after_one())

    esp32.emergency_stop.assert_called()


async def test_tick_exception_triggers_emergency_stop() -> None:
    """An exception in tick() triggers emergency_stop."""
    orch, esp32 = _build_mock_orchestrator(
        tick_timeout_s=1.0,
        tick_side_effect=RuntimeError("sensor driver crash"),
    )

    orch._running = True

    async def stop_after_one() -> None:
        await asyncio.sleep(0.1)
        orch._running = False

    await asyncio.gather(orch.run(), stop_after_one())

    esp32.emergency_stop.assert_called()


async def test_successful_tick_does_not_trigger_emergency_stop() -> None:
    """A successful tick does NOT trigger emergency_stop."""

    async def fast_tick() -> None:
        pass  # Instant success

    orch, esp32 = _build_mock_orchestrator(
        tick_timeout_s=1.0,
        tick_side_effect=fast_tick,
    )

    orch._running = True

    async def stop_after_ticks() -> None:
        await asyncio.sleep(0.15)
        orch._running = False

    await asyncio.gather(orch.run(), stop_after_ticks())

    esp32.emergency_stop.assert_not_called()


async def test_tick_timeout_config_from_loop_config() -> None:
    """tick_timeout_s is read from cfg.loop.tick_timeout_s."""
    cfg = Settings(mock_hardware=True)
    # Default should be 1.0
    assert cfg.loop.tick_timeout_s == 1.0


def test_tick_timeout_config_custom_value() -> None:
    """tick_timeout_s accepts custom values."""
    cfg = Settings(mock_hardware=True)
    cfg.loop.tick_timeout_s = 2.5  # type: ignore[misc]
    assert cfg.loop.tick_timeout_s == 2.5


def test_tick_timeout_config_rejects_zero() -> None:
    """tick_timeout_s rejects zero."""
    with pytest.raises(ValueError, match="greater than 0"):
        Settings(mock_hardware=True, loop={"tick_timeout_s": 0})  # type: ignore[arg-type]


def test_tick_timeout_config_rejects_negative() -> None:
    """tick_timeout_s rejects negative values."""
    with pytest.raises(ValueError, match="greater than 0"):
        Settings(mock_hardware=True, loop={"tick_timeout_s": -1})  # type: ignore[arg-type]
