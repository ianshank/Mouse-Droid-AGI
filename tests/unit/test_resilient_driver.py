"""Tests for ResilientESP32Driver — retry, circuit breaker, and safety bypass."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.schema import CircuitBreakerConfig, RetryConfig
from mousedroid.resilience.circuit_breaker import CircuitOpenError, CircuitState
from mousedroid.resilience.resilient_driver import ResilientESP32Driver
from mousedroid.resilience.retry import RetryExhaustedError


def _make_driver(
    *,
    max_attempts: int = 2,
    base_delay_s: float = 0.001,
    failure_threshold: int = 3,
    recovery_timeout_s: float = 30.0,
) -> tuple[ResilientESP32Driver, AsyncMock]:
    inner = AsyncMock()
    inner.connect = AsyncMock()
    inner.disconnect = AsyncMock()
    inner.send_velocity = AsyncMock()
    inner.read_encoders = AsyncMock(return_value=EncoderReading())
    inner.get_battery_voltage = AsyncMock(return_value=12.0)
    inner.emergency_stop = AsyncMock()

    retry_cfg = RetryConfig(
        max_attempts=max_attempts,
        base_delay_s=base_delay_s,
        max_delay_s=1.0,
    )
    cb_cfg = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout_s=recovery_timeout_s,
    )
    driver = ResilientESP32Driver(inner, retry_cfg, cb_cfg)
    return driver, inner


# -- Delegation -----------------------------------------------------------


async def test_delegates_connect():
    driver, inner = _make_driver()
    await driver.connect()
    inner.connect.assert_awaited_once()


async def test_delegates_disconnect():
    driver, inner = _make_driver()
    await driver.disconnect()
    inner.disconnect.assert_awaited_once()


async def test_delegates_send_velocity():
    driver, inner = _make_driver()
    await driver.send_velocity(0.5, 0.0, 0.1)
    inner.send_velocity.assert_awaited_once_with(0.5, 0.0, 0.1)


async def test_delegates_read_encoders():
    driver, inner = _make_driver()
    result = await driver.read_encoders()
    assert isinstance(result, EncoderReading)
    inner.read_encoders.assert_awaited_once()


async def test_delegates_get_battery_voltage():
    driver, inner = _make_driver()
    result = await driver.get_battery_voltage()
    assert result == 12.0


async def test_delegates_emergency_stop():
    driver, inner = _make_driver()
    await driver.emergency_stop()
    inner.emergency_stop.assert_awaited_once()


# -- Retry on failure ------------------------------------------------------


async def test_retries_send_velocity_on_failure():
    driver, inner = _make_driver(max_attempts=3)
    inner.send_velocity = AsyncMock(
        side_effect=[ConnectionError("x"), ConnectionError("y"), None],
    )
    await driver.send_velocity(0.1, 0.0, 0.0)
    assert inner.send_velocity.await_count == 3


async def test_retries_read_encoders_on_failure():
    driver, inner = _make_driver(max_attempts=2)
    inner.read_encoders = AsyncMock(
        side_effect=[ConnectionError("x"), EncoderReading()],
    )
    result = await driver.read_encoders()
    assert isinstance(result, EncoderReading)
    assert inner.read_encoders.await_count == 2


async def test_retries_battery_on_failure():
    driver, inner = _make_driver(max_attempts=2)
    inner.get_battery_voltage = AsyncMock(
        side_effect=[ConnectionError("x"), 11.5],
    )
    result = await driver.get_battery_voltage()
    assert result == 11.5


async def test_connect_retries_on_failure():
    driver, inner = _make_driver(max_attempts=3)
    inner.connect = AsyncMock(
        side_effect=[ConnectionError("x"), ConnectionError("y"), None],
    )
    await driver.connect()
    assert inner.connect.await_count == 3


# -- Circuit breaker -------------------------------------------------------


async def test_circuit_opens_after_threshold():
    driver, inner = _make_driver(failure_threshold=2, max_attempts=1)
    inner.send_velocity = AsyncMock(side_effect=ConnectionError("fail"))

    for _ in range(2):
        with pytest.raises((RetryExhaustedError, CircuitOpenError)):
            await driver.send_velocity(0.1, 0.0, 0.0)

    assert driver.command_circuit_state == CircuitState.OPEN


async def test_circuit_open_rejects_velocity():
    driver, inner = _make_driver(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.send_velocity = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await driver.send_velocity(0.1, 0.0, 0.0)

    with pytest.raises(CircuitOpenError):
        await driver.send_velocity(0.1, 0.0, 0.0)


async def test_separate_circuits_for_command_and_query():
    driver, inner = _make_driver(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.send_velocity = AsyncMock(side_effect=ConnectionError("cmd fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await driver.send_velocity(0.1, 0.0, 0.0)

    assert driver.command_circuit_state == CircuitState.OPEN
    # Query circuit should still be closed
    assert driver.query_circuit_state == CircuitState.CLOSED

    result = await driver.read_encoders()
    assert isinstance(result, EncoderReading)


# -- Emergency stop bypass -------------------------------------------------


async def test_emergency_stop_bypasses_circuit_breaker():
    driver, inner = _make_driver(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.send_velocity = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await driver.send_velocity(0.1, 0.0, 0.0)
    assert driver.command_circuit_state == CircuitState.OPEN

    # Emergency stop should still work
    await driver.emergency_stop()
    inner.emergency_stop.assert_awaited_once()


async def test_emergency_stop_propagates_error():
    driver, inner = _make_driver()
    inner.emergency_stop = AsyncMock(side_effect=ConnectionError("estop fail"))
    with pytest.raises(ConnectionError, match="estop fail"):
        await driver.emergency_stop()


# -- Disconnect error handling ---------------------------------------------


async def test_disconnect_handles_error_gracefully():
    driver, inner = _make_driver()
    inner.disconnect = AsyncMock(side_effect=ConnectionError("disconnect fail"))
    # Should not raise
    await driver.disconnect()


# -- Stats -----------------------------------------------------------------


async def test_stats_tracking():
    driver, inner = _make_driver()
    await driver.send_velocity(0.1, 0.0, 0.0)
    await driver.read_encoders()

    stats = driver.stats
    assert stats["total_calls"] == 2
    assert stats["total_failures"] == 0
    assert stats["command_circuit"] == "closed"
    assert stats["query_circuit"] == "closed"


async def test_stats_count_failures():
    driver, inner = _make_driver(failure_threshold=10, max_attempts=1, recovery_timeout_s=999.0)
    inner.send_velocity = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await driver.send_velocity(0.1, 0.0, 0.0)

    stats = driver.stats
    assert stats["total_calls"] == 1
    assert stats["command_failures"] == 1


# -- Reset -----------------------------------------------------------------


async def test_reset_clears_circuits():
    driver, inner = _make_driver(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.send_velocity = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await driver.send_velocity(0.1, 0.0, 0.0)
    assert driver.command_circuit_state == CircuitState.OPEN

    driver.reset()
    assert driver.command_circuit_state == CircuitState.CLOSED
    assert driver.query_circuit_state == CircuitState.CLOSED


# -- Inner property --------------------------------------------------------


def test_inner_property():
    driver, inner = _make_driver()
    assert driver.inner is inner
