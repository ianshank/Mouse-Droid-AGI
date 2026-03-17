"""Tests for ResilientFlightController — retry, circuit breaker, and safety bypass."""

from __future__ import annotations

from unittest.mock import AsyncMock, PropertyMock

import numpy as np
import pytest

from mousedroid.config.schema import CircuitBreakerConfig, RetryConfig
from mousedroid.resilience.circuit_breaker import CircuitOpenError, CircuitState
from mousedroid.resilience.resilient_flight_controller import ResilientFlightController
from mousedroid.resilience.retry import RetryExhaustedError


def _make_controller(
    *,
    max_attempts: int = 2,
    base_delay_s: float = 0.001,
    failure_threshold: int = 3,
    recovery_timeout_s: float = 30.0,
) -> tuple[ResilientFlightController, AsyncMock]:
    """Build a ResilientFlightController with an AsyncMock inner driver."""
    inner = AsyncMock()
    inner.connect = AsyncMock()
    inner.disconnect = AsyncMock()
    inner.arm = AsyncMock()
    inner.disarm = AsyncMock()
    inner.takeoff = AsyncMock()
    inner.land = AsyncMock()
    inner.send_velocity_ned = AsyncMock()
    inner.get_altitude_m = AsyncMock(return_value=10.0)
    inner.get_gps_position = AsyncMock(return_value=(47.0, 8.0, 500.0))
    inner.get_imu_data = AsyncMock(return_value=np.zeros(6, dtype=np.float32))
    inner.get_battery_voltage = AsyncMock(return_value=16.8)
    inner.set_flight_mode = AsyncMock()
    inner.return_to_launch = AsyncMock()
    inner.emergency_stop = AsyncMock()
    type(inner).armed = PropertyMock(return_value=False)
    type(inner).flight_mode = PropertyMock(return_value="STABILIZE")

    retry_cfg = RetryConfig(
        max_attempts=max_attempts,
        base_delay_s=base_delay_s,
        max_delay_s=1.0,
    )
    cb_cfg = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout_s=recovery_timeout_s,
    )
    controller = ResilientFlightController(inner, retry_cfg, cb_cfg)
    return controller, inner


# -- Delegation (happy path) -----------------------------------------------


async def test_delegates_connect():
    ctrl, inner = _make_controller()
    await ctrl.connect()
    inner.connect.assert_awaited_once()


async def test_delegates_disconnect():
    ctrl, inner = _make_controller()
    await ctrl.disconnect()
    inner.disconnect.assert_awaited_once()


async def test_delegates_arm():
    ctrl, inner = _make_controller()
    await ctrl.arm()
    inner.arm.assert_awaited_once()


async def test_delegates_disarm():
    ctrl, inner = _make_controller()
    await ctrl.disarm()
    inner.disarm.assert_awaited_once()


async def test_delegates_takeoff():
    ctrl, inner = _make_controller()
    await ctrl.takeoff(5.0)
    inner.takeoff.assert_awaited_once_with(5.0)


async def test_delegates_land():
    ctrl, inner = _make_controller()
    await ctrl.land()
    inner.land.assert_awaited_once()


async def test_delegates_send_velocity_ned():
    ctrl, inner = _make_controller()
    await ctrl.send_velocity_ned(1.0, 2.0, -0.5, 0.1)
    inner.send_velocity_ned.assert_awaited_once_with(1.0, 2.0, -0.5, 0.1)


async def test_delegates_get_altitude_m():
    ctrl, _inner = _make_controller()
    result = await ctrl.get_altitude_m()
    assert result == 10.0


async def test_delegates_get_gps_position():
    ctrl, _inner = _make_controller()
    result = await ctrl.get_gps_position()
    assert result == (47.0, 8.0, 500.0)


async def test_delegates_get_imu_data():
    ctrl, _inner = _make_controller()
    result = await ctrl.get_imu_data()
    assert result.shape == (6,)
    assert result.dtype == np.float32


async def test_delegates_get_battery_voltage():
    ctrl, _inner = _make_controller()
    result = await ctrl.get_battery_voltage()
    assert result == 16.8


async def test_delegates_set_flight_mode():
    ctrl, inner = _make_controller()
    await ctrl.set_flight_mode("GUIDED")
    inner.set_flight_mode.assert_awaited_once_with("GUIDED")


async def test_delegates_return_to_launch():
    ctrl, inner = _make_controller()
    await ctrl.return_to_launch()
    inner.return_to_launch.assert_awaited_once()


async def test_delegates_emergency_stop():
    ctrl, inner = _make_controller()
    await ctrl.emergency_stop()
    inner.emergency_stop.assert_awaited_once()


# -- Properties ------------------------------------------------------------


def test_armed_property():
    ctrl, inner = _make_controller()
    type(inner).armed = PropertyMock(return_value=True)
    assert ctrl.armed is True


def test_flight_mode_property():
    ctrl, inner = _make_controller()
    type(inner).flight_mode = PropertyMock(return_value="GUIDED")
    assert ctrl.flight_mode == "GUIDED"


def test_command_circuit_state_property():
    ctrl, _inner = _make_controller()
    assert ctrl.command_circuit_state == CircuitState.CLOSED


def test_query_circuit_state_property():
    ctrl, _inner = _make_controller()
    assert ctrl.query_circuit_state == CircuitState.CLOSED


def test_inner_property():
    ctrl, inner = _make_controller()
    assert ctrl.inner is inner


async def test_stats_tracking():
    ctrl, _inner = _make_controller()
    await ctrl.arm()
    await ctrl.get_altitude_m()

    stats = ctrl.stats
    assert stats["total_calls"] == 2
    assert stats["total_failures"] == 0
    assert stats["command_circuit"] == CircuitState.CLOSED.value
    assert stats["query_circuit"] == CircuitState.CLOSED.value
    assert stats["command_failures"] == 0
    assert stats["query_failures"] == 0


# -- Retry on failure ------------------------------------------------------


async def test_connect_retries_on_failure():
    ctrl, inner = _make_controller(max_attempts=3)
    inner.connect = AsyncMock(
        side_effect=[ConnectionError("x"), ConnectionError("y"), None],
    )
    await ctrl.connect()
    assert inner.connect.await_count == 3


async def test_arm_retries_on_failure():
    ctrl, inner = _make_controller(max_attempts=3)
    inner.arm = AsyncMock(
        side_effect=[ConnectionError("x"), ConnectionError("y"), None],
    )
    await ctrl.arm()
    assert inner.arm.await_count == 3


async def test_disarm_retries_on_failure():
    ctrl, inner = _make_controller(max_attempts=2)
    inner.disarm = AsyncMock(
        side_effect=[ConnectionError("x"), None],
    )
    await ctrl.disarm()
    assert inner.disarm.await_count == 2


async def test_takeoff_retries_on_failure():
    ctrl, inner = _make_controller(max_attempts=2)
    inner.takeoff = AsyncMock(
        side_effect=[ConnectionError("x"), None],
    )
    await ctrl.takeoff(5.0)
    assert inner.takeoff.await_count == 2


async def test_land_retries_on_failure():
    ctrl, inner = _make_controller(max_attempts=2)
    inner.land = AsyncMock(
        side_effect=[ConnectionError("x"), None],
    )
    await ctrl.land()
    assert inner.land.await_count == 2


async def test_send_velocity_ned_retries_on_failure():
    ctrl, inner = _make_controller(max_attempts=3)
    inner.send_velocity_ned = AsyncMock(
        side_effect=[ConnectionError("x"), ConnectionError("y"), None],
    )
    await ctrl.send_velocity_ned(1.0, 0.0, 0.0, 0.0)
    assert inner.send_velocity_ned.await_count == 3


async def test_get_altitude_m_retries_on_failure():
    ctrl, inner = _make_controller(max_attempts=2)
    inner.get_altitude_m = AsyncMock(
        side_effect=[ConnectionError("x"), 15.0],
    )
    result = await ctrl.get_altitude_m()
    assert result == 15.0


async def test_get_gps_position_retries_on_failure():
    ctrl, inner = _make_controller(max_attempts=2)
    inner.get_gps_position = AsyncMock(
        side_effect=[ConnectionError("x"), (48.0, 9.0, 600.0)],
    )
    result = await ctrl.get_gps_position()
    assert result == (48.0, 9.0, 600.0)


async def test_get_imu_data_retries_on_failure():
    ctrl, inner = _make_controller(max_attempts=2)
    expected = np.ones(6, dtype=np.float32)
    inner.get_imu_data = AsyncMock(
        side_effect=[ConnectionError("x"), expected],
    )
    result = await ctrl.get_imu_data()
    np.testing.assert_array_equal(result, expected)


async def test_get_battery_voltage_retries_on_failure():
    ctrl, inner = _make_controller(max_attempts=2)
    inner.get_battery_voltage = AsyncMock(
        side_effect=[ConnectionError("x"), 15.5],
    )
    result = await ctrl.get_battery_voltage()
    assert result == 15.5


async def test_set_flight_mode_retries_on_failure():
    ctrl, inner = _make_controller(max_attempts=2)
    inner.set_flight_mode = AsyncMock(
        side_effect=[ConnectionError("x"), None],
    )
    await ctrl.set_flight_mode("LOITER")
    assert inner.set_flight_mode.await_count == 2


async def test_return_to_launch_retries_on_failure():
    ctrl, inner = _make_controller(max_attempts=2)
    inner.return_to_launch = AsyncMock(
        side_effect=[ConnectionError("x"), None],
    )
    await ctrl.return_to_launch()
    assert inner.return_to_launch.await_count == 2


# -- Circuit breaker opens after threshold ---------------------------------


async def test_command_circuit_opens_after_threshold():
    ctrl, inner = _make_controller(failure_threshold=2, max_attempts=1)
    inner.arm = AsyncMock(side_effect=ConnectionError("fail"))

    for _ in range(2):
        with pytest.raises((RetryExhaustedError, CircuitOpenError)):
            await ctrl.arm()

    assert ctrl.command_circuit_state == CircuitState.OPEN


async def test_query_circuit_opens_after_threshold():
    ctrl, inner = _make_controller(failure_threshold=2, max_attempts=1)
    inner.get_altitude_m = AsyncMock(side_effect=ConnectionError("fail"))

    for _ in range(2):
        with pytest.raises((RetryExhaustedError, CircuitOpenError)):
            await ctrl.get_altitude_m()

    assert ctrl.query_circuit_state == CircuitState.OPEN


# -- CircuitOpenError for command methods ----------------------------------


async def test_circuit_open_rejects_arm():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.arm = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.arm()

    with pytest.raises(CircuitOpenError):
        await ctrl.arm()

    assert ctrl.stats["total_failures"] >= 1


async def test_circuit_open_rejects_disarm():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.disarm = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.disarm()

    with pytest.raises(CircuitOpenError):
        await ctrl.disarm()


async def test_circuit_open_rejects_takeoff():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.takeoff = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.takeoff(5.0)

    with pytest.raises(CircuitOpenError):
        await ctrl.takeoff(5.0)


async def test_circuit_open_rejects_land():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.land = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.land()

    with pytest.raises(CircuitOpenError):
        await ctrl.land()


async def test_circuit_open_rejects_send_velocity_ned():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.send_velocity_ned = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.send_velocity_ned(1.0, 0.0, 0.0, 0.0)

    with pytest.raises(CircuitOpenError):
        await ctrl.send_velocity_ned(1.0, 0.0, 0.0, 0.0)


async def test_circuit_open_rejects_set_flight_mode():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.set_flight_mode = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.set_flight_mode("GUIDED")

    with pytest.raises(CircuitOpenError):
        await ctrl.set_flight_mode("GUIDED")


async def test_circuit_open_rejects_return_to_launch():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.return_to_launch = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.return_to_launch()

    with pytest.raises(CircuitOpenError):
        await ctrl.return_to_launch()


# -- CircuitOpenError for query methods ------------------------------------


async def test_circuit_open_rejects_get_altitude_m():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.get_altitude_m = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.get_altitude_m()

    assert ctrl.query_circuit_state == CircuitState.OPEN

    with pytest.raises(CircuitOpenError):
        await ctrl.get_altitude_m()

    assert ctrl.stats["total_failures"] >= 1


async def test_circuit_open_rejects_get_gps_position():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.get_gps_position = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.get_gps_position()

    with pytest.raises(CircuitOpenError):
        await ctrl.get_gps_position()


async def test_circuit_open_rejects_get_imu_data():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.get_imu_data = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.get_imu_data()

    with pytest.raises(CircuitOpenError):
        await ctrl.get_imu_data()


async def test_circuit_open_rejects_get_battery_voltage():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.get_battery_voltage = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.get_battery_voltage()

    assert ctrl.query_circuit_state == CircuitState.OPEN

    with pytest.raises(CircuitOpenError):
        await ctrl.get_battery_voltage()

    assert ctrl.stats["total_failures"] >= 1


# -- Separate circuits for command and query --------------------------------


async def test_separate_circuits_for_command_and_query():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.arm = AsyncMock(side_effect=ConnectionError("cmd fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.arm()

    assert ctrl.command_circuit_state == CircuitState.OPEN
    # Query circuit should still be closed
    assert ctrl.query_circuit_state == CircuitState.CLOSED

    result = await ctrl.get_altitude_m()
    assert result == 10.0


# -- Generic Exception propagation (failure counting) ----------------------


async def test_arm_exception_increments_failure_count():
    ctrl, inner = _make_controller(failure_threshold=10, max_attempts=1)
    inner.arm = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await ctrl.arm()

    assert ctrl.stats["total_calls"] == 1
    assert ctrl.stats["total_failures"] == 1


async def test_disarm_exception_increments_failure_count():
    ctrl, inner = _make_controller(failure_threshold=10, max_attempts=1)
    inner.disarm = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await ctrl.disarm()

    assert ctrl.stats["total_failures"] == 1


async def test_takeoff_exception_increments_failure_count():
    ctrl, inner = _make_controller(failure_threshold=10, max_attempts=1)
    inner.takeoff = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await ctrl.takeoff(5.0)

    assert ctrl.stats["total_failures"] == 1


async def test_land_exception_increments_failure_count():
    ctrl, inner = _make_controller(failure_threshold=10, max_attempts=1)
    inner.land = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await ctrl.land()

    assert ctrl.stats["total_failures"] == 1


async def test_velocity_exception_increments_failure_count():
    ctrl, inner = _make_controller(failure_threshold=10, max_attempts=1)
    inner.send_velocity_ned = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await ctrl.send_velocity_ned(1.0, 0.0, 0.0, 0.0)

    assert ctrl.stats["total_failures"] == 1


async def test_altitude_exception_increments_failure_count():
    ctrl, inner = _make_controller(failure_threshold=10, max_attempts=1)
    inner.get_altitude_m = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await ctrl.get_altitude_m()

    assert ctrl.stats["total_failures"] == 1


async def test_gps_exception_increments_failure_count():
    ctrl, inner = _make_controller(failure_threshold=10, max_attempts=1)
    inner.get_gps_position = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await ctrl.get_gps_position()

    assert ctrl.stats["total_failures"] == 1


async def test_imu_exception_increments_failure_count():
    ctrl, inner = _make_controller(failure_threshold=10, max_attempts=1)
    inner.get_imu_data = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await ctrl.get_imu_data()

    assert ctrl.stats["total_failures"] == 1


async def test_battery_exception_increments_failure_count():
    ctrl, inner = _make_controller(failure_threshold=10, max_attempts=1)
    inner.get_battery_voltage = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await ctrl.get_battery_voltage()

    assert ctrl.stats["total_failures"] == 1


async def test_set_flight_mode_exception_increments_failure_count():
    ctrl, inner = _make_controller(failure_threshold=10, max_attempts=1)
    inner.set_flight_mode = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await ctrl.set_flight_mode("GUIDED")

    assert ctrl.stats["total_failures"] == 1


async def test_rtl_exception_increments_failure_count():
    ctrl, inner = _make_controller(failure_threshold=10, max_attempts=1)
    inner.return_to_launch = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await ctrl.return_to_launch()

    assert ctrl.stats["total_failures"] == 1


# -- Emergency stop bypass -------------------------------------------------


async def test_emergency_stop_bypasses_circuit_breaker():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.arm = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.arm()
    assert ctrl.command_circuit_state == CircuitState.OPEN

    # Emergency stop should still work despite open circuit
    await ctrl.emergency_stop()
    inner.emergency_stop.assert_awaited_once()


async def test_emergency_stop_propagates_error():
    ctrl, inner = _make_controller()
    inner.emergency_stop = AsyncMock(side_effect=ConnectionError("estop fail"))
    with pytest.raises(ConnectionError, match="estop fail"):
        await ctrl.emergency_stop()


# -- Disconnect error handling ---------------------------------------------


async def test_disconnect_handles_error_gracefully():
    ctrl, inner = _make_controller()
    inner.disconnect = AsyncMock(side_effect=ConnectionError("disconnect fail"))
    # Should not raise — error is caught and logged as warning
    await ctrl.disconnect()


# -- Reset -----------------------------------------------------------------


async def test_reset_clears_circuits():
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.arm = AsyncMock(side_effect=ConnectionError("fail"))
    inner.get_altitude_m = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.arm()
    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.get_altitude_m()

    assert ctrl.command_circuit_state == CircuitState.OPEN
    assert ctrl.query_circuit_state == CircuitState.OPEN

    ctrl.reset()
    assert ctrl.command_circuit_state == CircuitState.CLOSED
    assert ctrl.query_circuit_state == CircuitState.CLOSED


# -- Stats count failures --------------------------------------------------


async def test_stats_count_command_failures():
    ctrl, inner = _make_controller(failure_threshold=10, max_attempts=1, recovery_timeout_s=999.0)
    inner.arm = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await ctrl.arm()

    stats = ctrl.stats
    assert stats["total_calls"] == 1
    assert stats["total_failures"] == 1
    assert stats["command_failures"] == 1
    assert stats["query_failures"] == 0


async def test_stats_count_query_failures():
    ctrl, inner = _make_controller(failure_threshold=10, max_attempts=1, recovery_timeout_s=999.0)
    inner.get_altitude_m = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await ctrl.get_altitude_m()

    stats = ctrl.stats
    assert stats["total_calls"] == 1
    assert stats["total_failures"] == 1
    assert stats["query_failures"] == 1
    assert stats["command_failures"] == 0


async def test_stats_count_circuit_open_failures():
    """CircuitOpenError path also increments total_failures."""
    ctrl, inner = _make_controller(failure_threshold=1, max_attempts=1, recovery_timeout_s=999.0)
    inner.arm = AsyncMock(side_effect=ConnectionError("fail"))

    # First call: RetryExhaustedError opens circuit, increments total_failures
    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await ctrl.arm()

    # Second call: CircuitOpenError, also increments total_failures
    with pytest.raises(CircuitOpenError):
        await ctrl.arm()

    assert ctrl.stats["total_calls"] == 2
    assert ctrl.stats["total_failures"] == 2
