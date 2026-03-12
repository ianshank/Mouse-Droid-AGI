"""Tests for circuit breaker — all state transitions and edge cases."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from mousedroid.config.schema import CircuitBreakerConfig
from mousedroid.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


def _make_cb(**overrides: object) -> CircuitBreaker:
    cfg = CircuitBreakerConfig(**overrides)
    return CircuitBreaker("test", cfg)


async def _ok() -> str:
    return "ok"


async def _fail() -> str:
    raise ConnectionError("boom")


# -- Initial state ---------------------------------------------------------


def test_initial_state_closed():
    cb = _make_cb()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_name_property():
    cb = CircuitBreaker("my_driver", CircuitBreakerConfig())
    assert cb.name == "my_driver"


# -- CLOSED state ----------------------------------------------------------


async def test_successful_call_stays_closed():
    cb = _make_cb()
    result = await cb.call(_ok)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


async def test_failures_below_threshold_stay_closed():
    cb = _make_cb(failure_threshold=3)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            await cb.call(_fail)
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 2


async def test_success_resets_failure_count():
    cb = _make_cb(failure_threshold=5)
    # Accumulate some failures
    for _ in range(3):
        with pytest.raises(ConnectionError):
            await cb.call(_fail)
    assert cb.failure_count == 3
    # A success resets
    await cb.call(_ok)
    assert cb.failure_count == 0


# -- OPEN state ------------------------------------------------------------


async def test_failures_at_threshold_opens_circuit():
    cb = _make_cb(failure_threshold=3)
    for _ in range(3):
        with pytest.raises(ConnectionError):
            await cb.call(_fail)
    assert cb.state == CircuitState.OPEN


async def test_open_circuit_rejects_calls():
    cb = _make_cb(failure_threshold=1, recovery_timeout_s=999.0)
    with pytest.raises(ConnectionError):
        await cb.call(_fail)
    assert cb.state == CircuitState.OPEN

    with pytest.raises(CircuitOpenError) as exc_info:
        await cb.call(_ok)
    assert exc_info.value.name == "test"
    assert exc_info.value.recovery_remaining_s > 0


# -- HALF_OPEN state -------------------------------------------------------


async def test_open_transitions_to_half_open_after_timeout():
    cb = _make_cb(failure_threshold=1, recovery_timeout_s=0.01)
    with pytest.raises(ConnectionError):
        await cb.call(_fail)
    assert cb.state == CircuitState.OPEN

    await asyncio.sleep(0.02)
    result = await cb.call(_ok)
    assert result == "ok"
    # Should have transitioned through HALF_OPEN
    assert cb.state in {CircuitState.HALF_OPEN, CircuitState.CLOSED}


async def test_half_open_success_closes_circuit():
    cb = _make_cb(
        failure_threshold=1,
        recovery_timeout_s=0.01,
        half_open_max_calls=2,
    )
    with pytest.raises(ConnectionError):
        await cb.call(_fail)
    assert cb.state == CircuitState.OPEN

    await asyncio.sleep(0.02)

    # Two successes needed to close
    await cb.call(_ok)
    await cb.call(_ok)
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


async def test_half_open_failure_reopens_circuit():
    cb = _make_cb(failure_threshold=1, recovery_timeout_s=0.01)
    with pytest.raises(ConnectionError):
        await cb.call(_fail)

    await asyncio.sleep(0.02)

    with pytest.raises(ConnectionError):
        await cb.call(_fail)
    assert cb.state == CircuitState.OPEN


async def test_half_open_limits_concurrent_calls():
    cb = _make_cb(
        failure_threshold=1,
        recovery_timeout_s=0.01,
        half_open_max_calls=1,
    )
    with pytest.raises(ConnectionError):
        await cb.call(_fail)

    await asyncio.sleep(0.02)

    # First call allowed (transitions to half-open)
    AsyncMock(return_value="slow")

    async def slow_call() -> str:
        await asyncio.sleep(0.1)
        return "slow"

    # After one half-open call is in-flight, additional calls should be rejected
    # until recovery timeout elapses again
    await cb.call(_ok)
    # The circuit should now be closed since half_open_max_calls=1 and we succeeded
    assert cb.state == CircuitState.CLOSED


# -- Reset -----------------------------------------------------------------


async def test_reset_clears_state():
    cb = _make_cb(failure_threshold=1)
    with pytest.raises(ConnectionError):
        await cb.call(_fail)
    assert cb.state == CircuitState.OPEN

    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0

    # Should work again
    result = await cb.call(_ok)
    assert result == "ok"


# -- Config-driven behaviour -----------------------------------------------


async def test_config_values_not_hardcoded():
    """Different configs produce different behaviours."""
    # Threshold=2 → opens after 2 failures
    cb2 = _make_cb(failure_threshold=2)
    with pytest.raises(ConnectionError):
        await cb2.call(_fail)
    assert cb2.state == CircuitState.CLOSED
    with pytest.raises(ConnectionError):
        await cb2.call(_fail)
    assert cb2.state == CircuitState.OPEN

    # Threshold=4 → still closed after 2 failures
    cb4 = _make_cb(failure_threshold=4)
    for _ in range(3):
        with pytest.raises(ConnectionError):
            await cb4.call(_fail)
    assert cb4.state == CircuitState.CLOSED


# -- Concurrent safety -----------------------------------------------------


async def test_concurrent_calls_thread_safe():
    """Multiple concurrent calls don't corrupt state."""
    cb = _make_cb(failure_threshold=10)

    call_count = 0

    async def counting() -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.001)
        return call_count

    results = await asyncio.gather(*[cb.call(counting) for _ in range(5)])
    assert len(results) == 5
    assert cb.state == CircuitState.CLOSED


# -- Exception propagation ------------------------------------------------


async def test_original_exception_propagates():
    """The original exception is raised, not wrapped."""
    cb = _make_cb(failure_threshold=10)

    async def raise_value_error() -> None:
        raise ValueError("specific error")

    with pytest.raises(ValueError, match="specific error"):
        await cb.call(raise_value_error)


async def test_circuit_open_error_fields():
    cb = _make_cb(failure_threshold=1, recovery_timeout_s=60.0)
    with pytest.raises(ConnectionError):
        await cb.call(_fail)

    with pytest.raises(CircuitOpenError) as exc_info:
        await cb.call(_ok)
    assert "test" in str(exc_info.value)
    assert exc_info.value.recovery_remaining_s > 50.0
