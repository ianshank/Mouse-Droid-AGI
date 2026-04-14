"""Tests for retry with exponential backoff — timing, jitter, and edge cases."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from mousedroid.config.schema import RetryConfig
from mousedroid.resilience.retry import (
    RetryExhaustedError,
    _compute_delay,
    retry_async,
    with_retry,
)


def _cfg(**overrides: object) -> RetryConfig:
    return RetryConfig(**overrides)


# -- Success on first try --------------------------------------------------


async def test_succeeds_first_try_no_retry():
    mock = AsyncMock(return_value="ok")
    result = await retry_async(mock, cfg=_cfg())
    assert result == "ok"
    assert mock.await_count == 1


# -- Retry then success ----------------------------------------------------


async def test_retries_on_failure_then_succeeds():
    mock = AsyncMock(side_effect=[ConnectionError("x"), "ok"])
    result = await retry_async(mock, cfg=_cfg(max_attempts=3, base_delay_s=0.001))
    assert result == "ok"
    assert mock.await_count == 2


async def test_retries_multiple_times():
    mock = AsyncMock(
        side_effect=[ConnectionError("1"), ConnectionError("2"), "ok"],
    )
    result = await retry_async(mock, cfg=_cfg(max_attempts=3, base_delay_s=0.001))
    assert result == "ok"
    assert mock.await_count == 3


# -- Exhaustion ------------------------------------------------------------


async def test_exhausts_all_attempts():
    mock = AsyncMock(side_effect=ConnectionError("fail"))
    with pytest.raises(RetryExhaustedError) as exc_info:
        await retry_async(mock, cfg=_cfg(max_attempts=3, base_delay_s=0.001))
    assert exc_info.value.attempts == 3
    assert mock.await_count == 3


async def test_raises_retry_exhausted_with_last_exception():
    errors = [ConnectionError("first"), ConnectionError("last")]
    mock = AsyncMock(side_effect=errors)
    with pytest.raises(RetryExhaustedError) as exc_info:
        await retry_async(mock, cfg=_cfg(max_attempts=2, base_delay_s=0.001))
    assert str(exc_info.value.last_exception) == "last"


# -- Max attempts config ---------------------------------------------------


async def test_respects_max_attempts_config():
    mock = AsyncMock(side_effect=ConnectionError("fail"))

    with pytest.raises(RetryExhaustedError):
        await retry_async(mock, cfg=_cfg(max_attempts=1, base_delay_s=0.001))
    assert mock.await_count == 1

    mock.reset_mock()
    with pytest.raises(RetryExhaustedError):
        await retry_async(mock, cfg=_cfg(max_attempts=5, base_delay_s=0.001))
    assert mock.await_count == 5


# -- Backoff timing --------------------------------------------------------


def test_exponential_backoff_timing():
    cfg = _cfg(base_delay_s=1.0, exponential_base=2.0, max_delay_s=100.0)
    # attempt 0: 1.0 * 2^0 = 1.0 (plus jitter)
    # attempt 1: 1.0 * 2^1 = 2.0 (plus jitter)
    # attempt 2: 1.0 * 2^2 = 4.0 (plus jitter)
    for attempt, expected_base in [(0, 1.0), (1, 2.0), (2, 4.0), (3, 8.0)]:
        delay = _compute_delay(attempt, cfg)
        # With 10% jitter, delay should be in [base, base * 1.1]
        assert delay >= expected_base
        assert delay <= expected_base * 1.1 + 0.001


def test_max_delay_caps_backoff():
    cfg = _cfg(base_delay_s=1.0, exponential_base=2.0, max_delay_s=5.0)
    # attempt 10: 1.0 * 2^10 = 1024, capped to 5.0
    delay = _compute_delay(10, cfg)
    assert delay <= 5.0 * 1.1 + 0.001


def test_jitter_adds_randomness():
    cfg = _cfg(base_delay_s=1.0, exponential_base=2.0, max_delay_s=100.0)
    delays = {_compute_delay(0, cfg) for _ in range(20)}
    # With randomness, we should get multiple distinct values
    assert len(delays) > 1


# -- Exception filtering --------------------------------------------------


async def test_only_retries_specified_exceptions():
    mock = AsyncMock(side_effect=[ConnectionError("retry"), "ok"])
    result = await retry_async(
        mock,
        cfg=_cfg(max_attempts=3, base_delay_s=0.001),
        retryable_exceptions=(ConnectionError,),
    )
    assert result == "ok"


async def test_non_retryable_exception_raises_immediately():
    mock = AsyncMock(side_effect=ValueError("not retryable"))
    with pytest.raises(ValueError, match="not retryable"):
        await retry_async(
            mock,
            cfg=_cfg(max_attempts=3, base_delay_s=0.001),
            retryable_exceptions=(ConnectionError,),
        )
    assert mock.await_count == 1


# -- Decorator form --------------------------------------------------------


async def test_decorator_form():
    call_count = 0

    @with_retry(_cfg(max_attempts=3, base_delay_s=0.001))
    async def flaky() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("flaky")
        return "ok"

    result = await flaky()
    assert result == "ok"
    assert call_count == 2


async def test_decorator_exhaustion():
    @with_retry(_cfg(max_attempts=2, base_delay_s=0.001))
    async def always_fails() -> str:
        raise ConnectionError("always")

    with pytest.raises(RetryExhaustedError):
        await always_fails()


# -- Config-driven ---------------------------------------------------------


async def test_config_values_not_hardcoded():
    """Different configs produce different retry counts."""
    mock1 = AsyncMock(side_effect=ConnectionError("fail"))
    with pytest.raises(RetryExhaustedError):
        await retry_async(mock1, cfg=_cfg(max_attempts=2, base_delay_s=0.001))
    assert mock1.await_count == 2

    mock2 = AsyncMock(side_effect=ConnectionError("fail"))
    with pytest.raises(RetryExhaustedError):
        await retry_async(mock2, cfg=_cfg(max_attempts=4, base_delay_s=0.001))
    assert mock2.await_count == 4


# -- CancelledError passthrough --------------------------------------------


async def test_cancelled_error_propagates_immediately():
    """CancelledError must pass through without retry or wrapping."""
    call_count = 0

    async def cancellable() -> str:
        nonlocal call_count
        call_count += 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await retry_async(
            cancellable,
            cfg=_cfg(max_attempts=5, base_delay_s=0.001),
        )
    # Must not retry; exactly one call
    assert call_count == 1


# -- RetryExhaustedError ---------------------------------------------------


def test_retry_exhausted_error_str():
    exc = RetryExhaustedError(3, ConnectionError("inner"))
    assert "3 attempts" in str(exc)
    assert "inner" in str(exc)


# -- Non-retryable exceptions -----------------------------------------------


async def test_non_retryable_propagates_immediately():
    """FileNotFoundError in non_retryable_exceptions raises on first attempt."""
    mock = AsyncMock(side_effect=FileNotFoundError("/dev/ttyUSB0"))
    with pytest.raises(FileNotFoundError, match="/dev/ttyUSB0"):
        await retry_async(
            mock,
            cfg=_cfg(max_attempts=5, base_delay_s=0.001),
            non_retryable_exceptions=(FileNotFoundError,),
        )
    assert mock.await_count == 1


async def test_non_retryable_overrides_retryable():
    """non_retryable takes precedence even when exception matches retryable."""
    mock = AsyncMock(side_effect=FileNotFoundError("missing"))
    with pytest.raises(FileNotFoundError):
        await retry_async(
            mock,
            cfg=_cfg(max_attempts=5, base_delay_s=0.001),
            retryable_exceptions=(Exception,),
            non_retryable_exceptions=(FileNotFoundError,),
        )
    assert mock.await_count == 1


async def test_non_retryable_empty_allows_retry():
    """Default empty non_retryable_exceptions preserves existing behaviour."""
    mock = AsyncMock(side_effect=[FileNotFoundError("x"), "ok"])
    result = await retry_async(
        mock,
        cfg=_cfg(max_attempts=3, base_delay_s=0.001),
        retryable_exceptions=(Exception,),
        non_retryable_exceptions=(),
    )
    assert result == "ok"
    assert mock.await_count == 2


async def test_decorator_non_retryable():
    """with_retry decorator forwards non_retryable_exceptions."""
    call_count = 0

    @with_retry(
        _cfg(max_attempts=5, base_delay_s=0.001),
        non_retryable_exceptions=(FileNotFoundError,),
    )
    async def check_port() -> str:
        nonlocal call_count
        call_count += 1
        raise FileNotFoundError("/dev/ttyUSB0")

    with pytest.raises(FileNotFoundError):
        await check_port()
    assert call_count == 1
