"""Async retry with exponential backoff and jitter.

All timing parameters are read from
:class:`~mousedroid.config.schema.RetryConfig`.  Nothing is hardcoded.
"""

from __future__ import annotations

import asyncio
import functools
import random
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import RetryConfig

_log = get_logger(__name__)

T = TypeVar("T")


class RetryExhaustedError(Exception):
    """All retry attempts have been exhausted.

    Attributes:
        attempts: Total number of attempts made.
        last_exception: The exception from the final attempt.
    """

    def __init__(self, attempts: int, last_exception: BaseException) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Retry exhausted after {attempts} attempts: {last_exception}")


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    cfg: RetryConfig,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
    non_retryable_exceptions: tuple[type[BaseException], ...] = (),
    **kwargs: Any,
) -> T:
    """Execute *func* with retry and exponential backoff.

    Args:
        func: Async callable to retry.
        *args: Positional arguments forwarded to *func*.
        cfg: Retry configuration (timing from config).
        retryable_exceptions: Exception types that trigger a retry.
            Non-matching exceptions propagate immediately.
        non_retryable_exceptions: Exception types that always propagate
            immediately, even if they match *retryable_exceptions*.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        The result of *func* on the first successful attempt.

    Raises:
        RetryExhaustedError: If all attempts fail with retryable exceptions.
    """
    last_exc: BaseException | None = None

    for attempt in range(cfg.max_attempts):
        try:
            return await func(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except retryable_exceptions as exc:
            if isinstance(exc, non_retryable_exceptions):
                raise
            last_exc = exc
            remaining = cfg.max_attempts - attempt - 1

            if remaining <= 0:
                break

            delay = _compute_delay(attempt, cfg)

            _log.warning(
                "retry_attempt",
                attempt=attempt + 1,
                max_attempts=cfg.max_attempts,
                remaining=remaining,
                delay_s=round(delay, 3),
                error=str(exc),
            )

            await asyncio.sleep(delay)

    assert last_exc is not None
    raise RetryExhaustedError(cfg.max_attempts, last_exc)


def _compute_delay(attempt: int, cfg: RetryConfig) -> float:
    """Compute backoff delay with jitter for the given attempt number.

    Args:
        attempt: Zero-based attempt index.
        cfg: Retry configuration.

    Returns:
        Delay in seconds, capped at ``cfg.max_delay_s``.
    """
    delay = min(
        cfg.base_delay_s * (cfg.exponential_base**attempt),
        cfg.max_delay_s,
    )
    jitter_frac = cfg.jitter_fraction
    jitter = random.uniform(0.0, delay * jitter_frac)  # noqa: S311
    return delay + jitter


def with_retry(
    cfg: RetryConfig,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
    non_retryable_exceptions: tuple[type[BaseException], ...] = (),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator that wraps an async function with retry logic.

    Args:
        cfg: Retry configuration.
        retryable_exceptions: Exception types that trigger a retry.
        non_retryable_exceptions: Exception types that always propagate
            immediately, even if they match *retryable_exceptions*.

    Returns:
        Decorator that adds retry behaviour to the wrapped function.
    """

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_async(
                func,
                *args,
                cfg=cfg,
                retryable_exceptions=retryable_exceptions,
                non_retryable_exceptions=non_retryable_exceptions,
                **kwargs,
            )

        return wrapper

    return decorator
