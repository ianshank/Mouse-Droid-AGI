"""Async circuit breaker for fault-tolerant subsystem calls.

Implements the standard CLOSED → OPEN → HALF_OPEN → CLOSED pattern.
All thresholds are read from :class:`~mousedroid.config.schema.CircuitBreakerConfig`.
"""

from __future__ import annotations

import asyncio
import enum
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import CircuitBreakerConfig

_log = get_logger(__name__)

T = TypeVar("T")


class CircuitState(enum.Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, name: str, recovery_remaining_s: float) -> None:
        self.name = name
        self.recovery_remaining_s = recovery_remaining_s
        super().__init__(f"Circuit '{name}' is open; recovery in {recovery_remaining_s:.1f}s")


class CircuitBreaker:
    """Async circuit breaker wrapping any awaitable callable.

    Args:
        name: Human-readable name for logging.
        cfg: Circuit breaker configuration (thresholds from config).
    """

    def __init__(self, name: str, cfg: CircuitBreakerConfig) -> None:
        self._name = name
        self._cfg = cfg
        self._state = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._half_open_calls: int = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    # -- Properties --------------------------------------------------------

    @property
    def name(self) -> str:
        """Circuit breaker name."""
        return self._name

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Consecutive failure count in CLOSED state."""
        return self._failure_count

    # -- Public API --------------------------------------------------------

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute *func* through the circuit breaker.

        Args:
            func: Async callable to protect.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            The result of *func*.

        Raises:
            CircuitOpenError: If the circuit is open and recovery timeout
                has not elapsed.
        """
        async with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.OPEN:
                remaining = self._recovery_remaining_s()
                raise CircuitOpenError(self._name, remaining)

            if (
                self._state == CircuitState.HALF_OPEN
                and self._half_open_calls >= self._cfg.half_open_max_calls
            ):
                remaining = self._recovery_remaining_s()
                raise CircuitOpenError(self._name, remaining)

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        # Execute outside the lock so we don't block other callers.
        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            await self._record_failure(exc)
            raise

        await self._record_success()
        return result

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED state."""
        old = self._state
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        if old != CircuitState.CLOSED:
            _log.info(
                "circuit_breaker_reset",
                name=self._name,
                from_state=old.value,
            )

    # -- Internal ----------------------------------------------------------

    def _maybe_transition_to_half_open(self) -> None:
        """Transition OPEN → HALF_OPEN if recovery timeout has elapsed."""
        if self._state != CircuitState.OPEN:
            return
        if self._recovery_remaining_s() <= 0.0:
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            self._success_count = 0
            _log.info(
                "circuit_breaker_half_open",
                name=self._name,
            )

    def _recovery_remaining_s(self) -> float:
        elapsed = time.monotonic() - self._last_failure_time
        return max(0.0, self._cfg.recovery_timeout_s - elapsed)

    async def _record_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._cfg.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_calls = 0
                    _log.info(
                        "circuit_breaker_closed",
                        name=self._name,
                    )
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _record_failure(self, exc: Exception) -> None:
        async with self._lock:
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                _log.warning(
                    "circuit_breaker_reopened",
                    name=self._name,
                    error=str(exc),
                )
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self._cfg.failure_threshold:
                    self._state = CircuitState.OPEN
                    _log.warning(
                        "circuit_breaker_opened",
                        name=self._name,
                        failure_count=self._failure_count,
                        recovery_timeout_s=self._cfg.recovery_timeout_s,
                        error=str(exc),
                    )
