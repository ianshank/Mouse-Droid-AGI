"""Async token-bucket rate limiter shared by MCP and REST control planes.

Extracted from :mod:`mousedroid.mcp.tool_bridge` so the same algorithm can
gate the OpenClaw-driven REST mission endpoint without duplicating the
implementation. The bucket itself has no MCP-specific dependencies and is
fully driven by its constructor arguments — refill rate per second and
optional burst capacity (defaults to ``max(1.0, rate_per_s)`` for a
one-second burst, matching the historical MCP behaviour).

PR #76 follow-up: the bucket now accepts an optional ``ClockProtocol``
so consumers can drive simulated time in tests without wall-clock waits.
The default :class:`RealClock` keeps production behaviour byte-identical
to the prior ``time.monotonic()`` implementation.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from mousedroid.common.time.protocol import RealClock

if TYPE_CHECKING:
    from mousedroid.common.time.protocol import ClockProtocol


class TokenBucket:
    """Per-session token bucket for cheap async rate limiting.

    Thread-safe within a single event loop via an :class:`asyncio.Lock`.
    Callers ``await`` :meth:`take` and respond with ``rate_limited`` when
    it returns ``False``.

    The bucket reads time via an injected :class:`ClockProtocol` so unit
    tests can advance simulated time deterministically without wall-clock
    waits. Production code injects a :class:`RealClock` (default).
    """

    __slots__ = ("_capacity", "_clock", "_last", "_lock", "_refill_per_s", "_tokens")

    def __init__(
        self,
        rate_per_s: float,
        *,
        capacity: float | None = None,
        clock: ClockProtocol | None = None,
    ) -> None:
        """Initialise the bucket.

        Args:
            rate_per_s: Sustained refill rate in tokens / second. Must be
                positive; the caller is responsible for validating this
                against its own config field constraints.
            capacity: Burst capacity. Defaults to ``max(1.0, rate_per_s)``,
                which keeps memory bounded and matches the
                MCP-config-driven envelope.
            clock: Optional :class:`ClockProtocol` for time. Defaults to
                :class:`RealClock` (production); tests inject
                :class:`MockClock` for deterministic refills.
        """
        self._capacity = capacity if capacity is not None else max(1.0, rate_per_s)
        self._refill_per_s = rate_per_s
        self._tokens: float = self._capacity
        self._clock: ClockProtocol = clock if clock is not None else RealClock()
        self._last: float = self._clock.monotonic()
        self._lock = asyncio.Lock()

    async def take(self) -> tuple[bool, float]:
        """Consume one token if available.

        Returns:
            ``(True, 0.0)`` when a token was consumed; ``(False, hint)``
            when the bucket was empty, where ``hint`` is the approximate
            seconds until the next refill makes a token available.

            Returning the hint atomically with the take outcome avoids
            torn reads against ``_tokens`` from a separate accessor —
            the value is computed under the same lock as the refill.
        """
        async with self._lock:
            now = self._clock.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_s)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True, 0.0
            return False, self._retry_after_s_locked()

    async def retry_after_s(self) -> float:
        """Async snapshot of the seconds-until-refill hint.

        Acquires the same lock as :meth:`take` so concurrent callers
        can't observe a torn ``_tokens`` value. For callers in the hot
        path, prefer the ``(taken, hint)`` tuple from :meth:`take`
        directly — this accessor exists for observability code that
        wants the hint without consuming a token.
        """
        async with self._lock:
            return self._retry_after_s_locked()

    def _retry_after_s_locked(self) -> float:
        """Compute the hint while holding ``_lock`` (caller's responsibility)."""
        deficit = max(0.0, 1.0 - self._tokens)
        if self._refill_per_s <= 0:  # pragma: no cover - guarded by config
            return float("inf")
        return deficit / self._refill_per_s


__all__ = ["TokenBucket"]
