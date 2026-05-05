"""Async token-bucket rate limiter shared by MCP and REST control planes.

Extracted from :mod:`mousedroid.mcp.tool_bridge` so the same algorithm can
gate the OpenClaw-driven REST mission endpoint without duplicating the
implementation. The bucket itself has no MCP-specific dependencies and is
fully driven by its constructor arguments — refill rate per second and
optional burst capacity (defaults to ``max(1.0, rate_per_s)`` for a
one-second burst, matching the historical MCP behaviour).
"""

from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Per-session token bucket for cheap async rate limiting.

    Thread-safe within a single event loop via an :class:`asyncio.Lock`.
    Callers ``await`` :meth:`take` and respond with ``rate_limited`` when
    it returns ``False``.
    """

    __slots__ = ("_capacity", "_last", "_lock", "_refill_per_s", "_tokens")

    def __init__(self, rate_per_s: float, *, capacity: float | None = None) -> None:
        """Initialise the bucket.

        Args:
            rate_per_s: Sustained refill rate in tokens / second. Must be
                positive; the caller is responsible for validating this
                against its own config field constraints.
            capacity: Burst capacity. Defaults to ``max(1.0, rate_per_s)``,
                which keeps memory bounded and matches the
                MCP-config-driven envelope.
        """
        self._capacity = capacity if capacity is not None else max(1.0, rate_per_s)
        self._refill_per_s = rate_per_s
        self._tokens: float = self._capacity
        self._last: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def take(self) -> bool:
        """Consume one token if available.

        Returns:
            ``True`` when a token was consumed, ``False`` when the bucket
            was empty (caller should respond with a rate-limit error).
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_s)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def retry_after_s(self) -> float:
        """Approximate seconds until at least one token is available.

        Used for the ``Retry-After`` hint in HTTP 429 responses. Does not
        consume a token.
        """
        deficit = max(0.0, 1.0 - self._tokens)
        if self._refill_per_s <= 0:  # pragma: no cover - guarded by config
            return float("inf")
        return deficit / self._refill_per_s


__all__ = ["TokenBucket"]
