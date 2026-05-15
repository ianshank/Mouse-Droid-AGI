"""Clock abstraction for deterministic testing of time-dependent components.

Provides :class:`ClockProtocol` — a structural protocol for monotonic time
and async sleep — along with :class:`RealClock` (production) and
:class:`MockClock` (tests).

Usage::

    # Production — identical behaviour to time.monotonic / asyncio.sleep
    clock = RealClock()

    # Tests — advance time deterministically; no wall-clock waiting
    clock = MockClock(start=0.0)
    clock.advance(5.0)   # fires all coroutines that were sleeping ≤ 5 s
"""

from __future__ import annotations

import asyncio
import bisect
import time
from typing import runtime_checkable

from typing_extensions import Protocol


@runtime_checkable
class ClockProtocol(Protocol):
    """Structural protocol for time primitives.

    Any object with a matching ``monotonic()`` and ``sleep()`` signature
    satisfies this protocol — no explicit inheritance needed.
    """

    def monotonic(self) -> float:
        """Return a monotonically increasing timestamp (seconds).

        Semantics match :func:`time.monotonic`.
        """
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the current coroutine for ``seconds``.

        Semantics match :func:`asyncio.sleep`.

        Args:
            seconds: Duration in seconds (non-negative).
        """
        ...


class RealClock:
    """Production clock backed by :mod:`time` and :mod:`asyncio`.

    Drop-in replacement anywhere a :class:`ClockProtocol` is expected.
    """

    def monotonic(self) -> float:
        """Return :func:`time.monotonic`."""
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        """Delegate to :func:`asyncio.sleep`.

        Args:
            seconds: Duration in seconds.
        """
        await asyncio.sleep(seconds)


class MockClock:
    """Deterministic clock for unit and integration tests.

    Time only advances when :meth:`advance` is called explicitly.
    Coroutines awaiting :meth:`sleep` are resolved in order as the
    simulated clock crosses their wakeup deadline.

    Example::

        clock = MockClock(start=0.0)
        task = asyncio.create_task(clock.sleep(3.0))
        clock.advance(5.0)   # task resolves (deadline 3.0 ≤ now 5.0)
    """

    def __init__(self, start: float = 0.0) -> None:
        """Initialise the mock clock.

        Args:
            start: Initial clock value (seconds). Defaults to 0.0.
        """
        self._now: float = start
        self._waiters: list[tuple[float, asyncio.Future[None]]] = []

    def monotonic(self) -> float:
        """Return the current simulated time."""
        return self._now

    async def sleep(self, seconds: float) -> None:
        """Suspend until :meth:`advance` pushes the clock past ``seconds``.

        Args:
            seconds: Duration in seconds (non-negative).
        """
        if seconds <= 0:
            return
        deadline = self._now + seconds
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        # PR #82 follow-up (Gemini): keep ``_waiters`` sorted by deadline
        # via ``bisect.insort`` (O(log N) bisect + O(N) shift) instead of
        # ``list.append + list.sort`` (O(N log N)). Materially faster
        # when many concurrent simulated tasks queue up in tests.
        bisect.insort(self._waiters, (deadline, future), key=lambda t: t[0])
        await future

    def advance(self, delta: float) -> None:
        """Advance simulated time by ``delta`` seconds and resolve expired waiters.

        Args:
            delta: Amount to advance (seconds, must be ≥ 0).
        """
        if delta < 0:
            msg = "MockClock.advance() delta must be non-negative"
            raise ValueError(msg)
        self._now += delta
        # PR #82 follow-up (Gemini): ``_waiters`` is sorted by deadline,
        # so a single ``bisect_right`` finds the split between expired
        # and pending waiters in O(log N) rather than scanning every
        # entry. The two-slice rebuild keeps the post-condition: the
        # remaining list stays sorted and contains only deadlines >
        # ``self._now``.
        idx = bisect.bisect_right(self._waiters, self._now, key=lambda t: t[0])
        expired = self._waiters[:idx]
        self._waiters = self._waiters[idx:]
        for _deadline, future in expired:
            if not future.done():
                future.set_result(None)
