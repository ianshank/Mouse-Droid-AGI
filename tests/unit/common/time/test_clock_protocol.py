"""Tests for ClockProtocol, RealClock, and MockClock."""

from __future__ import annotations

import asyncio
import time

import pytest

from mousedroid.common.time.protocol import ClockProtocol, MockClock, RealClock


class TestClockProtocolConformance:
    """Both concrete clocks satisfy the structural protocol."""

    def test_real_clock_satisfies_protocol(self) -> None:
        assert isinstance(RealClock(), ClockProtocol)

    def test_mock_clock_satisfies_protocol(self) -> None:
        assert isinstance(MockClock(), ClockProtocol)


class TestRealClock:
    """RealClock delegates to time.monotonic and asyncio.sleep."""

    def test_monotonic_is_non_decreasing(self) -> None:
        clock = RealClock()
        t1 = clock.monotonic()
        t2 = clock.monotonic()
        assert t2 >= t1

    def test_monotonic_tracks_wall_time(self) -> None:
        clock = RealClock()
        before = time.monotonic()
        result = clock.monotonic()
        after = time.monotonic()
        assert before <= result <= after

    async def test_sleep_returns(self) -> None:
        clock = RealClock()
        await clock.sleep(0.0)

    async def test_sleep_zero_returns_immediately(self) -> None:
        clock = RealClock()
        start = time.monotonic()
        await clock.sleep(0.0)
        assert time.monotonic() - start < 0.5


class TestMockClock:
    """MockClock advances deterministically via advance()."""

    def test_initial_time(self) -> None:
        clock = MockClock(start=10.0)
        assert clock.monotonic() == 10.0

    def test_advance_increases_time(self) -> None:
        clock = MockClock()
        clock.advance(3.0)
        assert clock.monotonic() == 3.0

    def test_advance_accumulates(self) -> None:
        clock = MockClock()
        clock.advance(1.0)
        clock.advance(2.0)
        assert clock.monotonic() == 3.0

    def test_advance_negative_raises(self) -> None:
        clock = MockClock()
        with pytest.raises(ValueError, match="non-negative"):
            clock.advance(-1.0)

    def test_advance_zero_is_allowed(self) -> None:
        clock = MockClock(start=5.0)
        clock.advance(0.0)
        assert clock.monotonic() == 5.0

    async def test_sleep_resolves_when_advance_crosses_deadline(self) -> None:
        clock = MockClock()
        resolved: list[float] = []

        async def waiter() -> None:
            await clock.sleep(3.0)
            resolved.append(clock.monotonic())

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0)  # let task reach its await
        assert not task.done()

        clock.advance(2.0)
        await asyncio.sleep(0)
        assert not task.done()

        clock.advance(1.5)  # total 3.5 >= deadline 3.0
        await asyncio.sleep(0)
        assert task.done()
        assert resolved == [3.5]

    async def test_sleep_zero_or_negative_returns_immediately(self) -> None:
        clock = MockClock()
        await clock.sleep(0.0)
        await clock.sleep(-1.0)

    async def test_multiple_waiters_resolved_in_order(self) -> None:
        clock = MockClock()
        order: list[int] = []

        async def waiter(label: int, delay: float) -> None:
            await clock.sleep(delay)
            order.append(label)

        tasks = [
            asyncio.create_task(waiter(3, 3.0)),
            asyncio.create_task(waiter(1, 1.0)),
            asyncio.create_task(waiter(2, 2.0)),
        ]
        await asyncio.sleep(0)

        clock.advance(3.5)
        await asyncio.sleep(0)

        for t in tasks:
            assert t.done()
        assert order == [1, 2, 3]

    async def test_partial_advance_leaves_unresolved_waiters(self) -> None:
        clock = MockClock()
        short_done: list[bool] = []
        long_done: list[bool] = []

        async def short_waiter() -> None:
            await clock.sleep(1.0)
            short_done.append(True)

        async def long_waiter() -> None:
            await clock.sleep(5.0)
            long_done.append(True)

        t_short = asyncio.create_task(short_waiter())
        t_long = asyncio.create_task(long_waiter())
        await asyncio.sleep(0)

        clock.advance(2.0)
        await asyncio.sleep(0)

        assert t_short.done()
        assert not t_long.done()

        clock.advance(3.0)
        await asyncio.sleep(0)
        assert t_long.done()
