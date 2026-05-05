"""Unit tests for the shared :class:`TokenBucket`."""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.common.rate_limit import TokenBucket


async def _take_ok(bucket: TokenBucket) -> bool:
    """Tiny helper: extract the bool from the ``(ok, hint)`` tuple."""
    ok, _ = await bucket.take()
    return ok


@pytest.mark.asyncio
async def test_initial_burst_capacity_default() -> None:
    """Default capacity equals max(1.0, rate_per_s)."""
    bucket = TokenBucket(rate_per_s=5.0)
    # Burst capacity = 5; consume 5 in a row.
    for _ in range(5):
        assert await _take_ok(bucket) is True
    assert await _take_ok(bucket) is False


@pytest.mark.asyncio
async def test_explicit_capacity_overrides_default() -> None:
    """Explicit capacity wins over the rate-derived default."""
    bucket = TokenBucket(rate_per_s=10.0, capacity=2.0)
    assert await _take_ok(bucket) is True
    assert await _take_ok(bucket) is True
    assert await _take_ok(bucket) is False


@pytest.mark.asyncio
async def test_refills_over_time() -> None:
    """After draining, waiting refill_per_s seconds yields one new token."""
    bucket = TokenBucket(rate_per_s=20.0, capacity=1.0)
    assert await _take_ok(bucket) is True
    assert await _take_ok(bucket) is False
    # 1/20 s + a little slack so the loop's monotonic clock advances.
    await asyncio.sleep(0.06)
    assert await _take_ok(bucket) is True


@pytest.mark.asyncio
async def test_low_rate_floors_capacity_at_one() -> None:
    """rate_per_s < 1 still yields a usable bucket of 1 token."""
    bucket = TokenBucket(rate_per_s=0.1)
    assert await _take_ok(bucket) is True
    assert await _take_ok(bucket) is False


@pytest.mark.asyncio
async def test_take_returns_atomic_retry_after_when_empty() -> None:
    """``take`` returns ``(False, hint)`` consistently under the lock.

    REGRESSION: Copilot review — the previous ``retry_after_s()`` was a
    sync accessor that could observe a torn ``_tokens`` value. The
    atomic tuple return computes the hint while still holding the lock
    that maintained the bucket invariant.
    """
    bucket = TokenBucket(rate_per_s=5.0, capacity=1.0)
    ok, hint = await bucket.take()
    assert ok is True
    assert hint == 0.0
    ok, hint = await bucket.take()
    assert ok is False
    assert 0.0 < hint < 1.0  # 1 token / 5 rps == 0.2 s in the worst case


@pytest.mark.asyncio
async def test_async_retry_after_s_acquires_lock() -> None:
    """The standalone observability accessor is async + lock-protected."""
    bucket = TokenBucket(rate_per_s=5.0, capacity=1.0)
    await _take_ok(bucket)
    hint = await bucket.retry_after_s()
    assert hint > 0.0
    assert hint < 1.0


@pytest.mark.asyncio
async def test_concurrent_takes_never_overshoot_capacity() -> None:
    """``asyncio.gather`` of N takes can hand out at most ``capacity`` tokens.

    REGRESSION: Copilot review — the rate limiter must remain correct
    when multiple coroutines race on the same bucket.
    """
    bucket = TokenBucket(rate_per_s=2.0, capacity=3.0)
    results = await asyncio.gather(*(bucket.take() for _ in range(20)))
    accepted = sum(1 for ok, _ in results if ok)
    assert accepted == 3


@settings(max_examples=20, deadline=None)
@given(
    rate=st.floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_property_take_count_bounded_by_capacity(rate: float) -> None:
    """Without sleeping, take() succeeds at most ``capacity`` times in a row."""

    async def _drive() -> int:
        bucket = TokenBucket(rate_per_s=rate)
        consumed = 0
        for _ in range(int(max(1.0, rate)) + 5):
            ok, _ = await bucket.take()
            if ok:
                consumed += 1
        return consumed

    consumed = asyncio.run(_drive())
    assert consumed <= int(max(1.0, rate)) + 1  # +1 for float rounding.
