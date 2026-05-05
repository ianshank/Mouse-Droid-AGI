"""Unit tests for the shared :class:`TokenBucket`."""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.common.rate_limit import TokenBucket


@pytest.mark.asyncio
async def test_initial_burst_capacity_default() -> None:
    """Default capacity equals max(1.0, rate_per_s)."""
    bucket = TokenBucket(rate_per_s=5.0)
    # Burst capacity = 5; consume 5 in a row.
    for _ in range(5):
        assert await bucket.take() is True
    assert await bucket.take() is False


@pytest.mark.asyncio
async def test_explicit_capacity_overrides_default() -> None:
    """Explicit capacity wins over the rate-derived default."""
    bucket = TokenBucket(rate_per_s=10.0, capacity=2.0)
    assert await bucket.take() is True
    assert await bucket.take() is True
    assert await bucket.take() is False


@pytest.mark.asyncio
async def test_refills_over_time() -> None:
    """After draining, waiting refill_per_s seconds yields one new token."""
    bucket = TokenBucket(rate_per_s=20.0, capacity=1.0)
    assert await bucket.take() is True
    assert await bucket.take() is False
    # 1/20 s + a little slack so the loop's monotonic clock advances.
    await asyncio.sleep(0.06)
    assert await bucket.take() is True


@pytest.mark.asyncio
async def test_low_rate_floors_capacity_at_one() -> None:
    """rate_per_s < 1 still yields a usable bucket of 1 token."""
    bucket = TokenBucket(rate_per_s=0.1)
    assert await bucket.take() is True
    assert await bucket.take() is False


@pytest.mark.asyncio
async def test_retry_after_s_finite_after_drain() -> None:
    """retry_after_s returns a finite hint once the bucket is empty."""
    bucket = TokenBucket(rate_per_s=5.0, capacity=1.0)
    assert await bucket.take() is True
    hint = bucket.retry_after_s()
    assert hint > 0.0
    assert hint < 1.0


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
            if await bucket.take():
                consumed += 1
        return consumed

    consumed = asyncio.run(_drive())
    assert consumed <= int(max(1.0, rate)) + 1  # +1 for float rounding.
