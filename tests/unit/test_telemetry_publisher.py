"""Tests for TelemetryPublisher — rate limiting, queue overflow, stats."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

from mousedroid.config.schema import TelemetryConfig
from mousedroid.telemetry.protocol import TelemetryFrame
from mousedroid.telemetry.publisher import TelemetryPublisher


def _make_publisher(**kwargs) -> TelemetryPublisher:
    cfg = TelemetryConfig(**kwargs)
    return TelemetryPublisher(cfg)


def _make_frame(**kwargs) -> TelemetryFrame:
    return TelemetryFrame(**kwargs)


def test_publisher_initial_stats():
    pub = _make_publisher()
    assert pub.stats == {"frames_published": 0, "frames_dropped": 0}


def test_publisher_get_queue():
    pub = _make_publisher(queue_size=16)
    q = pub.get_queue()
    assert isinstance(q, asyncio.Queue)
    assert q.maxsize == 16


async def test_publish_single_frame():
    pub = _make_publisher(publish_hz=60.0)
    frame = _make_frame(timestamp=1.0, tick_count=1)
    await pub.publish(frame)
    assert pub.stats["frames_published"] == 1
    assert pub.stats["frames_dropped"] == 0


async def test_publish_frame_appears_in_queue():
    pub = _make_publisher(publish_hz=60.0)
    frame = _make_frame(timestamp=1.0)
    await pub.publish(frame)
    q = pub.get_queue()
    result = q.get_nowait()
    assert result.timestamp == 1.0


async def test_publish_rate_limiting():
    pub = _make_publisher(publish_hz=2.0)  # max 2 per second

    # First publish succeeds
    await pub.publish(_make_frame(tick_count=1))
    assert pub.stats["frames_published"] == 1

    # Immediate second publish is rate-limited (skipped, not dropped)
    await pub.publish(_make_frame(tick_count=2))
    assert pub.stats["frames_published"] == 1  # still 1


async def test_publish_after_rate_interval():
    pub = _make_publisher(publish_hz=60.0)  # very fast

    await pub.publish(_make_frame(tick_count=1))
    assert pub.stats["frames_published"] == 1

    # Simulate time passing
    pub._last_publish = time.monotonic() - 1.0
    await pub.publish(_make_frame(tick_count=2))
    assert pub.stats["frames_published"] == 2


async def test_queue_overflow_drops_frame():
    pub = _make_publisher(publish_hz=60.0, queue_size=2)

    # Fill the queue
    pub._last_publish = 0.0
    await pub.publish(_make_frame(tick_count=1))
    pub._last_publish = 0.0
    await pub.publish(_make_frame(tick_count=2))

    # Third should be dropped
    pub._last_publish = 0.0
    await pub.publish(_make_frame(tick_count=3))
    assert pub.stats["frames_published"] == 2
    assert pub.stats["frames_dropped"] == 1


async def test_queue_overflow_increments_counter():
    pub = _make_publisher(publish_hz=60.0, queue_size=1)

    pub._last_publish = 0.0
    await pub.publish(_make_frame(tick_count=1))

    # Fill and drop multiple
    for i in range(5):
        pub._last_publish = 0.0
        await pub.publish(_make_frame(tick_count=i + 10))

    assert pub.stats["frames_dropped"] == 5


async def test_stats_property():
    pub = _make_publisher(publish_hz=60.0)
    stats = pub.stats
    assert "frames_published" in stats
    assert "frames_dropped" in stats
    assert isinstance(stats["frames_published"], int)
    assert isinstance(stats["frames_dropped"], int)


async def test_publisher_multiple_consumers():
    pub = _make_publisher(publish_hz=60.0)
    q = pub.get_queue()

    pub._last_publish = 0.0
    await pub.publish(_make_frame(tick_count=1))

    # Same queue reference
    assert q is pub.get_queue()
    assert not q.empty()


async def test_publisher_zero_time_does_not_crash():
    """Ensure monotonic time edge cases are handled."""
    pub = _make_publisher(publish_hz=60.0)
    with patch("mousedroid.telemetry.publisher.time") as mock_time:
        mock_time.monotonic.return_value = 100.0
        await pub.publish(_make_frame())
    assert pub.stats["frames_published"] == 1
