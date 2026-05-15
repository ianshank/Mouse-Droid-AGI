"""Tests for the raw-LiDAR streaming additions to ``TelemetryPublisher``."""

from __future__ import annotations

import asyncio
import time

from mousedroid.config.schema import TelemetryConfig
from mousedroid.telemetry.protocol import LidarRawScan, TelemetryFrame
from mousedroid.telemetry.publisher import TelemetryPublisher


def _scan(ts: float, n: int = 8) -> LidarRawScan:
    angles = [i * 0.1 for i in range(n)]
    distances = [1.0 + 0.1 * i for i in range(n)]
    return LidarRawScan(
        timestamp=ts,
        angles_rad=angles,
        distances_m=distances,
        n_points=n,
        scan_duration_s=0.1,
    )


class TestLidarRawQueue:
    """The publisher exposes a separate, rate-limited raw LiDAR queue."""

    async def test_publish_lidar_raw_enqueues(self) -> None:
        cfg = TelemetryConfig(enabled=True, lidar_raw_publish_hz=30.0)
        publisher = TelemetryPublisher(cfg)
        # First publish always succeeds because last_publish=0 satisfies
        # ``now - last_publish >= min_interval`` immediately.
        await publisher.publish_lidar_raw(_scan(0.0))
        queue = publisher.get_lidar_raw_queue()
        assert queue.qsize() == 1
        item = queue.get_nowait()
        assert isinstance(item, LidarRawScan)
        assert publisher.stats["lidar_raw_published"] == 1

    async def test_publish_lidar_raw_is_rate_limited(self) -> None:
        # 2 Hz → min interval 500 ms. Rapid-fire publishes drop after the first.
        cfg = TelemetryConfig(enabled=True, lidar_raw_publish_hz=2.0)
        publisher = TelemetryPublisher(cfg)
        await publisher.publish_lidar_raw(_scan(0.0))
        await publisher.publish_lidar_raw(_scan(0.01))
        await publisher.publish_lidar_raw(_scan(0.02))
        assert publisher.stats["lidar_raw_published"] == 1

    async def test_publish_lidar_raw_drops_when_queue_full(self) -> None:
        cfg = TelemetryConfig(
            enabled=True,
            lidar_raw_publish_hz=30.0,
            lidar_raw_queue_size=2,
        )
        publisher = TelemetryPublisher(cfg)
        await publisher.publish_lidar_raw(_scan(0.0))
        # Force the rate-limit clock so the next puts are accepted.
        publisher._lidar_raw_last_publish = 0.0  # type: ignore[attr-defined]
        await publisher.publish_lidar_raw(_scan(1.0))
        publisher._lidar_raw_last_publish = 0.0  # type: ignore[attr-defined]
        await publisher.publish_lidar_raw(_scan(2.0))  # would-be 3rd → dropped
        assert publisher.stats["lidar_raw_published"] == 2
        assert publisher.stats["lidar_raw_dropped"] == 1

    async def test_frame_queue_and_raw_queue_are_independent(self) -> None:
        """Pushing to the frame queue does not touch the raw queue."""
        cfg = TelemetryConfig(enabled=True, publish_hz=60.0, lidar_raw_publish_hz=30.0)
        publisher = TelemetryPublisher(cfg)
        await publisher.publish(TelemetryFrame(timestamp=time.monotonic()))
        assert publisher.get_queue().qsize() == 1
        assert publisher.get_lidar_raw_queue().qsize() == 0


async def test_stats_include_raw_counters() -> None:
    cfg = TelemetryConfig(enabled=True, lidar_raw_publish_hz=30.0)
    publisher = TelemetryPublisher(cfg)
    stats = publisher.stats
    assert "lidar_raw_published" in stats
    assert "lidar_raw_dropped" in stats


async def test_get_lidar_raw_queue_is_asyncio_queue() -> None:
    cfg = TelemetryConfig(enabled=True)
    publisher = TelemetryPublisher(cfg)
    queue = publisher.get_lidar_raw_queue()
    assert isinstance(queue, asyncio.Queue)
