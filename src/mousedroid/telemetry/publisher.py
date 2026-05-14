"""Telemetry publisher — async queue bridge between orchestrator and server.

The orchestrator calls ``publish()`` at the end of each tick. The publisher
rate-limits and enqueues frames for the telemetry server to consume. If the
queue is full, the frame is silently dropped to avoid backpressure on the
30 Hz control loop.

PR #4 added a second, independent queue for raw LiDAR scans
(``publish_lidar_raw`` / ``get_lidar_raw_queue``). Raw scans flow at a
different rate than ``TelemetryFrame`` (LD19 native is ~10 Hz) and feed a
separate ``/ws/v1/lidar/raw`` WebSocket so the 360° polar dashboard can
animate at the driver's native cadence without throttling the main
control-frame channel.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger
from mousedroid.telemetry.protocol import LidarRawScan, TelemetryFrame

if TYPE_CHECKING:
    from mousedroid.config.schema import TelemetryConfig

_log = get_logger(__name__)


class TelemetryPublisher:
    """Publish telemetry frames + raw LiDAR scans to internal async queues.

    Non-blocking: if any queue is full, the new payload is dropped.
    The server consumer reads from each queue.

    Implements ``TelemetryPublisherProtocol``.
    """

    def __init__(self, cfg: TelemetryConfig) -> None:
        """Initialise publisher from telemetry config.

        Args:
            cfg: Telemetry configuration.
        """
        self._queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=cfg.queue_size)
        self._min_interval: float = 1.0 / cfg.publish_hz
        self._last_publish: float = 0.0
        self._frames_published: int = 0
        self._frames_dropped: int = 0

        # Raw LiDAR streaming — separate rate-limit and queue.
        self._lidar_raw_queue: asyncio.Queue[LidarRawScan] = asyncio.Queue(
            maxsize=cfg.lidar_raw_queue_size,
        )
        self._lidar_raw_min_interval: float = 1.0 / cfg.lidar_raw_publish_hz
        self._lidar_raw_last_publish: float = 0.0
        self._lidar_raw_published: int = 0
        self._lidar_raw_dropped: int = 0

    async def publish(self, frame: TelemetryFrame) -> None:
        """Publish a telemetry frame, rate-limited.

        Skips the frame if called faster than ``publish_hz``. Drops the
        frame (instead of blocking) if the queue is full.

        Args:
            frame: Telemetry snapshot to publish.
        """
        now = time.monotonic()
        if now - self._last_publish < self._min_interval:
            return

        try:
            self._queue.put_nowait(frame)
            self._frames_published += 1
            self._last_publish = now
        except asyncio.QueueFull:
            self._frames_dropped += 1
            _log.debug(
                "telemetry_frame_dropped",
                queue_size=self._queue.maxsize,
                total_dropped=self._frames_dropped,
            )

    async def publish_lidar_raw(self, scan: LidarRawScan) -> None:
        """Publish a raw LiDAR scan, rate-limited.

        Skips the scan if called faster than ``lidar_raw_publish_hz``.
        Drops the scan (instead of blocking) if the raw queue is full.

        Args:
            scan: Raw scan snapshot.
        """
        now = time.monotonic()
        if now - self._lidar_raw_last_publish < self._lidar_raw_min_interval:
            return

        try:
            self._lidar_raw_queue.put_nowait(scan)
            self._lidar_raw_published += 1
            self._lidar_raw_last_publish = now
        except asyncio.QueueFull:
            self._lidar_raw_dropped += 1
            _log.debug(
                "telemetry_lidar_raw_dropped",
                queue_size=self._lidar_raw_queue.maxsize,
                total_dropped=self._lidar_raw_dropped,
            )

    def get_queue(self) -> asyncio.Queue[TelemetryFrame]:
        """Return the internal queue for the server to consume.

        Returns:
            The bounded ``asyncio.Queue`` of ``TelemetryFrame`` objects.
        """
        return self._queue

    def get_lidar_raw_queue(self) -> asyncio.Queue[LidarRawScan]:
        """Return the raw LiDAR streaming queue for the server.

        Returns:
            The bounded ``asyncio.Queue`` of ``LidarRawScan`` objects.
        """
        return self._lidar_raw_queue

    @property
    def stats(self) -> dict[str, int]:
        """Publishing statistics.

        Returns:
            Dict with frame + raw-LiDAR published/dropped counters.
        """
        return {
            "frames_published": self._frames_published,
            "frames_dropped": self._frames_dropped,
            "lidar_raw_published": self._lidar_raw_published,
            "lidar_raw_dropped": self._lidar_raw_dropped,
        }
