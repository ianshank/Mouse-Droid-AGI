"""Telemetry publisher — async queue bridge between orchestrator and server.

The orchestrator calls ``publish()`` at the end of each tick. The publisher
rate-limits and enqueues frames for the telemetry server to consume. If the
queue is full, the frame is silently dropped to avoid backpressure on the
30 Hz control loop.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger
from mousedroid.telemetry.protocol import TelemetryFrame

if TYPE_CHECKING:
    from mousedroid.config.schema import TelemetryConfig

_log = get_logger(__name__)


class TelemetryPublisher:
    """Publish telemetry frames to an internal async queue.

    Non-blocking: if the queue is full, the new frame is dropped.
    The server consumer reads from the same queue.

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

    def get_queue(self) -> asyncio.Queue[TelemetryFrame]:
        """Return the internal queue for the server to consume.

        Returns:
            The bounded ``asyncio.Queue``.
        """
        return self._queue

    @property
    def stats(self) -> dict[str, int]:
        """Publishing statistics.

        Returns:
            Dict with ``frames_published`` and ``frames_dropped`` counts.
        """
        return {
            "frames_published": self._frames_published,
            "frames_dropped": self._frames_dropped,
        }
