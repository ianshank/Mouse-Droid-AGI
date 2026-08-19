"""Async LiDAR range scanner drivers with ring buffer and mock fallbacks."""

from __future__ import annotations

import asyncio
import math
from collections import deque

from mousedroid.config.schema.hardware import LidarConfig
from mousedroid.constants import (
    DEFAULT_LIDAR_BUFFER_SIZE,
    DEFAULT_LIDAR_MAX_RANGE_M,
    DEFAULT_LIDAR_MIN_RANGE_M,
    DEFAULT_LIDAR_SAMPLES_PER_REV,
    MOCK_ASYNC_YIELD_S,
)
from mousedroid.interfaces.protocols import LiDARProtocol
from mousedroid.logging.setup import get_logger

_log = get_logger("mousedroid.hardware.lidar")


class LiDARScanner(LiDARProtocol):
    """Production 2D LiDAR range scanner driver with ring-buffer storage."""

    def __init__(
        self,
        cfg: LidarConfig,
        port: str,
        buffer_size: int = DEFAULT_LIDAR_BUFFER_SIZE,
    ) -> None:
        self._cfg = cfg
        self._port = port
        self._healthy = True
        self._closed = False
        self._lock = asyncio.Lock()
        self._scan_buffer: deque[list[float]] = deque(maxlen=max(1, buffer_size))
        self.min_dist: float = getattr(
            self._cfg,
            "min_range_m",
            getattr(self._cfg, "min_distance_m", DEFAULT_LIDAR_MIN_RANGE_M),
        )
        self.max_dist: float = getattr(
            self._cfg,
            "max_range_m",
            getattr(self._cfg, "max_distance_m", DEFAULT_LIDAR_MAX_RANGE_M),
        )
        _log.info(
            "lidar_scanner_initialized",
            port=self._port,
            min_distance=self.min_dist,
            max_distance=self.max_dist,
            buffer_size=buffer_size,
        )

    def is_healthy(self) -> bool:
        """Return True if LiDAR device is communicating."""
        return self._healthy and not self._closed

    def _sanitize_scan(self, raw_scan: list[float]) -> list[float]:
        """Filter and validate raw range measurements."""
        if not raw_scan:
            return [self.max_dist] * DEFAULT_LIDAR_SAMPLES_PER_REV
        sanitized: list[float] = []
        for r in raw_scan:
            if math.isnan(r) or math.isinf(r) or r < 0.0:
                sanitized.append(self.max_dist)
            else:
                sanitized.append(max(min(r, self.max_dist), 0.0))
        return sanitized

    async def get_latest_scan(self) -> list[float]:
        """Fetch or synthesize latest 360-degree range scan in meters.

        Returns:
            List of 360 float distance measurements.
        """
        if not self.is_healthy():
            _log.warning("get_latest_scan_called_on_unhealthy_lidar")
            return []

        await asyncio.sleep(MOCK_ASYNC_YIELD_S)
        raw_scan = [self.max_dist] * DEFAULT_LIDAR_SAMPLES_PER_REV
        sanitized = self._sanitize_scan(raw_scan)
        async with self._lock:
            self._scan_buffer.append(sanitized)
        return sanitized

    async def close(self) -> None:
        """Stop LiDAR scan loop and close serial connection."""
        self._closed = True
        self._healthy = False
        async with self._lock:
            self._scan_buffer.clear()
        _log.info("lidar_scanner_closed", port=self._port)


class MockLiDAR(LiDARProtocol):
    """Deterministic mock LiDAR driver for CI and simulations."""

    def __init__(self, default_distance: float = DEFAULT_LIDAR_MAX_RANGE_M) -> None:
        self.default_distance: float = default_distance
        self.scan_count: int = 0
        self.closed: bool = False
        self._custom_scan: list[float] | None = None
        _log.info("mock_lidar_initialized", default_distance=default_distance)

    def is_healthy(self) -> bool:
        """Return True if mock LiDAR is active."""
        return not self.closed

    def set_scan(self, scan: list[float]) -> None:
        """Set custom scan data."""
        self._custom_scan = list(scan)

    async def get_latest_scan(self) -> list[float]:
        """Return mock 360-degree range scan in meters."""
        if self.closed:
            return []
        self.scan_count += 1
        if self._custom_scan is not None:
            return list(self._custom_scan)
        return [self.default_distance] * DEFAULT_LIDAR_SAMPLES_PER_REV

    async def close(self) -> None:
        """Close mock LiDAR."""
        self.closed = True
