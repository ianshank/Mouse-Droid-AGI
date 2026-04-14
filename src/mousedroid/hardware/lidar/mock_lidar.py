"""Mock FHL-LD19 LiDAR driver for testing.

Implements :class:`~mousedroid.hardware.protocols.LidarProtocol` with
configurable scan data, following the ``MockUltrasonic`` / ``MockMicrophone``
pattern.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from mousedroid.constants import (
    LIDAR_DEFAULT_MOCK_CONFIDENCE,
    LIDAR_FULL_ROTATION_DEG,
    LIDAR_MM_TO_M,
)
from mousedroid.logging.setup import get_logger
from mousedroid.sensing.lidar_scan import LidarScan

if TYPE_CHECKING:
    from mousedroid.config.schema import LidarConfig

_log = get_logger(__name__)


class MockLidar:
    """Mock FHL-LD19 implementing LidarProtocol.

    Returns configurable scan data for deterministic testing.  Default
    behaviour produces 360 uniformly-spaced points at the midpoint of
    the configured range.

    Args:
        cfg: LiDAR configuration.
    """

    def __init__(self, cfg: LidarConfig) -> None:
        self._cfg = cfg
        self._started = False
        self._custom_scan: LidarScan | None = None

    # -- LidarProtocol properties -----------------------------------------

    @property
    def max_range_m(self) -> float:
        """Maximum detection range in metres."""
        return self._cfg.max_range_m

    @property
    def min_range_m(self) -> float:
        """Minimum detection range in metres."""
        return self._cfg.min_range_m

    @property
    def scan_frequency_hz(self) -> float:
        """Nominal scan rotation frequency in Hz."""
        return self._cfg.scan_frequency_hz

    # -- Lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Start mock LiDAR (no-op beyond state tracking)."""
        self._started = True
        _log.info("mock_lidar_started")

    async def stop(self) -> None:
        """Stop mock LiDAR (no-op beyond state tracking)."""
        self._started = False
        _log.info("mock_lidar_stopped")

    # -- Data acquisition -------------------------------------------------

    async def read_scan(self) -> LidarScan:
        """Return the current mock scan data.

        Returns:
            A custom scan if set via :meth:`set_scan`, otherwise a
            default scan of 360 uniformly-spaced points at midrange.
        """
        if self._custom_scan is not None:
            return self._custom_scan
        return self._default_scan()

    # -- Test control -----------------------------------------------------

    def set_scan(self, scan: LidarScan) -> None:
        """Set a custom scan returned by :meth:`read_scan`.

        Args:
            scan: Custom :class:`LidarScan` to return.
        """
        self._custom_scan = scan

    @property
    def started(self) -> bool:
        """Whether :meth:`start` has been called without a subsequent :meth:`stop`."""
        return self._started

    # -- Private ----------------------------------------------------------

    def _default_scan(self) -> LidarScan:
        """Generate a default scan with 360 uniformly-spaced points."""
        n_points = int(LIDAR_FULL_ROTATION_DEG)
        mid_range_mm = (self._cfg.max_range_m + self._cfg.min_range_m) / 2.0 * LIDAR_MM_TO_M
        angles = np.linspace(0.0, LIDAR_FULL_ROTATION_DEG - 1.0, num=n_points, dtype=np.float32)
        distances = np.full(n_points, mid_range_mm, dtype=np.float32)
        confidences = np.full(n_points, LIDAR_DEFAULT_MOCK_CONFIDENCE, dtype=np.uint8)
        return LidarScan(
            angles_deg=angles,
            distances_mm=distances,
            confidences=confidences,
            timestamp=time.monotonic(),
            n_points=n_points,
        )
