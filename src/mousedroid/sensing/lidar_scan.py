"""LiDAR scan data — immutable container for a single 360-degree scan.

Provides a frozen dataclass representing one full rotation of LiDAR
measurement data, plus a factory for empty (fallback) scans.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LidarScan:
    """Immutable container for a single 360-degree LiDAR scan.

    Each scan contains parallel arrays of angle, distance, and confidence
    values for every measured point in one rotation.

    Attributes:
        angles_deg: Angular positions in degrees ``[0, 360)``, shape ``(n_points,)``.
        distances_mm: Measured distances in millimetres, shape ``(n_points,)``.
        confidences: Per-point confidence scores ``[0, 255]``, shape ``(n_points,)``.
        timestamp: Monotonic timestamp when the scan was captured.
        n_points: Number of valid measurement points in this scan.
    """

    angles_deg: NDArray[np.float32]
    distances_mm: NDArray[np.float32]
    confidences: NDArray[np.uint8]
    timestamp: float
    n_points: int


def empty_scan() -> LidarScan:
    """Create an empty LiDAR scan with zero measurement points.

    Used as a fallback when the sensor fails to produce valid data.

    Returns:
        A :class:`LidarScan` with empty arrays and zero points.
    """
    return LidarScan(
        angles_deg=np.array([], dtype=np.float32),
        distances_mm=np.array([], dtype=np.float32),
        confidences=np.array([], dtype=np.uint8),
        timestamp=time.monotonic(),
        n_points=0,
    )
