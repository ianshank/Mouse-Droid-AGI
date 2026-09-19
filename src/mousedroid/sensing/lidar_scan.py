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
        sensor_responding: Whether the sensor produced any raw frames for
            this scan. See the note below -- this is the field that keeps
            "the LiDAR is dead" apart from "the LiDAR sees a clear room".
    """

    angles_deg: NDArray[np.float32]
    distances_mm: NDArray[np.float32]
    confidences: NDArray[np.uint8]
    timestamp: float
    n_points: int

    #: ``n_points == 0`` has two completely different meanings and the
    #: consumers cannot tell them apart from the arrays alone (peer review
    #: D-24):
    #:
    #: * **The sensor is not talking** -- the serial port was never opened,
    #:   or the motor stalled and no valid frame arrived. Nothing is known
    #:   about the surroundings.
    #: * **The sensor is talking and the room is clear** -- frames arrived,
    #:   but every point fell outside ``[min_range_m, max_range_m]``. On the
    #:   LD19 a no-return beam reports ``0`` mm, which is below
    #:   ``min_range_m``, so an open space genuinely yields zero points.
    #:   "Nothing within ``max_range_m``" is a true and useful reading.
    #:
    #: :func:`~mousedroid.hardware.lidar.feature_extractor.LidarFeatureExtractor.extract`
    #: maps a zero-point scan to ``np.ones(n_sectors)`` -- normalised range
    #: fractions, so *maximum range in every sector*. That is correct for
    #: the second case and a fabricated all-clear for the first, which is
    #: the fail-open S-2 closed one layer higher in
    #: ``SensorManager._safe_lidar_read`` and which survived down here.
    #:
    #: Defaults ``True`` so every existing construction site keeps its
    #: current meaning (CLAUDE.md invariant 6); only the driver's
    #: "no serial" / "no frames" paths set it ``False``.
    #:
    #: Deliberately a bool and not a tunable threshold: a ``min_scan_points``
    #: knob would only be a way to configure the fail-closed path back open,
    #: the same reasoning as ``EmergencyLatchConfig``'s absent fail-open knob.
    sensor_responding: bool = True


def empty_scan(*, sensor_responding: bool = True) -> LidarScan:
    """Create an empty LiDAR scan with zero measurement points.

    Used as a fallback when the sensor fails to produce valid data, and as
    the honest representation of a rotation in which every measured point
    fell outside the configured range band.

    Args:
        sensor_responding: ``False`` when the sensor produced no raw frames
            at all, so the emptiness means "nothing is known" rather than
            "nothing is within range". Keyword-only and defaulting ``True``
            so existing callers keep their current meaning; see
            :class:`LidarScan` for why the distinction exists.

    Returns:
        A :class:`LidarScan` with empty arrays and zero points.
    """
    return LidarScan(
        angles_deg=np.array([], dtype=np.float32),
        distances_mm=np.array([], dtype=np.float32),
        confidences=np.array([], dtype=np.uint8),
        timestamp=time.monotonic(),
        n_points=0,
        sensor_responding=sensor_responding,
    )
