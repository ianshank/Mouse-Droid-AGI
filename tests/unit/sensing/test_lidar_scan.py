"""Tests for LidarScan dataclass and empty_scan factory."""

from __future__ import annotations

import time

import numpy as np
import pytest
from numpy.typing import NDArray

from mousedroid.sensing.lidar_scan import LidarScan, empty_scan


def _make_scan(
    n_points: int = 10,
    *,
    angles: NDArray[np.float32] | None = None,
    distances: NDArray[np.float32] | None = None,
    confidences: NDArray[np.uint8] | None = None,
) -> LidarScan:
    """Build a LidarScan with sensible defaults."""
    if angles is None:
        angles = np.linspace(0.0, 359.0, num=n_points, dtype=np.float32)
    if distances is None:
        distances = np.full(n_points, 1000.0, dtype=np.float32)
    if confidences is None:
        confidences = np.full(n_points, 200, dtype=np.uint8)
    return LidarScan(
        angles_deg=angles,
        distances_mm=distances,
        confidences=confidences,
        timestamp=time.monotonic(),
        n_points=n_points,
    )


# -- Construction -----------------------------------------------------------


def test_construction_basic() -> None:
    """LidarScan can be constructed with valid arrays."""
    scan = _make_scan(5)
    assert scan.n_points == 5
    assert scan.angles_deg.shape == (5,)
    assert scan.distances_mm.shape == (5,)
    assert scan.confidences.shape == (5,)


def test_n_points_matches_arrays() -> None:
    """n_points field matches the length of the parallel arrays."""
    for n in (0, 1, 100, 360):
        scan = _make_scan(n)
        assert scan.n_points == n
        assert len(scan.angles_deg) == n
        assert len(scan.distances_mm) == n
        assert len(scan.confidences) == n


def test_timestamp_is_positive() -> None:
    """Timestamp is a positive monotonic value."""
    scan = _make_scan(1)
    assert scan.timestamp > 0.0


def test_dtypes() -> None:
    """Arrays have the expected numpy dtypes."""
    scan = _make_scan(3)
    assert scan.angles_deg.dtype == np.float32
    assert scan.distances_mm.dtype == np.float32
    assert scan.confidences.dtype == np.uint8


# -- Frozen immutability ----------------------------------------------------


def test_frozen_cannot_set_attribute() -> None:
    """Frozen dataclass prevents attribute reassignment."""
    scan = _make_scan(3)
    with pytest.raises(AttributeError):
        scan.n_points = 999  # type: ignore[misc]


def test_frozen_cannot_set_timestamp() -> None:
    """Frozen dataclass prevents timestamp reassignment."""
    scan = _make_scan(3)
    with pytest.raises(AttributeError):
        scan.timestamp = 0.0  # type: ignore[misc]


# -- empty_scan factory -----------------------------------------------------


def test_empty_scan_has_zero_points() -> None:
    """empty_scan returns a scan with zero measurement points."""
    scan = empty_scan()
    assert scan.n_points == 0


def test_empty_scan_arrays_are_empty() -> None:
    """empty_scan arrays have length zero."""
    scan = empty_scan()
    assert len(scan.angles_deg) == 0
    assert len(scan.distances_mm) == 0
    assert len(scan.confidences) == 0


def test_empty_scan_dtypes() -> None:
    """empty_scan arrays have the correct dtypes."""
    scan = empty_scan()
    assert scan.angles_deg.dtype == np.float32
    assert scan.distances_mm.dtype == np.float32
    assert scan.confidences.dtype == np.uint8


def test_empty_scan_timestamp_is_recent() -> None:
    """empty_scan timestamp is a recent monotonic value."""
    before = time.monotonic()
    scan = empty_scan()
    after = time.monotonic()
    assert before <= scan.timestamp <= after
