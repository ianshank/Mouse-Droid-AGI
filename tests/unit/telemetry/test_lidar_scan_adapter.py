"""Tests for ``lidar_scan_to_raw`` — driver → telemetry adapter."""

from __future__ import annotations

import math

import numpy as np

from mousedroid.sensing.lidar_scan import LidarScan
from mousedroid.telemetry.protocol import LidarRawScan, lidar_scan_to_raw


def _make_scan() -> LidarScan:
    return LidarScan(
        angles_deg=np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32),
        distances_mm=np.array([1000.0, 2000.0, 3000.0, 500.0], dtype=np.float32),
        confidences=np.array([255, 200, 128, 64], dtype=np.uint8),
        timestamp=12345.0,
        n_points=4,
    )


def test_converts_degrees_to_radians() -> None:
    raw = lidar_scan_to_raw(_make_scan())
    expected = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    for actual, want in zip(raw.angles_rad, expected, strict=True):
        assert actual == math.isclose(actual, want, rel_tol=1e-6) or abs(actual - want) < 1e-6


def test_converts_mm_to_metres() -> None:
    raw = lidar_scan_to_raw(_make_scan())
    assert raw.distances_m == [1.0, 2.0, 3.0, 0.5]


def test_normalises_confidences_to_unit_interval() -> None:
    raw = lidar_scan_to_raw(_make_scan())
    assert raw.intensities[0] == 1.0  # 255/255
    assert raw.intensities[-1] == 64 / 255
    assert all(0.0 <= c <= 1.0 for c in raw.intensities)


def test_preserves_timestamp_and_point_count() -> None:
    raw = lidar_scan_to_raw(_make_scan())
    assert raw.timestamp == 12345.0
    assert raw.n_points == 4
    assert len(raw.angles_rad) == 4
    assert len(raw.distances_m) == 4


def test_returns_lidar_raw_scan_instance() -> None:
    raw = lidar_scan_to_raw(_make_scan())
    assert isinstance(raw, LidarRawScan)


def test_empty_scan_produces_empty_lists() -> None:
    empty = LidarScan(
        angles_deg=np.array([], dtype=np.float32),
        distances_mm=np.array([], dtype=np.float32),
        confidences=np.array([], dtype=np.uint8),
        timestamp=0.0,
        n_points=0,
    )
    raw = lidar_scan_to_raw(empty)
    assert raw.angles_rad == []
    assert raw.distances_m == []
    assert raw.intensities == []
    assert raw.n_points == 0
