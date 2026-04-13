"""Tests for MockLidar driver."""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.config.schema import LidarConfig
from mousedroid.hardware.lidar.mock_lidar import MockLidar
from mousedroid.sensing.lidar_scan import LidarScan


@pytest.fixture
def lidar_cfg() -> LidarConfig:
    """Default LidarConfig for testing."""
    return LidarConfig(
        enabled=True,
        serial_port="/dev/ttyUSB1",
        baud_rate=230400,
        max_range_m=12.0,
        min_range_m=0.15,
        scan_frequency_hz=10.0,
        min_confidence=0,
        read_timeout_s=0.2,
        n_sectors=36,
        feature_dim=36,
    )


@pytest.fixture
def mock_lidar(lidar_cfg: LidarConfig) -> MockLidar:
    """MockLidar instance."""
    return MockLidar(lidar_cfg)


def test_construct(lidar_cfg: LidarConfig) -> None:
    """MockLidar can be constructed."""
    lidar = MockLidar(lidar_cfg)
    assert lidar is not None


def test_max_range_m(mock_lidar: MockLidar) -> None:
    """max_range_m matches config."""
    assert mock_lidar.max_range_m == 12.0


def test_min_range_m(mock_lidar: MockLidar) -> None:
    """min_range_m matches config."""
    assert mock_lidar.min_range_m == 0.15


def test_scan_frequency_hz(mock_lidar: MockLidar) -> None:
    """scan_frequency_hz matches config."""
    assert mock_lidar.scan_frequency_hz == 10.0


async def test_default_scan_has_360_points(mock_lidar: MockLidar) -> None:
    """Default scan returns 360 uniformly-spaced points."""
    scan = await mock_lidar.read_scan()
    assert scan.n_points == 360
    assert len(scan.angles_deg) == 360
    assert len(scan.distances_mm) == 360


async def test_default_scan_midrange(mock_lidar: MockLidar) -> None:
    """Default scan distances are at midrange."""
    scan = await mock_lidar.read_scan()
    expected_mm = (12.0 + 0.15) / 2.0 * 1000.0
    np.testing.assert_allclose(scan.distances_mm, expected_mm, rtol=1e-5)


async def test_set_scan_changes_value(mock_lidar: MockLidar) -> None:
    """set_scan overrides the returned scan data."""
    custom = LidarScan(
        angles_deg=np.array([0.0, 90.0, 180.0], dtype=np.float32),
        distances_mm=np.array([100.0, 200.0, 300.0], dtype=np.float32),
        confidences=np.array([255, 255, 255], dtype=np.uint8),
        timestamp=1.0,
        n_points=3,
    )
    mock_lidar.set_scan(custom)
    scan = await mock_lidar.read_scan()
    assert scan.n_points == 3
    np.testing.assert_array_equal(scan.distances_mm, custom.distances_mm)


def test_not_started_initially(mock_lidar: MockLidar) -> None:
    """MockLidar is not started after construction."""
    assert mock_lidar.started is False


async def test_default_scan_high_confidence(mock_lidar: MockLidar) -> None:
    """Default scan points have confidence 200."""
    scan = await mock_lidar.read_scan()
    np.testing.assert_array_equal(scan.confidences, 200)


async def test_start_stop_lifecycle(mock_lidar: MockLidar) -> None:
    """start/stop toggle the started property."""
    assert not mock_lidar.started
    await mock_lidar.start()
    assert mock_lidar.started
    await mock_lidar.stop()
    assert not mock_lidar.started


async def test_start_stop_idempotent(mock_lidar: MockLidar) -> None:
    """Multiple start/stop cycles work without error."""
    await mock_lidar.start()
    await mock_lidar.start()
    assert mock_lidar.started is True
    await mock_lidar.stop()
    await mock_lidar.stop()
    assert mock_lidar.started is False
