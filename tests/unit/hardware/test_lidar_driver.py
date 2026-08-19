"""Unit tests for LiDARScanner and MockLiDAR drivers."""

from __future__ import annotations

import pytest

from mousedroid.config.schema.hardware import LidarConfig
from mousedroid.hardware.lidar_driver import LiDARScanner, MockLiDAR


@pytest.mark.asyncio
async def test_lidar_scanner_scan_and_buffer() -> None:
    """Verify LiDARScanner scan fetching and ring-buffer storage."""
    cfg = LidarConfig()
    lidar = LiDARScanner(cfg=cfg, port="/dev/ttyUSB1", buffer_size=10)

    assert lidar.is_healthy() is True
    scan = await lidar.get_latest_scan()
    assert len(scan) == 360

    await lidar.close()
    assert lidar.is_healthy() is False
    assert await lidar.get_latest_scan() == []


@pytest.mark.asyncio
async def test_mock_lidar() -> None:
    """Verify MockLiDAR provides deterministic distance readings."""
    lidar = MockLiDAR(default_distance=3.5)
    assert lidar.is_healthy() is True

    scan = await lidar.get_latest_scan()
    assert len(scan) == 360
    assert scan[0] == 3.5
    assert lidar.scan_count == 1

    await lidar.close()
    assert lidar.is_healthy() is False
    assert lidar.closed is True
