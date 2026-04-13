"""Tests for SensorManager."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np

from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.schema import Settings
from mousedroid.sensing.manager import SensorManager


def _make_manager() -> tuple[SensorManager, AsyncMock, AsyncMock, AsyncMock]:
    cfg = Settings(mock_hardware=True)

    vision = AsyncMock()
    vision.capture_features = AsyncMock(
        return_value=np.ones(cfg.camera.feature_dim, dtype=np.float32)
    )
    vision.start = AsyncMock()
    vision.stop = AsyncMock()

    distance = AsyncMock()
    distance.read_distance_m = AsyncMock(return_value=1.5)
    distance.max_range_m = 4.0

    esp32 = AsyncMock()
    esp32.read_encoders = AsyncMock(return_value=EncoderReading())
    esp32.get_battery_voltage = AsyncMock(return_value=12.0)

    mgr = SensorManager(vision, distance, esp32, cfg)
    return mgr, vision, distance, esp32


def test_constructor():
    mgr, _, _, _ = _make_manager()
    assert mgr._vision is not None


async def test_read_all_returns_bundle():
    mgr, _, _, _ = _make_manager()
    bundle = await mgr.read_all()
    assert bundle.timestamp > 0
    assert bundle.distance_m == 1.5
    np.testing.assert_array_equal(bundle.valid_mask, [1.0, 1.0, 1.0, 0.0])


async def test_read_all_handles_vision_failure():
    mgr, vision, _, _ = _make_manager()
    vision.capture_features.side_effect = RuntimeError("camera fail")
    bundle = await mgr.read_all()
    assert bundle.valid_mask[0] == 0.0


async def test_read_all_handles_distance_failure():
    mgr, _, distance, _ = _make_manager()
    distance.read_distance_m.side_effect = RuntimeError("sensor fail")
    bundle = await mgr.read_all()
    assert bundle.valid_mask[1] == 0.0


async def test_read_all_handles_motor_failure():
    mgr, _, _, esp32 = _make_manager()
    esp32.read_encoders.side_effect = RuntimeError("serial fail")
    bundle = await mgr.read_all()
    assert bundle.valid_mask[2] == 0.0


async def test_read_all_microphone_none_backwards_compat():
    """SensorManager with microphone=None still returns a 4-element valid_mask."""
    mgr, _, _, _ = _make_manager()
    # _make_manager does not pass a microphone, so it defaults to None
    bundle = await mgr.read_all()
    assert bundle.valid_mask.shape == (4,)
    # Audio slot should be invalid when no microphone is configured
    assert bundle.valid_mask[3] == 0.0


def _make_manager_with_mic():
    cfg = Settings(mock_hardware=True)

    vision = AsyncMock()
    vision.capture_features = AsyncMock(
        return_value=np.ones(cfg.camera.feature_dim, dtype=np.float32),
    )
    vision.start = AsyncMock()
    vision.stop = AsyncMock()

    distance = AsyncMock()
    distance.read_distance_m = AsyncMock(return_value=1.5)
    distance.max_range_m = 4.0

    esp32 = AsyncMock()
    esp32.read_encoders = AsyncMock(return_value=EncoderReading())
    esp32.get_battery_voltage = AsyncMock(return_value=12.0)

    mic = AsyncMock()
    mic.chunk_size = 1024
    mic.channels = 1
    mic.read_chunk = AsyncMock(
        return_value=np.ones(1024, dtype=np.float32),
    )

    mgr = SensorManager(vision, distance, esp32, cfg, microphone=mic)
    return mgr, mic


async def test_read_all_with_microphone():
    mgr, mic = _make_manager_with_mic()
    bundle = await mgr.read_all()
    assert bundle.valid_mask[3] == 1.0
    mic.read_chunk.assert_awaited_once()


async def test_read_all_microphone_failure():
    mgr, mic = _make_manager_with_mic()
    mic.read_chunk.side_effect = RuntimeError("mic fail")
    bundle = await mgr.read_all()
    assert bundle.valid_mask[3] == 0.0


async def test_start_with_microphone():
    """Test start() calls microphone.start() when microphone is configured."""
    mgr, mic = _make_manager_with_mic()
    mic.start = AsyncMock()
    await mgr.start()
    mic.start.assert_awaited_once()


async def test_stop_with_microphone():
    """Test stop() calls microphone.stop() when microphone is configured."""
    mgr, mic = _make_manager_with_mic()
    mic.stop = AsyncMock()
    await mgr.stop()
    mic.stop.assert_awaited_once()


# ---------------------------------------------------------------------------
# LiDAR integration tests
# ---------------------------------------------------------------------------


def _make_manager_with_lidar():
    """Create a SensorManager with a MockLidar attached."""
    from mousedroid.config.schema import LidarConfig
    from mousedroid.hardware.lidar.mock_lidar import MockLidar

    lidar_cfg = LidarConfig()
    cfg = Settings(mock_hardware=True, lidar=lidar_cfg)

    vision = AsyncMock()
    vision.capture_features = AsyncMock(
        return_value=np.ones(cfg.camera.feature_dim, dtype=np.float32),
    )
    vision.start = AsyncMock()
    vision.stop = AsyncMock()

    distance = AsyncMock()
    distance.read_distance_m = AsyncMock(return_value=1.5)
    distance.max_range_m = 4.0

    esp32 = AsyncMock()
    esp32.read_encoders = AsyncMock(return_value=EncoderReading())
    esp32.get_battery_voltage = AsyncMock(return_value=12.0)

    lidar = MockLidar(lidar_cfg)

    mgr = SensorManager(vision, distance, esp32, cfg, lidar=lidar)
    return mgr, lidar


async def test_read_all_with_lidar_produces_5_element_mask():
    """SensorManager with LiDAR produces a 5-element valid_mask."""
    mgr, _ = _make_manager_with_lidar()
    bundle = await mgr.read_all()
    assert bundle.valid_mask.shape == (5,)
    assert bundle.lidar_features is not None


async def test_read_all_without_lidar_backwards_compat():
    """SensorManager without LiDAR produces a 4-element mask and None lidar_features."""
    mgr, _, _, _ = _make_manager()
    bundle = await mgr.read_all()
    assert bundle.valid_mask.shape == (4,)
    assert bundle.lidar_features is None


async def test_start_stop_with_lidar():
    """Verify lidar.start() and lidar.stop() are called by SensorManager."""
    mgr, lidar = _make_manager_with_lidar()
    await mgr.start()
    assert lidar.started is True
    await mgr.stop()
    assert lidar.started is False
