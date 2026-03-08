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
    np.testing.assert_array_equal(bundle.valid_mask, [1.0, 1.0, 1.0])


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
