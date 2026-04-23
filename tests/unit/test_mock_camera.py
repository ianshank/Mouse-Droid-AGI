from __future__ import annotations

import numpy as np

from mousedroid.config.schema import CameraConfig
from mousedroid.hardware.camera.mock_camera import MockCamera


def _make_camera(feature_dim: int = 256) -> MockCamera:
    return MockCamera(CameraConfig(feature_dim=feature_dim))


def test_construct():
    cam = _make_camera()
    assert cam is not None


async def test_capture_features_shape():
    cam = _make_camera(128)
    features = await cam.capture_features()
    assert features.shape == (128,)


async def test_capture_features_dtype():
    cam = _make_camera()
    features = await cam.capture_features()
    assert features.dtype == np.float32


def test_feature_dim_property():
    cam = _make_camera(64)
    assert cam.feature_dim == 64


async def test_start_stop():
    cam = _make_camera()
    await cam.start()
    await cam.stop()
    # No exception means success


async def test_capture_features_not_all_zero():
    cam = _make_camera(256)
    features = await cam.capture_features()
    # Random features should not be all zero (extremely unlikely)
    assert not np.allclose(features, 0.0)


async def test_screen_capture_mode_uses_latest_rgb_features():
    cam = MockCamera(CameraConfig(feature_dim=12, mock_source="screen_capture"))
    latest_rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    latest_rgb[:12, :12, 0] = 255
    cam._latest_rgb = latest_rgb

    features = await cam.capture_features()

    assert cam._mode == "screen_capture"
    assert features.shape == (12,)
    assert np.all(np.isfinite(features))
    assert not np.allclose(features, 0.0)
