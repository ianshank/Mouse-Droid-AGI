"""Tests for JetsonCSICamera — mock-based coverage for hardware-dependent code."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mousedroid.config.schema import CameraConfig


def _cfg(**overrides):
    defaults = {"resolution_width": 320, "resolution_height": 240, "fps": 30, "feature_dim": 64}
    defaults.update(overrides)
    return CameraConfig(**defaults)


def _patch_backends(mod, ju=None, cv2=None):
    """Return combined context manager patching both backends."""
    return (
        patch.object(mod, "_jetson_utils", ju),
        patch.object(mod, "_cv2", cv2),
    )


# ---------------------------------------------------------------------------
# _extract_features (pure numpy, no hardware)
# ---------------------------------------------------------------------------


def test_extract_features_normal():
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    cam = JetsonCSICamera(_cfg())
    frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    features = cam._extract_features(frame)
    assert features.shape == (64,)
    assert features.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(features), 1.0, atol=1e-5)


def test_extract_features_small_frame():
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    cam = JetsonCSICamera(_cfg(feature_dim=1024))
    frame = np.ones((2, 2, 3), dtype=np.uint8)
    features = cam._extract_features(frame)
    assert features.shape == (1024,)
    assert features.dtype == np.float32


def test_extract_features_zero_frame():
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    cam = JetsonCSICamera(_cfg())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    features = cam._extract_features(frame)
    assert features.shape == (64,)
    assert np.allclose(features, 0.0)


# ---------------------------------------------------------------------------
# start() — error when no backend available
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_no_backend_raises():
    from mousedroid.hardware.camera import jetson_csi

    cam = jetson_csi.JetsonCSICamera(_cfg())
    p_ju, p_cv = _patch_backends(jetson_csi)
    with p_ju, p_cv, pytest.raises(RuntimeError, match="Neither jetson_utils nor OpenCV"):
        await cam.start()


# ---------------------------------------------------------------------------
# start() / stop() with mocked jetson_utils backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_stop_jetson_utils_backend():
    from mousedroid.hardware.camera import jetson_csi

    mock_ju = MagicMock()
    mock_source = MagicMock()
    mock_ju.videoSource.return_value = mock_source

    cam = jetson_csi.JetsonCSICamera(_cfg())

    p_ju, p_cv = _patch_backends(jetson_csi, ju=mock_ju)
    with p_ju, p_cv:
        await cam.start()
        assert cam._backend == "jetson_utils"
        assert cam._camera is mock_source

        await cam.stop()
        assert cam._camera is None
        assert cam._backend is None


# ---------------------------------------------------------------------------
# start() / stop() with mocked OpenCV/GStreamer backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_stop_gstreamer_backend():
    from mousedroid.hardware.camera import jetson_csi

    mock_cv2 = MagicMock()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.CAP_GSTREAMER = 1800

    cam = jetson_csi.JetsonCSICamera(_cfg())

    p_ju, p_cv = _patch_backends(jetson_csi, cv2=mock_cv2)
    with p_ju, p_cv:
        await cam.start()
        assert cam._backend == "gstreamer"

        await cam.stop()
        mock_cap.release.assert_called_once()
        assert cam._camera is None


@pytest.mark.asyncio
async def test_start_gstreamer_open_fails():
    from mousedroid.hardware.camera import jetson_csi

    mock_cv2 = MagicMock()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.CAP_GSTREAMER = 1800

    cam = jetson_csi.JetsonCSICamera(_cfg())

    p_ju, p_cv = _patch_backends(jetson_csi, cv2=mock_cv2)
    with p_ju, p_cv, pytest.raises(RuntimeError, match="Failed to open CSI camera"):
        await cam.start()


@pytest.mark.asyncio
async def test_start_jetson_utils_fallback_to_gstreamer():
    """When jetson_utils raises, falls back to gstreamer."""
    from mousedroid.hardware.camera import jetson_csi

    mock_ju = MagicMock()
    mock_ju.videoSource.side_effect = RuntimeError("no camera")

    mock_cv2 = MagicMock()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.CAP_GSTREAMER = 1800

    cam = jetson_csi.JetsonCSICamera(_cfg())

    p_ju, p_cv = _patch_backends(jetson_csi, ju=mock_ju, cv2=mock_cv2)
    with p_ju, p_cv:
        await cam.start()
        assert cam._backend == "gstreamer"


# ---------------------------------------------------------------------------
# capture_features with mocked backends
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_features_jetson_utils():
    from mousedroid.hardware.camera import jetson_csi

    mock_ju = MagicMock()
    mock_source = MagicMock()
    cuda_img = MagicMock()
    mock_source.Capture.return_value = cuda_img
    fake_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    mock_ju.cudaToNumpy.return_value = fake_frame
    mock_ju.videoSource.return_value = mock_source

    cam = jetson_csi.JetsonCSICamera(_cfg())

    p_ju, p_cv = _patch_backends(jetson_csi, ju=mock_ju)
    with p_ju, p_cv:
        await cam.start()
        features = await cam.capture_features()
        assert features.shape == (64,)
        assert features.dtype == np.float32


@pytest.mark.asyncio
async def test_capture_features_gstreamer_success():
    from mousedroid.hardware.camera import jetson_csi

    mock_cv2 = MagicMock()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    fake_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, fake_frame)
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.CAP_GSTREAMER = 1800

    cam = jetson_csi.JetsonCSICamera(_cfg())

    p_ju, p_cv = _patch_backends(jetson_csi, cv2=mock_cv2)
    with p_ju, p_cv:
        await cam.start()
        features = await cam.capture_features()
        assert features.shape == (64,)


@pytest.mark.asyncio
async def test_capture_frame_gstreamer_failure():
    from mousedroid.hardware.camera import jetson_csi

    mock_cv2 = MagicMock()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.CAP_GSTREAMER = 1800

    cam = jetson_csi.JetsonCSICamera(_cfg())

    p_ju, p_cv = _patch_backends(jetson_csi, cv2=mock_cv2)
    with p_ju, p_cv:
        await cam.start()
        features = await cam.capture_features()
        assert features.shape == (64,)
        assert np.allclose(features, 0.0)


# ---------------------------------------------------------------------------
# feature_dim property
# ---------------------------------------------------------------------------


def test_feature_dim_property():
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    cam = JetsonCSICamera(_cfg(feature_dim=128))
    assert cam.feature_dim == 128


# ---------------------------------------------------------------------------
# stop() when no camera started (no-op)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_without_start():
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    cam = JetsonCSICamera(_cfg())
    await cam.stop()  # Should not raise
