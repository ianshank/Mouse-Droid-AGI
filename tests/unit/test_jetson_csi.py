"""Tests for JetsonCSICamera — mock-based coverage for hardware-dependent code."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mousedroid.config.schema import CameraConfig


def _cfg(**overrides):
    defaults = {
        "resolution_width": 320,
        "resolution_height": 240,
        "fps": 30,
        "feature_dim": 64,
        "device_path": "/dev/video0",
    }
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
    mock_gstreamer_cap = MagicMock()
    mock_gstreamer_cap.isOpened.return_value = False
    mock_v4l2_cap = MagicMock()
    mock_v4l2_cap.isOpened.return_value = False
    mock_cv2.VideoCapture.side_effect = [mock_gstreamer_cap, mock_v4l2_cap]
    mock_cv2.CAP_GSTREAMER = 1800

    cam = jetson_csi.JetsonCSICamera(_cfg())

    p_ju, p_cv = _patch_backends(jetson_csi, cv2=mock_cv2)
    with p_ju, p_cv, pytest.raises(RuntimeError, match="Failed to open CSI camera"):
        await cam.start()


@pytest.mark.asyncio
async def test_start_gstreamer_fallback_to_v4l2():
    from mousedroid.hardware.camera import jetson_csi

    mock_cv2 = MagicMock()
    mock_gstreamer_cap = MagicMock()
    mock_gstreamer_cap.isOpened.return_value = False
    mock_v4l2_cap = MagicMock()
    mock_v4l2_cap.isOpened.return_value = True
    mock_cv2.VideoCapture.side_effect = [mock_gstreamer_cap, mock_v4l2_cap]
    mock_cv2.CAP_GSTREAMER = 1800
    mock_cv2.CAP_PROP_FRAME_WIDTH = 3
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
    mock_cv2.CAP_PROP_FPS = 5

    cam = jetson_csi.JetsonCSICamera(_cfg(device_path="/dev/video2"))

    p_ju, p_cv = _patch_backends(jetson_csi, cv2=mock_cv2)
    with p_ju, p_cv:
        await cam.start()
        assert cam._backend == "v4l2"
        mock_cv2.VideoCapture.assert_any_call("/dev/video2")
        mock_v4l2_cap.set.assert_any_call(mock_cv2.CAP_PROP_FRAME_WIDTH, 320)
        mock_v4l2_cap.set.assert_any_call(mock_cv2.CAP_PROP_FRAME_HEIGHT, 240)
        mock_v4l2_cap.set.assert_any_call(mock_cv2.CAP_PROP_FPS, 30)


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


@pytest.mark.asyncio
async def test_capture_frame_v4l2_resizes_to_configured_resolution():
    from mousedroid.hardware.camera import jetson_csi

    mock_cv2 = MagicMock()
    mock_gstreamer_cap = MagicMock()
    mock_gstreamer_cap.isOpened.return_value = False
    mock_v4l2_cap = MagicMock()
    mock_v4l2_cap.isOpened.return_value = True
    source_frame = np.random.randint(0, 255, (2592, 4608, 3), dtype=np.uint8)
    resized_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mock_v4l2_cap.read.return_value = (True, source_frame)
    mock_cv2.VideoCapture.side_effect = [mock_gstreamer_cap, mock_v4l2_cap]
    mock_cv2.resize.return_value = resized_frame
    mock_cv2.CAP_GSTREAMER = 1800
    mock_cv2.CAP_PROP_FRAME_WIDTH = 3
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
    mock_cv2.CAP_PROP_FPS = 5

    cam = jetson_csi.JetsonCSICamera(_cfg(resolution_width=640, resolution_height=480))

    p_ju, p_cv = _patch_backends(jetson_csi, cv2=mock_cv2)
    with p_ju, p_cv:
        await cam.start()
        frame = cam._capture_frame()
        assert frame.shape == (480, 640, 3)
        mock_cv2.resize.assert_called_once_with(source_frame, (640, 480))


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


# ---------------------------------------------------------------------------
# capture_raw_jpeg + _frame_to_rgb_for_snapshot — RawFrameSourceProtocol
# conformance + the V4L2 grayscale-extract workaround for IMX708 sensors
# whose container lacks the nvarguscamerasrc GStreamer plugin (PR #104
# harden-2 follow-up).
# ---------------------------------------------------------------------------


def test_frame_to_rgb_jetson_utils_passes_rgb_through():
    """jetson_utils backend already returns RGB — no channel swap."""
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    cam = JetsonCSICamera(_cfg())
    cam._backend = "jetson_utils"
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[..., 0] = 200  # red plane only
    out = cam._frame_to_rgb_for_snapshot(frame)
    # No swap — red stays in slot 0.
    assert out[0, 0, 0] == 200
    assert out[0, 0, 1] == 0
    assert out[0, 0, 2] == 0


def test_frame_to_rgb_gstreamer_swaps_bgr_to_rgb():
    """gstreamer backend returns BGR — swap to RGB before Pillow encode."""
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    cam = JetsonCSICamera(_cfg())
    cam._backend = "gstreamer"
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[..., 0] = 200  # BGR: blue plane is 200
    out = cam._frame_to_rgb_for_snapshot(frame)
    # After swap, the blue value lands in RGB slot 2.
    assert out[0, 0, 2] == 200
    assert out[0, 0, 0] == 0


def test_frame_to_rgb_v4l2_grayscale_extract_uses_green_channel():
    """v4l2 backend with the workaround on → green channel cloned to R/G/B."""
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    cam = JetsonCSICamera(_cfg(v4l2_grayscale_extract=True))
    cam._backend = "v4l2"
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[..., 1] = 153  # the YUYV-misinterpretation puts luma into green
    frame[..., 0] = 0
    frame[..., 2] = 0
    out = cam._frame_to_rgb_for_snapshot(frame)
    # Every pixel should be grey at exactly the green value.
    assert np.all(out == 153)


def test_frame_to_rgb_v4l2_grayscale_extract_disabled_falls_back_to_bgr_swap():
    """v4l2 backend with the workaround OFF → standard BGR -> RGB swap."""
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    cam = JetsonCSICamera(_cfg(v4l2_grayscale_extract=False))
    cam._backend = "v4l2"
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[..., 0] = 200  # BGR blue
    out = cam._frame_to_rgb_for_snapshot(frame)
    # Workaround off → standard BGR -> RGB swap (200 lands in slot 2).
    assert out[0, 0, 2] == 200


@pytest.mark.asyncio
async def test_capture_raw_jpeg_round_trips_through_pillow():
    """Full capture_raw_jpeg → real Pillow-encoded JPEG bytes."""
    from io import BytesIO

    from PIL import Image

    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    cam = JetsonCSICamera(_cfg(snapshot_jpeg_quality=85))
    cam._backend = "gstreamer"
    fake = np.zeros((10, 16, 3), dtype=np.uint8)
    fake[..., 2] = 240  # BGR red plane
    cam._capture_frame = lambda: fake  # type: ignore[assignment]

    jpeg = await cam.capture_raw_jpeg()
    assert jpeg is not None
    assert len(jpeg) > 0
    img = Image.open(BytesIO(jpeg))
    assert img.format == "JPEG"
    assert img.size == (16, 10)


@pytest.mark.asyncio
async def test_capture_raw_jpeg_returns_none_on_empty_frame():
    """A zero-size frame from the underlying driver short-circuits to None."""
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    cam = JetsonCSICamera(_cfg())
    cam._backend = "v4l2"
    cam._capture_frame = lambda: np.zeros((0, 0, 3), dtype=np.uint8)  # type: ignore[assignment]

    assert await cam.capture_raw_jpeg() is None


@pytest.mark.asyncio
async def test_capture_raw_jpeg_satisfies_raw_frame_source_protocol():
    """JetsonCSICamera now satisfies RawFrameSourceProtocol structurally."""
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera
    from mousedroid.hardware.protocols import RawFrameSourceProtocol

    cam = JetsonCSICamera(_cfg())
    # Runtime-checkable Protocol via isinstance — what the telemetry server
    # factory uses to gate /camera/frame.jpg + /camera/stream registration.
    assert isinstance(cam, RawFrameSourceProtocol)


@pytest.mark.asyncio
async def test_capture_raw_jpeg_returns_none_when_pillow_unavailable():
    """Defensive: missing Pillow extra → return None so the server can 503."""
    import sys

    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    cam = JetsonCSICamera(_cfg())
    cam._backend = "v4l2"
    # Stash + clear the cached PIL.Image import so the inline `from PIL
    # import Image` inside capture_raw_jpeg triggers an ImportError.
    saved_pil = sys.modules.pop("PIL", None)
    saved_image = sys.modules.pop("PIL.Image", None)
    sys.modules["PIL"] = None  # type: ignore[assignment]
    try:
        jpeg = await cam.capture_raw_jpeg()
        assert jpeg is None
    finally:
        # Restore the module so other tests can still encode JPEGs.
        if saved_pil is not None:
            sys.modules["PIL"] = saved_pil
        elif "PIL" in sys.modules:
            del sys.modules["PIL"]
        if saved_image is not None:
            sys.modules["PIL.Image"] = saved_image


@pytest.mark.asyncio
async def test_capture_raw_jpeg_returns_none_when_pillow_rejects_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: Image.fromarray raising TypeError → return None (no 500)."""
    from PIL import Image

    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    cam = JetsonCSICamera(_cfg())
    cam._backend = "gstreamer"
    cam._capture_frame = lambda: np.ones((4, 6, 3), dtype=np.uint8)  # type: ignore[assignment]

    def _raises(*_a, **_kw):
        msg = "simulated PIL rejection (e.g. unsupported dtype)"
        raise TypeError(msg)

    monkeypatch.setattr(Image, "fromarray", _raises)
    jpeg = await cam.capture_raw_jpeg()
    assert jpeg is None
