"""Jetson Nano CSI camera driver.

Implements ``VisionProtocol`` using ``jetson_utils`` (from jetson-inference)
with OpenCV fallbacks via GStreamer ``nvarguscamerasrc`` and direct V4L2.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import CameraConfig

_jetson_utils: Any
try:
    import jetson_utils as _jetson_utils_mod

    _jetson_utils = _jetson_utils_mod
except ImportError:  # pragma: no cover
    _jetson_utils = None

_cv2: Any
try:
    import cv2 as _cv2_mod

    _cv2 = _cv2_mod
except ImportError:  # pragma: no cover
    _cv2 = None

_log = get_logger(__name__)


class JetsonCSICamera:
    """Jetson CSI camera implementing ``VisionProtocol``.

    Prefers ``jetson_utils`` for zero-copy CUDA capture. Falls back to
    OpenCV with a GStreamer ``nvarguscamerasrc`` pipeline, then to direct
    V4L2 capture on the configured ``device_path``.

    All blocking camera operations are delegated to ``asyncio.to_thread``.
    """

    def __init__(self, cfg: CameraConfig, **kwargs: Any) -> None:
        """Initialise Jetson CSI camera from config.

        Args:
            cfg: Camera configuration with resolution, FPS, and model path.
            **kwargs: Optional ``hailo_runtime`` for accelerated feature extraction.
        """
        self._cfg = cfg
        self._camera: Any = None
        self._backend: str | None = None

        from mousedroid.hardware.camera.feature_extractor import build_feature_extractor

        self._extractor = build_feature_extractor(cfg, hailo_runtime=kwargs.get("hailo_runtime"))

    async def start(self) -> None:
        """Start the camera capture pipeline."""
        if _jetson_utils is None and _cv2 is None:
            msg = (
                "Neither jetson_utils nor OpenCV is installed — "
                "install jetson-inference or opencv-python with GStreamer/V4L2 support"
            )
            raise RuntimeError(msg)
        start_timeout_s: float = self._cfg.start_timeout_s
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._start_camera),
                timeout=start_timeout_s,
            )
        except TimeoutError:
            _log.warning(
                "jetson_csi_start_timeout_degraded",
                timeout_s=start_timeout_s,
                device_path=self._cfg.device_path,
            )
            # Non-fatal: camera remains None (degraded). The orchestrator
            # can continue startup and bind the telemetry server. Capture
            # calls will fail individually rather than blocking everything.
            return
        except Exception:
            _log.warning(
                "jetson_csi_start_failed_degraded",
                device_path=self._cfg.device_path,
                exc_info=True,
            )
            return
        _log.info(
            "jetson_csi_started",
            backend=self._backend,
            width=self._cfg.resolution_width,
            height=self._cfg.resolution_height,
            fps=self._cfg.fps,
        )

    def _start_camera(self) -> None:  # pragma: no cover
        """Configure and start the CSI camera (blocking).

        Tries ``jetson_utils.videoSource`` first, then falls back to
        OpenCV with a GStreamer ``nvarguscamerasrc`` pipeline and finally
        direct V4L2 capture.
        """
        if _jetson_utils is not None:
            try:
                self._camera = _jetson_utils.videoSource(
                    "csi://0",
                    argv=[
                        f"--input-width={self._cfg.resolution_width}",
                        f"--input-height={self._cfg.resolution_height}",
                        f"--input-rate={self._cfg.fps}",
                    ],
                )
                self._backend = "jetson_utils"
                return
            except Exception:
                _log.warning("jetson_utils_csi_failed_falling_back_to_gstreamer")

        if _cv2 is None:
            msg = "OpenCV is not available for GStreamer fallback"
            raise RuntimeError(msg)

        gst_pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM),"
            f"width={self._cfg.resolution_width},"
            f"height={self._cfg.resolution_height},"
            f"framerate={self._cfg.fps}/1 ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=1"
        )
        cap = _cv2.VideoCapture(gst_pipeline, _cv2.CAP_GSTREAMER)
        if cap.isOpened():
            self._camera = cap
            self._backend = "gstreamer"
            return

        cap.release()
        _log.warning(
            "jetson_csi_gstreamer_failed_falling_back_to_v4l2",
            device_path=self._cfg.device_path,
        )

        v4l2_cap = _cv2.VideoCapture(self._cfg.device_path)
        if not v4l2_cap.isOpened():
            msg = (
                "Failed to open CSI camera via GStreamer pipeline or "
                f"V4L2 device {self._cfg.device_path}"
            )
            raise RuntimeError(msg)

        for prop, value in (
            (getattr(_cv2, "CAP_PROP_FRAME_WIDTH", None), self._cfg.resolution_width),
            (getattr(_cv2, "CAP_PROP_FRAME_HEIGHT", None), self._cfg.resolution_height),
            (getattr(_cv2, "CAP_PROP_FPS", None), self._cfg.fps),
        ):
            if prop is not None:
                v4l2_cap.set(prop, value)

        self._camera = v4l2_cap
        self._backend = "v4l2"

    async def stop(self) -> None:
        """Stop the camera capture pipeline."""
        if self._camera is not None:
            await asyncio.to_thread(self._stop_camera)
        _log.info("jetson_csi_stopped")

    def _stop_camera(self) -> None:  # pragma: no cover
        """Stop and release the camera (blocking)."""
        if self._backend in {"gstreamer", "v4l2"}:
            self._camera.release()
        else:
            # jetson_utils sources are cleaned up on deletion
            del self._camera
        self._camera = None
        self._backend = None

    async def capture_features(self) -> NDArray[np.float32]:
        """Capture a frame and extract feature vector.

        Returns:
            Feature vector of shape ``(feature_dim,)``.
        """
        frame = await asyncio.to_thread(self._capture_frame)
        return self._extract_features(frame)

    async def capture_raw_frame(self) -> NDArray[np.uint8]:
        """Capture a single raw frame as a uint8 NumPy array.

        Returns:
            Raw frame array of shape ``(H, W, channels)``.
        """
        return np.asarray(await asyncio.to_thread(self._capture_frame), dtype=np.uint8)

    async def capture_raw_jpeg(self) -> bytes | None:
        """Capture a frame and encode as JPEG for ``RawFrameSourceProtocol``.

        Implementing this method makes the camera satisfy
        :class:`mousedroid.hardware.protocols.RawFrameSourceProtocol`, which is
        what the telemetry server's ``isinstance`` check uses to decide
        whether to register the ``/camera/frame.jpg`` and ``/camera/stream``
        endpoints. Without it, the factory's ``raw_frame_source`` resolves to
        ``None`` and the dashboard's camera pane returns HTTP 404.

        Three backend-specific colour-conversion paths:

        * ``jetson_utils`` — already RGB; no swap needed.
        * ``gstreamer`` (the ``nvarguscamerasrc`` path) — already BGR after
          ``videoconvert``; swap to RGB before Pillow encoding.
        * ``v4l2`` — fallback path used when the container lacks the
          ``nvarguscamerasrc`` plugin. The IMX708 sensor only outputs RG10
          Bayer raw, the kernel driver advertises ``YUYV`` at the active
          format but the bytes are Bayer-packed, and OpenCV's ``YUYV->BGR``
          conversion produces uniform green output because the sensor isn't
          being driven through its ISP. With
          ``cfg.camera.v4l2_grayscale_extract = True`` (default), the green
          channel of the misinterpreted frame is extracted as luma and
          cloned across R/G/B so the operator sees the scene (with mosaic
          artefacts) instead of solid green. Disable the override once the
          container has proper ``nvarguscamerasrc`` support.

        Returns:
            JPEG bytes, or ``None`` if Pillow is unavailable / the frame
            grab failed.
        """
        try:
            from io import BytesIO

            from PIL import Image
        except ImportError:
            return None

        frame = await asyncio.to_thread(self._capture_frame)
        if frame is None or frame.size == 0:
            return None

        rgb = self._frame_to_rgb_for_snapshot(frame)
        try:
            img = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
        except (TypeError, ValueError):
            return None

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=int(self._cfg.snapshot_jpeg_quality))
        return buf.getvalue()

    def _frame_to_rgb_for_snapshot(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Backend-specific colour conversion for the JPEG snapshot path.

        Split out from :meth:`capture_raw_jpeg` so it stays pure + testable
        without needing a live ``/dev/video0`` device. See the
        ``capture_raw_jpeg`` docstring for the per-backend rationale.
        """
        if (
            self._backend == "v4l2"
            and self._cfg.v4l2_grayscale_extract
            and frame.ndim == 3
            and frame.shape[2] == 3
        ):
            # Green channel of OpenCV's YUYV-misinterpretation carries the
            # actual luma signal — clone it to R/G/B for an honest grayscale
            # snapshot. See the field's docstring for the IMX708 + container-
            # GStreamer-plugin background.
            gray = frame[..., 1]
            return np.stack([gray, gray, gray], axis=-1)
        # Defensive guard for a future V4L2 ``GREY`` mode that returns a 2-D
        # luma plane directly: skip the BGR-swap (which would IndexError on
        # ``frame[..., ::-1]`` for ndim==2) and clone the luma to RGB. Today
        # this path is unreachable because every backend produces a 3-D
        # frame, but the explicit branch is cheap insurance against a future
        # driver-mode addition.
        if frame.ndim == 2:
            return np.stack([frame, frame, frame], axis=-1)
        if self._backend != "jetson_utils":
            # OpenCV (gstreamer / v4l2 paths when grayscale-extract is off)
            # returns BGR; swap to RGB before Pillow encodes the snapshot.
            return frame[..., ::-1]
        # jetson_utils returns RGB directly.
        return frame

    def _capture_frame(self) -> NDArray[np.uint8]:  # pragma: no cover
        """Capture a single frame from the camera (blocking).

        Returns:
            Raw frame as uint8 numpy array.
        """
        if self._backend == "jetson_utils":
            cuda_img = self._camera.Capture()
            frame: NDArray[np.uint8] = _jetson_utils.cudaToNumpy(cuda_img)
            return frame

        # GStreamer / OpenCV path
        ret, frame_cv = self._camera.read()
        if not ret:
            _log.warning("jetson_csi_frame_capture_failed")
            return np.zeros(
                (self._cfg.resolution_height, self._cfg.resolution_width, 3),
                dtype=np.uint8,
            )
        frame = np.asarray(frame_cv, dtype=np.uint8)

        if self._backend == "v4l2" and (
            frame.shape[0] != self._cfg.resolution_height
            or frame.shape[1] != self._cfg.resolution_width
        ):
            _log.info(
                "jetson_csi_v4l2_resizing_frame",
                source_width=int(frame.shape[1]),
                source_height=int(frame.shape[0]),
                target_width=self._cfg.resolution_width,
                target_height=self._cfg.resolution_height,
            )
            frame = np.asarray(
                _cv2.resize(frame, (self._cfg.resolution_width, self._cfg.resolution_height)),
                dtype=np.uint8,
            )

        return frame

    def _extract_features(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Extract feature vector from a captured frame.

        Delegates to the configured feature extractor (mean-pool or TensorRT).

        Args:
            frame: Raw camera frame.

        Returns:
            Feature vector of shape ``(feature_dim,)``.
        """
        return self._extractor.extract(frame)

    @property
    def feature_dim(self) -> int:
        """Output feature vector dimension."""
        return self._cfg.feature_dim
