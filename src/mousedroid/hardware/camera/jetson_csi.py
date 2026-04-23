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
    import jetson_utils as _jetson_utils  # type: ignore[no-redef]
except ImportError:  # pragma: no cover
    _jetson_utils = None

_cv2: Any
try:
    import cv2 as _cv2  # type: ignore[no-redef,unused-ignore]
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
        await asyncio.to_thread(self._start_camera)
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
            msg = f"Failed to open CSI camera via GStreamer pipeline or V4L2 device {self._cfg.device_path}"
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
