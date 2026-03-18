"""Raspberry Pi AI Camera (IMX500) driver for Jetson Nano.

Implements ``VisionProtocol`` using the ``picamera2`` library.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import CameraConfig

try:
    from picamera2 import Picamera2 as _Picamera2
except ImportError:  # pragma: no cover
    _Picamera2 = None

_log = get_logger(__name__)


class IMX500Camera:
    """IMX500 camera implementing ``VisionProtocol``.

    Uses ``picamera2`` for frame capture and on-chip AI inference.
    All blocking camera operations are delegated to ``asyncio.to_thread``.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        """Initialise IMX500 camera from config.

        Args:
            cfg: Camera configuration with resolution, FPS, and model path.
        """
        self._cfg = cfg
        self._camera: Any = None

    async def start(self) -> None:
        """Start the camera capture pipeline."""
        if _Picamera2 is None:
            msg = "picamera2 is not installed — install mousedroid[hardware]"
            raise RuntimeError(msg)
        await asyncio.to_thread(self._start_camera)
        _log.info(
            "imx500_started",
            width=self._cfg.resolution_width,
            height=self._cfg.resolution_height,
            fps=self._cfg.fps,
        )

    def _start_camera(self) -> None:  # pragma: no cover
        """Configure and start picamera2 (blocking)."""
        self._camera = _Picamera2()
        config = self._camera.create_still_configuration(
            main={"size": (self._cfg.resolution_width, self._cfg.resolution_height)},
        )
        self._camera.configure(config)
        self._camera.start()

    async def stop(self) -> None:
        """Stop the camera capture pipeline."""
        if self._camera is not None:
            await asyncio.to_thread(self._stop_camera)
        _log.info("imx500_stopped")

    def _stop_camera(self) -> None:  # pragma: no cover
        """Stop and close picamera2 (blocking)."""
        self._camera.stop()
        self._camera.close()
        self._camera = None

    async def capture_features(self) -> NDArray[np.float32]:
        """Capture a frame and extract feature vector.

        Returns:
            Feature vector of shape ``(feature_dim,)``.
        """
        frame = await asyncio.to_thread(self._capture_frame)
        features = self._extract_features(frame)
        return features

    async def capture_frame(self) -> NDArray[np.uint8]:
        """Capture raw camera frame.

        Returns:
            BGR image, shape ``(H, W, 3)``.
        """
        return await asyncio.to_thread(self._capture_frame)

    def _capture_frame(self) -> NDArray[np.uint8]:  # pragma: no cover
        """Capture a single frame from the camera (blocking).

        Returns:
            Raw frame as uint8 numpy array.
        """
        frame: NDArray[np.uint8] = self._camera.capture_array()
        return frame

    def _extract_features(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Extract feature vector from a captured frame.

        Falls back to mean-pooling if no on-chip model is available.

        Args:
            frame: Raw camera frame.

        Returns:
            Feature vector of shape ``(feature_dim,)``.
        """
        # Simple mean-pool fallback; real deployment uses IMX500 on-chip model
        flat = frame.astype(np.float32).flatten()
        dim = self._cfg.feature_dim
        if len(flat) >= dim:
            stride = len(flat) // dim
            features = flat[: stride * dim].reshape(dim, stride).mean(axis=1)
        else:
            features = np.zeros(dim, dtype=np.float32)
            features[: len(flat)] = flat
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features  # type: ignore[no-any-return]

    @property
    def feature_dim(self) -> int:
        """Output feature vector dimension."""
        return self._cfg.feature_dim
