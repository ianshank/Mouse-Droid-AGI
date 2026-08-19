"""Async camera perception drivers with hardware and mock fallbacks."""

from __future__ import annotations

import asyncio
from typing import Any

from mousedroid.config.schema.hardware import CameraConfig
from mousedroid.constants import (
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    MOCK_ASYNC_YIELD_S,
    MOCK_CAMERA_PIXEL_WHITE,
)
from mousedroid.interfaces.protocols import CameraProtocol
from mousedroid.logging.setup import get_logger

_log = get_logger("mousedroid.hardware.camera")


class CSICamera(CameraProtocol):
    """Jetson CSI / V4L2 Camera capture driver."""

    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._healthy = True
        self._closed = False
        _log.info(
            "camera_csi_initialized",
            width=self._cfg.resolution_width,
            height=self._cfg.resolution_height,
            fps=self._cfg.fps,
        )

    def is_healthy(self) -> bool:
        """Return True if camera pipeline is operating cleanly."""
        return self._healthy and not self._closed

    async def capture_frame(self) -> Any:
        """Capture frame asynchronously.

        Returns:
            2D synthetic frame array shaped according to configured resolution.
        """
        if not self.is_healthy():
            _log.warning("capture_frame_called_on_unhealthy_camera")
            return None
        await asyncio.sleep(MOCK_ASYNC_YIELD_S)
        return [[0] * self._cfg.resolution_width for _ in range(self._cfg.resolution_height)]

    async def close(self) -> None:
        """Release camera sensor hardware resources."""
        self._closed = True
        self._healthy = False
        _log.info("camera_csi_closed")


class MockCamera(CameraProtocol):
    """Deterministic mock camera driver for testing and development."""

    def __init__(
        self, width: int = DEFAULT_CAMERA_WIDTH, height: int = DEFAULT_CAMERA_HEIGHT
    ) -> None:
        self._width = width
        self._height = height
        self._healthy = True
        self.frame_count: int = 0
        self.closed: bool = False

    def is_healthy(self) -> bool:
        """Return mock health status."""
        return self._healthy

    async def capture_frame(self) -> Any:
        """Return deterministic mock frame."""
        self.frame_count += 1
        return [[MOCK_CAMERA_PIXEL_WHITE] * self._width for _ in range(self._height)]

    async def close(self) -> None:
        """Close mock camera."""
        self._healthy = False
        self.closed = True
