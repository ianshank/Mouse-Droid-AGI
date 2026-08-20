"""Unit tests for CSICamera and MockCamera perception drivers."""

from __future__ import annotations

import pytest

from mousedroid.config.schema.hardware import CameraConfig
from mousedroid.hardware.camera_csi import CSICamera, MockCamera


@pytest.mark.asyncio
async def test_csi_camera_lifecycle() -> None:
    """Verify CSICamera initialization, frame capture, and teardown."""
    cfg = CameraConfig(resolution_width=320, resolution_height=240, fps=30)
    cam = CSICamera(cfg)

    assert cam.is_healthy() is True
    frame = await cam.capture_frame()
    assert len(frame) == 240
    assert len(frame[0]) == 320

    await cam.close()
    assert cam.is_healthy() is False
    assert await cam.capture_frame() is None


@pytest.mark.asyncio
async def test_mock_camera() -> None:
    """Verify MockCamera returns valid synthetic frames."""
    cam = MockCamera(width=640, height=480)
    assert cam.is_healthy() is True

    frame = await cam.capture_frame()
    assert len(frame) == 480
    assert len(frame[0]) == 640
    assert cam.frame_count == 1

    await cam.close()
    assert cam.is_healthy() is False
    assert cam.closed is True
