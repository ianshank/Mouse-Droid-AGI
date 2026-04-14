"""Hardware test configuration — skip guards for device-dependent tests.

Tests in this directory require physical hardware (Jetson, sensors, etc.).
They are skipped unless ``MOUSEDROID_HARDWARE_TESTS=true`` is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Auto-skip all hardware tests unless explicitly enabled
pytestmark = pytest.mark.hardware

_HARDWARE_ENABLED = os.environ.get("MOUSEDROID_HARDWARE_TESTS", "").lower() == "true"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip hardware-marked tests when MOUSEDROID_HARDWARE_TESTS is not set."""
    if _HARDWARE_ENABLED:
        return
    skip = pytest.mark.skip(reason="MOUSEDROID_HARDWARE_TESTS not set")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip)


# Device-existence skip decorators
requires_esp32 = pytest.mark.skipif(
    not Path("/dev/ttyUSB0").exists(),
    reason="ESP32 not connected at /dev/ttyUSB0",
)

requires_lidar = pytest.mark.skipif(
    not Path("/dev/ttyUSB1").exists(),
    reason="LiDAR not connected at /dev/ttyUSB1",
)

requires_camera = pytest.mark.skipif(
    not Path("/dev/video0").exists(),
    reason="Camera not available at /dev/video0",
)

requires_audio = pytest.mark.skipif(
    not Path("/dev/snd").exists(),
    reason="ALSA audio device not available at /dev/snd",
)

try:
    import torch

    _has_gpu = torch.cuda.is_available()
except ImportError:
    _has_gpu = False

requires_gpu = pytest.mark.skipif(not _has_gpu, reason="CUDA GPU not available")
