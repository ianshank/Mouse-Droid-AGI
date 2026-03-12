"""Hardware smoke tests for Jetson Nano deployment.

These tests are intended to run **on the Jetson** with real hardware
attached. They are marked with ``@pytest.mark.hardware`` and excluded
from the default CI suite.

Run with::

    pytest -m hardware -v --timeout=30 tests/hardware/
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytestmark = pytest.mark.hardware


# ---------------------------------------------------------------------------
# 1. GPU / CUDA
# ---------------------------------------------------------------------------


def test_gpu_available() -> None:
    """Verify CUDA GPU is visible to PyTorch."""
    import torch

    assert torch.cuda.is_available(), "CUDA GPU not available"


def test_tensorrt_importable() -> None:
    """Verify TensorRT can be imported and has a version string."""
    import tensorrt

    assert hasattr(tensorrt, "__version__"), "tensorrt missing __version__"
    assert isinstance(tensorrt.__version__, str)


# ---------------------------------------------------------------------------
# 2. GPIO
# ---------------------------------------------------------------------------


def test_gpio_pins_accessible() -> None:
    """Verify GPIO pins 23/24 can be set up and cleaned up."""
    import Jetson.GPIO as GPIO

    GPIO.setmode(GPIO.BCM)
    try:
        GPIO.setup(23, GPIO.OUT)
        GPIO.setup(24, GPIO.IN)
    finally:
        GPIO.cleanup()


# ---------------------------------------------------------------------------
# 3. Serial
# ---------------------------------------------------------------------------


def test_serial_port_exists() -> None:
    """Verify the ESP32 serial device node exists."""
    assert Path("/dev/ttyUSB0").exists(), "ESP32 serial port /dev/ttyUSB0 not found"


# ---------------------------------------------------------------------------
# 4. Camera
# ---------------------------------------------------------------------------


def test_camera_capture() -> None:
    """Capture one frame and verify expected resolution (480x640)."""
    frame = None

    # Try picamera2 first
    try:
        from picamera2 import Picamera2

        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": (640, 480)},
        )
        cam.configure(config)
        cam.start()
        try:
            import time

            time.sleep(0.5)  # allow auto-exposure
            frame = cam.capture_array()
        finally:
            cam.stop()
            cam.close()
    except ImportError:
        pass

    # Fallback: jetson_utils
    if frame is None:
        try:
            import jetson_utils

            cam = jetson_utils.videoSource(
                "csi://0",
                argv=["--input-width=640", "--input-height=480"],
            )
            cuda_img = cam.Capture()
            frame = jetson_utils.cudaToNumpy(cuda_img)
        except ImportError:
            pytest.skip("No camera library available (picamera2 or jetson_utils)")

    assert frame is not None, "Camera returned None frame"
    height, width = frame.shape[0], frame.shape[1]
    assert height == 480, f"Expected height 480, got {height}"
    assert width == 640, f"Expected width 640, got {width}"


# ---------------------------------------------------------------------------
# 5. Health monitor
# ---------------------------------------------------------------------------


def test_health_monitor() -> None:
    """Instantiate HealthMonitor and run check_health()."""
    from mousedroid.config.schema import HealthConfig, JetsonConfig
    from mousedroid.health.monitor import HealthMonitor

    health_cfg = HealthConfig()
    jetson_cfg = JetsonConfig()
    monitor = HealthMonitor(health_cfg=health_cfg, jetson_cfg=jetson_cfg)

    result = asyncio.run(monitor.check_health())
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] in ("ok", "warning", "critical")
    assert "gpu_temp_c" in result


# ---------------------------------------------------------------------------
# 6. Ultrasonic sensor
# ---------------------------------------------------------------------------


def test_ultrasonic_read() -> None:
    """Instantiate HcSr04 and attempt a distance read.

    On a bench without the physical sensor wired, the read may time out
    and return ``max_range_m``. We verify the driver initialises and
    returns a float within the configured range.
    """
    from mousedroid.config.schema import UltrasonicConfig
    from mousedroid.hardware.sensors.ultrasonic import HcSr04

    cfg = UltrasonicConfig(trigger_pin=23, echo_pin=24)
    sensor = HcSr04(cfg)

    try:
        distance = asyncio.run(sensor.read_distance_m())
        assert isinstance(distance, float)
        assert 0.0 <= distance <= cfg.max_range_m
    except RuntimeError:
        # GPIO unavailable on non-Jetson hardware
        pytest.skip("GPIO unavailable — not running on Jetson hardware")


# ---------------------------------------------------------------------------
# 7. ESP32 serial driver
# ---------------------------------------------------------------------------


def test_esp32_connect() -> None:
    """Connect to ESP32 over serial, send emergency stop, and disconnect."""
    from mousedroid.comms.serial_driver import SerialESP32Driver
    from mousedroid.config.schema import ESP32Config

    cfg = ESP32Config()
    driver = SerialESP32Driver(cfg)

    async def _run() -> None:
        await driver.connect()
        try:
            await driver.emergency_stop()
        finally:
            await driver.disconnect()

    asyncio.run(_run())
