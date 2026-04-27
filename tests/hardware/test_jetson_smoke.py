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

from mousedroid.validation.runtime import camera_unavailable_reason, capture_camera_frame

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


def test_gpio_pins_accessible(jetson_settings) -> None:
    """Verify configured GPIO pins can be set up and cleaned up."""
    import Jetson.GPIO as GPIO

    if jetson_settings.ultrasonic is None:
        pytest.skip("ultrasonic disabled in config")

    trigger_pin = jetson_settings.ultrasonic.trigger_pin
    echo_pin = jetson_settings.ultrasonic.echo_pin

    GPIO.setmode(GPIO.BCM)
    try:
        GPIO.setup(trigger_pin, GPIO.OUT)
        GPIO.setup(echo_pin, GPIO.IN)
    finally:
        GPIO.cleanup()


# ---------------------------------------------------------------------------
# 3. Serial
# ---------------------------------------------------------------------------


def test_serial_port_exists(jetson_settings) -> None:
    """Verify the ESP32 serial device node exists."""
    serial_port = Path(jetson_settings.esp32.serial_port)
    assert serial_port.exists(), f"ESP32 serial port {serial_port} not found"


# ---------------------------------------------------------------------------
# 4. Camera
# ---------------------------------------------------------------------------


def test_camera_capture(jetson_settings) -> None:
    """Capture one frame and verify the configured resolution."""
    try:
        frame, backend_name = asyncio.run(capture_camera_frame(jetson_settings))
    except RuntimeError as exc:
        reason = camera_unavailable_reason(jetson_settings, exc)
        if reason is not None:
            pytest.skip(reason)
        raise

    assert frame is not None, "Camera returned None frame"
    height, width = frame.shape[0], frame.shape[1]
    assert height == jetson_settings.camera.resolution_height, (
        f"Expected height {jetson_settings.camera.resolution_height}, got {height} "
        f"via {backend_name}"
    )
    assert (
        width == jetson_settings.camera.resolution_width
    ), f"Expected width {jetson_settings.camera.resolution_width}, got {width} via {backend_name}"


# ---------------------------------------------------------------------------
# 5. Health monitor
# ---------------------------------------------------------------------------


def test_health_monitor(jetson_settings) -> None:
    """Instantiate HealthMonitor and run check_health()."""
    from mousedroid.health.monitor import HealthMonitor

    monitor = HealthMonitor(
        health_cfg=jetson_settings.health,
        jetson_cfg=jetson_settings.jetson,
    )

    result = asyncio.run(monitor.check_health())
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] in ("ok", "warning", "critical")
    assert "gpu_temp_c" in result


# ---------------------------------------------------------------------------
# 6. Ultrasonic sensor
# ---------------------------------------------------------------------------


def test_ultrasonic_read(jetson_settings) -> None:
    """Instantiate HcSr04 and attempt a distance read.

    On a bench without the physical sensor wired, the read may time out
    and return ``max_range_m``. We verify the driver initialises and
    returns a float within the configured range.
    """
    from mousedroid.hardware.sensors.ultrasonic import HcSr04

    cfg = jetson_settings.ultrasonic
    if cfg is None:
        pytest.skip("ultrasonic disabled in config")

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


def test_esp32_connect(jetson_settings) -> None:
    """Connect to ESP32 over serial, send emergency stop, and disconnect."""
    from mousedroid.comms.serial_driver import SerialESP32Driver

    driver = SerialESP32Driver(jetson_settings.esp32)

    async def _run() -> None:
        await driver.connect()
        try:
            await driver.emergency_stop()
        finally:
            await driver.disconnect()

    asyncio.run(_run())
