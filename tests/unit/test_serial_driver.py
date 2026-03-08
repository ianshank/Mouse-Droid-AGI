"""Tests for SerialESP32Driver."""

from __future__ import annotations

from mousedroid.comms.serial_driver import SerialESP32Driver, _clamp
from mousedroid.config.schema import ESP32Config


def _make_driver() -> SerialESP32Driver:
    cfg = ESP32Config()
    return SerialESP32Driver(cfg)


def test_constructor():
    driver = _make_driver()
    assert driver._connected is False
    assert driver._port == "/dev/ttyUSB0"
    assert driver._baud == 1_000_000


def test_constructor_custom_config():
    cfg = ESP32Config(serial_port="/dev/ttyS1", serial_baud=115200)
    driver = SerialESP32Driver(cfg)
    assert driver._port == "/dev/ttyS1"
    assert driver._baud == 115200


def test_initial_velocity_zero():
    driver = _make_driver()
    assert driver._last_velocity == (0.0, 0.0, 0.0)


def test_clamp_within_bounds():
    assert _clamp(0.5, 0.0, 1.0) == 0.5


def test_clamp_below():
    assert _clamp(-1.0, 0.0, 1.0) == 0.0


def test_clamp_above():
    assert _clamp(2.0, 0.0, 1.0) == 1.0
