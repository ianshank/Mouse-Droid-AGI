"""Tests for factory functions — coverage for real-hardware branches."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema import Settings

_ULTRASONIC_CFG = {"trigger_pin": 17, "echo_pin": 27}


def _real_settings(**overrides):
    """Create Settings with mock_hardware=False and valid ultrasonic config."""
    defaults = {"mock_hardware": False, "ultrasonic": _ULTRASONIC_CFG}
    defaults.update(overrides)
    return Settings(**defaults)


def test_build_esp32_serial():
    cfg = _real_settings(esp32={"protocol": "serial"})
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(cfg)
    from mousedroid.comms.serial_driver import SerialESP32Driver

    assert isinstance(driver, SerialESP32Driver)


def test_build_esp32_wifi():
    cfg = _real_settings(esp32={"protocol": "wifi"})
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(cfg)
    from mousedroid.comms.wifi_driver import WiFiESP32Driver

    assert isinstance(driver, WiFiESP32Driver)


def test_build_distance_sensor_missing_config_raises():
    # Must bypass the Settings validator to test the factory guard
    cfg = MagicMock(spec=Settings)
    cfg.mock_hardware = False
    cfg.ultrasonic = None

    from mousedroid.factory import build_distance_sensor

    with pytest.raises(ValueError, match="ultrasonic config required"):
        build_distance_sensor(cfg)


def test_build_distance_sensor_real_hardware():
    cfg = _real_settings()
    from mousedroid.factory import build_distance_sensor

    sensor = build_distance_sensor(cfg)
    from mousedroid.hardware.sensors.ultrasonic import HcSr04

    assert isinstance(sensor, HcSr04)


def test_build_camera_real_hardware_jetson_csi():
    cfg = _real_settings(camera={"backend": "jetson_csi"})
    from mousedroid.factory import build_camera

    camera = build_camera(cfg)
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    assert isinstance(camera, JetsonCSICamera)


def test_build_camera_real_hardware_auto_fallback(monkeypatch):
    """Auto backend falls back to JetsonCSI when picamera2 is unavailable."""
    import sys

    monkeypatch.setitem(sys.modules, "picamera2", None)
    cfg = _real_settings()
    from mousedroid.factory import build_camera
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    camera = build_camera(cfg)
    assert isinstance(camera, JetsonCSICamera)
