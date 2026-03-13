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
    from mousedroid.resilience.resilient_driver import ResilientESP32Driver

    assert isinstance(driver, ResilientESP32Driver)
    assert isinstance(driver.inner, SerialESP32Driver)


def test_build_esp32_wifi():
    cfg = _real_settings(esp32={"protocol": "wifi"})
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(cfg)
    from mousedroid.comms.wifi_driver import WiFiESP32Driver
    from mousedroid.resilience.resilient_driver import ResilientESP32Driver

    assert isinstance(driver, ResilientESP32Driver)
    assert isinstance(driver.inner, WiFiESP32Driver)


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


def test_build_camera_real_hardware_auto_picamera2(monkeypatch):
    """Auto backend selects IMX500 when picamera2 is importable."""
    import types

    fake_picamera2 = types.ModuleType("picamera2")
    fake_picamera2.Picamera2 = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "picamera2", fake_picamera2)

    cfg = _real_settings()
    from mousedroid.factory import build_camera
    from mousedroid.hardware.camera.imx500 import IMX500Camera

    camera = build_camera(cfg)
    assert isinstance(camera, IMX500Camera)


def test_build_cognitive_core_with_random_init():
    """Test cognitive core initialization with random weights fallback."""
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.factory import build_cognitive_core

    cfg = Settings(mock_hardware=True, cognitive={"auto_download": False})
    core = build_cognitive_core(cfg)
    assert isinstance(core, CognitiveCore)
    assert core._bdi is not None
    assert core._metacog is not None
    assert core._checker is not None


def test_build_cognitive_core_returns_fully_initialized():
    """Cognitive core is always fully initialized (never None)."""
    from mousedroid.cognitive.bdi_model import NeuralBDI
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.cognitive.constitutional_rl import ConstitutionalChecker
    from mousedroid.cognitive.metacognitive import MetacognitiveModel
    from mousedroid.factory import build_cognitive_core

    cfg = Settings(mock_hardware=True, cognitive={"auto_download": False})
    core = build_cognitive_core(cfg)

    assert isinstance(core, CognitiveCore)
    assert isinstance(core._bdi, NeuralBDI)
    assert isinstance(core._metacog, MetacognitiveModel)
    assert isinstance(core._checker, ConstitutionalChecker)


def test_build_cognitive_core_with_auto_download_false():
    """Test cognitive core with auto_download disabled."""
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.factory import build_cognitive_core

    cfg = Settings(
        mock_hardware=True, cognitive={"auto_download": False, "weights_dir": "nonexistent/"}
    )
    core = build_cognitive_core(cfg)
    # Should still return initialized core with random weights
    assert isinstance(core, CognitiveCore)
    assert core._bdi is not None


def test_build_cognitive_core_respects_weights_dir_config():
    """Cognitive core respects weights_dir from config."""
    from pathlib import Path

    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.factory import build_cognitive_core

    custom_dir = Path("custom_weights/")
    cfg = Settings(
        mock_hardware=True, cognitive={"weights_dir": custom_dir, "auto_download": False}
    )
    core = build_cognitive_core(cfg)
    # Should still initialize even if weights_dir doesn't exist
    assert isinstance(core, CognitiveCore)
