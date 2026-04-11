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

    cfg = Settings(mock_hardware=True, cognitive={"auto_download": False, "enabled": True})
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

    cfg = Settings(mock_hardware=True, cognitive={"auto_download": False, "enabled": True})
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
        mock_hardware=True,
        cognitive={"auto_download": False, "weights_dir": "nonexistent/", "enabled": True},
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
        mock_hardware=True,
        cognitive={"weights_dir": custom_dir, "auto_download": False, "enabled": True},
    )
    core = build_cognitive_core(cfg)
    # Should still initialize even if weights_dir doesn't exist
    assert isinstance(core, CognitiveCore)


def test_resolve_bdi_weights_local_path(tmp_path):
    """Test _resolve_bdi_weights returns local weights when present."""
    from unittest.mock import patch

    from mousedroid.cognitive.bdi_model import NeuralBDI
    from mousedroid.factory import _resolve_bdi_weights

    # Create weight files on disk
    for name in ["belief.npz", "desire.npz", "intention.npz", "affect.npz"]:
        (tmp_path / name).touch()

    cfg = Settings(
        mock_hardware=True,
        cognitive={"weights_dir": str(tmp_path), "auto_download": False, "enabled": True},
    )

    # Mock NeuralBDI construction to avoid loading empty npz files
    mock_bdi = MagicMock(spec=NeuralBDI)
    with patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=mock_bdi):
        bdi, source = _resolve_bdi_weights(cfg)

    assert source == "local"
    assert bdi is mock_bdi


def test_resolve_bdi_weights_huggingface_download(tmp_path):
    """Test _resolve_bdi_weights downloads from HF when local missing."""
    from unittest.mock import patch

    from mousedroid.factory import _resolve_bdi_weights

    cfg = Settings(
        mock_hardware=True,
        cognitive={
            "weights_dir": str(tmp_path / "nonexistent"),
            "auto_download": True,
            "enabled": True,
        },
    )

    # Patch at the modules where the function imports from
    with (
        patch(
            "mousedroid.utils.download_weights_from_huggingface",
            return_value=True,
        ),
        patch("mousedroid.cognitive.bdi_model.NeuralBDI") as mock_bdi_cls,
    ):
        mock_bdi_cls.return_value = MagicMock()
        bdi, source = _resolve_bdi_weights(cfg)

    assert source == "huggingface"
    assert bdi is not None


def test_build_orchestrator_cognitive_fallback_on_failure():
    """Test build_orchestrator falls back to MCTS when cognitive init fails."""
    from unittest.mock import patch

    from mousedroid.factory import build_orchestrator
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    cfg = Settings(
        mock_hardware=True,
        cognitive={"enabled": True, "auto_download": False, "fallback_to_mcts": True},
    )

    with patch(
        "mousedroid.factory.build_cognitive_core",
        side_effect=RuntimeError("cognitive init failed"),
    ):
        orch = build_orchestrator(cfg)

    assert isinstance(orch, MouseDroidOrchestrator)
    assert orch._cognitive_core is None


def test_build_orchestrator_cognitive_no_fallback_raises():
    """Test build_orchestrator raises when cognitive fails and fallback disabled."""
    from unittest.mock import patch

    from mousedroid.factory import build_orchestrator

    cfg = Settings(
        mock_hardware=True,
        cognitive={"enabled": True, "auto_download": False, "fallback_to_mcts": False},
    )

    with (
        patch(
            "mousedroid.factory.build_cognitive_core",
            side_effect=RuntimeError("cognitive init failed"),
        ),
        pytest.raises(RuntimeError, match="cognitive init failed"),
    ):
        build_orchestrator(cfg)


def test_build_health_monitor():
    """Test build_health_monitor returns a HealthMonitor instance."""
    from mousedroid.factory import build_health_monitor
    from mousedroid.health.monitor import HealthMonitor

    cfg = Settings(mock_hardware=True)
    monitor = build_health_monitor(cfg)
    assert isinstance(monitor, HealthMonitor)


def test_build_sensor_manager():
    """Test build_sensor_manager returns a SensorManager instance."""
    from mousedroid.factory import build_sensor_manager
    from mousedroid.sensing.manager import SensorManager

    cfg = Settings(mock_hardware=True)
    vision = MagicMock()
    distance = MagicMock()
    esp32 = MagicMock()

    manager = build_sensor_manager(cfg, vision=vision, distance=distance, esp32=esp32)
    assert isinstance(manager, SensorManager)


def test_build_sensor_manager_with_microphone():
    """Test build_sensor_manager accepts optional microphone."""
    from mousedroid.factory import build_sensor_manager
    from mousedroid.sensing.manager import SensorManager

    cfg = Settings(mock_hardware=True)
    vision = MagicMock()
    distance = MagicMock()
    esp32 = MagicMock()
    microphone = MagicMock()

    manager = build_sensor_manager(
        cfg,
        vision=vision,
        distance=distance,
        esp32=esp32,
        microphone=microphone,
    )
    assert isinstance(manager, SensorManager)
