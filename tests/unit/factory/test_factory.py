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


def test_build_esp32_disabled_returns_mock_even_on_real_hardware() -> None:
    """``cfg.esp32.enabled = False`` swaps in MockESP32Driver even when mock_hardware=False.

    Regression net for the PR #104 harden-2 ESP32-tolerance landing: the
    Jetson runs without the motor controller plugged in (dashboard
    verification, hardware-bringup, etc.) and previously had to monkey-
    patch ``orchestrator.start()`` to swallow connect failures. With the new
    schema field the factory simply returns the mock so the orchestrator's
    full pipeline keeps working at real-hardware speeds.
    """
    cfg = _real_settings(esp32={"protocol": "serial", "enabled": False})

    from mousedroid.comms.mock_driver import MockESP32Driver
    from mousedroid.factory import build_esp32_driver
    from mousedroid.resilience.resilient_driver import ResilientESP32Driver

    driver = build_esp32_driver(cfg)
    assert isinstance(driver, ResilientESP32Driver)
    assert isinstance(driver.inner, MockESP32Driver)


def test_build_esp32_enabled_default_is_true_preserves_legacy_behavior() -> None:
    """Backwards-compat: ``enabled`` defaults to ``True`` — existing YAML keeps working."""
    from mousedroid.config.schema import ESP32Config

    cfg = ESP32Config()
    assert cfg.enabled is True
    # And the legacy ``protocol`` field is still ``serial`` by default.
    assert cfg.protocol == "serial"


def test_build_distance_sensor_missing_config_raises():
    # Must bypass the Settings validator to test the factory guard
    cfg = MagicMock(spec=Settings)
    cfg.mock_hardware = False
    cfg.ultrasonic = None

    from mousedroid.factory import build_distance_sensor

    with pytest.raises(ValueError, match="ultrasonic config required"):
        build_distance_sensor(cfg)


@pytest.mark.hardware
def test_build_distance_sensor_real_hardware():
    try:
        from mousedroid.hardware.sensors.ultrasonic import HcSr04
    except Exception:
        pytest.skip("Jetson.GPIO unavailable (container or non-Jetson host)")
    cfg = _real_settings()
    from mousedroid.factory import build_distance_sensor

    sensor = build_distance_sensor(cfg)
    assert isinstance(sensor, HcSr04)


def test_build_camera_real_hardware_jetson_csi():
    cfg = _real_settings(camera={"backend": "jetson_csi"})
    from mousedroid.factory import build_camera
    from mousedroid.resilience.resilient_camera import ResilientCamera

    camera = build_camera(cfg)
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    assert isinstance(camera, ResilientCamera)
    assert isinstance(camera.inner, JetsonCSICamera)


def test_build_camera_real_hardware_auto_fallback(monkeypatch):
    """Auto backend falls back to JetsonCSI when picamera2 is unavailable."""
    import sys

    monkeypatch.setitem(sys.modules, "picamera2", None)
    cfg = _real_settings()
    from mousedroid.factory import build_camera
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera
    from mousedroid.resilience.resilient_camera import ResilientCamera

    camera = build_camera(cfg)
    assert isinstance(camera, ResilientCamera)
    assert isinstance(camera.inner, JetsonCSICamera)


def test_build_camera_real_hardware_auto_picamera2(monkeypatch):
    """Auto backend selects IMX500 when picamera2 is importable."""
    import types

    fake_picamera2 = types.ModuleType("picamera2")
    fake_picamera2.Picamera2 = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "picamera2", fake_picamera2)

    cfg = _real_settings()
    from mousedroid.factory import build_camera
    from mousedroid.hardware.camera.imx500 import IMX500Camera
    from mousedroid.resilience.resilient_camera import ResilientCamera

    camera = build_camera(cfg)
    assert isinstance(camera, ResilientCamera)
    assert isinstance(camera.inner, IMX500Camera)


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

    # Patched where build_orchestrator actually looks it up
    # (mousedroid.factory.orchestrator), not the facade re-export -- the
    # mechanical rewiring rule (ADR-017) binds this as a module-level
    # `from mousedroid.factory.cognitive import build_cognitive_core` inside
    # factory/orchestrator.py, so the facade-level name is a separate,
    # already-diverged reference by the time this test runs.
    with patch(
        "mousedroid.factory.orchestrator.build_cognitive_core",
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

    # See test_build_orchestrator_cognitive_fallback_on_failure above for why
    # this patches mousedroid.factory.orchestrator, not the facade re-export.
    with (
        patch(
            "mousedroid.factory.orchestrator.build_cognitive_core",
            side_effect=RuntimeError("cognitive init failed"),
        ),
        pytest.raises(RuntimeError, match="cognitive init failed"),
    ):
        build_orchestrator(cfg)


def test_build_orchestrator_wires_log_buffer_from_telemetry_config():
    """build_orchestrator passes the configured log buffer size into TelemetryServer."""
    from mousedroid.factory import build_orchestrator

    cfg = Settings(
        mock_hardware=True,
        telemetry={"enabled": True, "force_real_server": True, "log_stream_buffer": 7},
    )

    orch = build_orchestrator(cfg)

    assert orch._telemetry_server is not None
    assert orch._telemetry_server._log_buffer is not None
    assert orch._telemetry_server._log_buffer._buffer.maxlen == 7


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


# ---------------------------------------------------------------------------
# Speaker / Voice factory tests
# ---------------------------------------------------------------------------


def test_build_speaker_disabled_no_config():
    """build_speaker returns None when speaker config is None."""
    from mousedroid.factory import build_speaker

    cfg = Settings(mock_hardware=True)  # speaker=None by default
    assert build_speaker(cfg) is None


def test_build_speaker_disabled_explicitly():
    """build_speaker returns None when speaker.enabled=False."""
    from mousedroid.factory import build_speaker

    cfg = Settings(mock_hardware=True, speaker={"enabled": False})
    assert build_speaker(cfg) is None


def test_build_speaker_mock_hardware():
    """build_speaker returns MockSpeaker when mock_hardware=True."""
    from mousedroid.factory import build_speaker
    from mousedroid.hardware.audio.mock_speaker import MockSpeaker

    cfg = Settings(mock_hardware=True, speaker={"enabled": True})
    speaker = build_speaker(cfg)
    assert isinstance(speaker, MockSpeaker)


def test_build_speaker_real_hardware():
    """build_speaker returns UsbSpeaker when mock_hardware=False."""
    from mousedroid.factory import build_speaker
    from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

    cfg = _real_settings(speaker={"enabled": True})
    speaker = build_speaker(cfg)
    assert isinstance(speaker, UsbSpeaker)


def test_build_voice_engine_disabled():
    """build_voice_engine returns None when voice.enabled=False."""
    from mousedroid.factory import build_voice_engine

    cfg = Settings(mock_hardware=True)  # voice.enabled=False by default
    assert build_voice_engine(cfg) is None


def test_build_voice_engine_no_speaker():
    """build_voice_engine returns None when no speaker available."""
    from mousedroid.factory import build_voice_engine

    cfg = Settings(mock_hardware=True, voice={"enabled": True})
    # No speaker config -> speaker=None -> voice disabled
    assert build_voice_engine(cfg) is None


def test_build_voice_engine_mock_hardware():
    """build_voice_engine returns RockyVoiceEngine with MockTTS."""
    from mousedroid.factory import build_voice_engine
    from mousedroid.voice.rocky import RockyVoiceEngine

    cfg = Settings(
        mock_hardware=True,
        voice={"enabled": True},
        speaker={"enabled": True},
    )
    engine = build_voice_engine(cfg)
    assert isinstance(engine, RockyVoiceEngine)


def test_build_voice_engine_with_provided_speaker():
    """build_voice_engine uses pre-built speaker when provided."""
    from mousedroid.factory import build_voice_engine
    from mousedroid.voice.rocky import RockyVoiceEngine

    cfg = Settings(mock_hardware=True, voice={"enabled": True})
    mock_speaker = MagicMock()
    mock_speaker.sample_rate = 22050
    mock_speaker.channels = 1
    mock_speaker.chunk_size = 1024
    engine = build_voice_engine(cfg, speaker=mock_speaker)
    assert isinstance(engine, RockyVoiceEngine)


def test_build_voice_engine_propagates_failure_recorder():
    """build_voice_engine threads its failure_recorder kwarg into the engine.

    Regression test for the factory-level wiring gap fixed alongside this
    test: build_orchestrator built a shared failure_recorder but never
    passed it to build_voice_engine, so RockyVoiceEngine silently fell
    back to its own NullFailureRecorder in production.
    """
    from mousedroid.factory import build_voice_engine
    from mousedroid.telemetry.failure_recorder import NullFailureRecorder

    cfg = Settings(mock_hardware=True, voice={"enabled": True}, speaker={"enabled": True})
    real_recorder = MagicMock(spec=NullFailureRecorder)
    engine = build_voice_engine(cfg, failure_recorder=real_recorder)
    assert engine is not None
    assert engine._failure_recorder is real_recorder


def test_build_voice_engine_sample_rate_mismatch():
    """build_voice_engine returns None when sample rates differ."""
    from mousedroid.factory import build_voice_engine

    cfg = Settings(
        mock_hardware=True,
        voice={"enabled": True, "tts_sample_rate": 22050},
    )
    mock_speaker = MagicMock()
    mock_speaker.sample_rate = 44100  # Mismatched
    mock_speaker.channels = 1
    mock_speaker.chunk_size = 1024
    assert build_voice_engine(cfg, speaker=mock_speaker) is None


# -- Hailo-8 factory functions -----------------------------------------------


def test_build_hailo_runtime_disabled_returns_none():
    from mousedroid.factory import build_hailo_runtime

    cfg = Settings(mock_hardware=True)
    assert build_hailo_runtime(cfg) is None


def test_build_hailo_runtime_none_config_returns_none():
    from mousedroid.factory import build_hailo_runtime

    cfg = Settings(mock_hardware=True, hailo=None)
    assert build_hailo_runtime(cfg) is None


def test_build_hailo_runtime_disabled_explicitly_returns_none():
    from mousedroid.config.schema import HailoConfig
    from mousedroid.factory import build_hailo_runtime

    cfg = Settings(mock_hardware=True, hailo=HailoConfig(enabled=False))
    assert build_hailo_runtime(cfg) is None


def test_build_hailo_runtime_mock_returns_mock():
    from mousedroid.config.schema import HailoConfig
    from mousedroid.factory import build_hailo_runtime
    from mousedroid.hardware.accelerator.hailo_runtime import MockHailoRuntime

    cfg = Settings(mock_hardware=True, hailo=HailoConfig(enabled=True))
    rt = build_hailo_runtime(cfg)
    assert isinstance(rt, MockHailoRuntime)


# ---------------------------------------------------------------------------
# LiDAR factory tests
# ---------------------------------------------------------------------------


def test_build_lidar_none_config_returns_none():
    """build_lidar returns None when settings.lidar is None."""
    from mousedroid.factory import build_lidar

    cfg = Settings(mock_hardware=True)  # lidar=None by default
    assert build_lidar(cfg) is None


def test_build_lidar_disabled_returns_none():
    """build_lidar returns None when settings.lidar.enabled is False."""
    from mousedroid.config.schema import LidarConfig
    from mousedroid.factory import build_lidar

    cfg = Settings(mock_hardware=True, lidar=LidarConfig(enabled=False))
    assert build_lidar(cfg) is None


def test_build_lidar_mock_hardware():
    """build_lidar returns a non-None driver when mock_hardware and lidar configured."""
    from mousedroid.config.schema import LidarConfig
    from mousedroid.factory import build_lidar

    cfg = Settings(mock_hardware=True, lidar=LidarConfig(enabled=True))
    lidar = build_lidar(cfg)
    assert lidar is not None


def test_build_lidar_feature_extractor_enabled():
    """build_lidar_feature_extractor returns LidarFeatureExtractor when lidar configured."""
    from mousedroid.config.schema import LidarConfig
    from mousedroid.factory import build_lidar_feature_extractor
    from mousedroid.hardware.lidar.feature_extractor import LidarFeatureExtractor

    cfg = Settings(mock_hardware=True, lidar=LidarConfig(enabled=True))
    extractor = build_lidar_feature_extractor(cfg)
    assert isinstance(extractor, LidarFeatureExtractor)


def test_build_lidar_feature_extractor_disabled():
    """build_lidar_feature_extractor returns None when lidar is None."""
    from mousedroid.factory import build_lidar_feature_extractor

    cfg = Settings(mock_hardware=True)  # lidar=None by default
    assert build_lidar_feature_extractor(cfg) is None


# ---------------------------------------------------------------------------
# F-025 — command-set threading through build_esp32_driver
# ---------------------------------------------------------------------------


def test_build_esp32_default_command_set_resolves_legacy_codec() -> None:
    """Default selector: the built driver carries the legacy codec.

    Asserts the default value inline first (the dispatch-test idiom) so a
    silent selector-default drift fails here, not on the rover.
    """
    cfg = _real_settings(esp32={"protocol": "serial"})
    assert cfg.esp32.command_set == "legacy"

    from mousedroid.comms.command_set import LEGACY_CODEC
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(cfg)
    assert driver.inner._codec is LEGACY_CODEC


def test_build_esp32_stock_command_set_resolves_stock_codec() -> None:
    cfg = _real_settings(esp32={"protocol": "serial", "command_set": "waveshare_stock"})

    from mousedroid.comms.command_set import WAVESHARE_STOCK_CODEC
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(cfg)
    assert driver.inner._codec is WAVESHARE_STOCK_CODEC
    # The after-validator also derived the stock baud (no explicit pin).
    assert driver.inner._baud == 115200


@pytest.mark.parametrize("command_set", ["legacy", "waveshare_stock"])
def test_build_esp32_mock_and_disabled_win_over_command_set(command_set: str) -> None:
    """The mock/disabled guard runs BEFORE codec-bearing driver selection.

    Mirrors the enabled-vs-protocol conditional-ordering pin: whatever the
    selector says, a disabled ESP32 must produce the mock driver (which
    builds no command dicts at all), so a bench without the board never
    opens a serial port regardless of firmware flavour.
    """
    cfg = _real_settings(esp32={"protocol": "serial", "enabled": False, "command_set": command_set})

    from mousedroid.comms.mock_driver import MockESP32Driver
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(cfg)
    assert isinstance(driver.inner, MockESP32Driver)
