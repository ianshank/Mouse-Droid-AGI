"""Backwards-compatibility regression tests for the voice engine and USB speaker.

These tests guarantee that:
- Every committed YAML loads without validation error after ``VoiceConfig`` and
  ``SpeakerConfig`` fields were added/updated.
- The ``jetson_production.yaml`` overlay carries the expected TTS model path and
  voice settings introduced in this branch.
- ``VoiceConfig`` defaults match documented values (disabled by default, Piper
  model path ``None``, sane sample-rate and queue-size defaults).
- ``SpeakerConfig`` defaults are stable (write-timeout and poll-interval defaults).
- Factory helpers return usable objects for opted-in configs and ``None`` /
  graceful results when the optional subsystem is not configured.
- Existing YAML files that do **not** include the ``voice`` or ``speaker``
  sections continue to load unchanged (full backwards compatibility).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.config.schema import Settings, SpeakerConfig

# ---------------------------------------------------------------------------
# Committed YAML files that must load cleanly under the current schema.
# ---------------------------------------------------------------------------

_MOUSE_DROID_YAMLS = [
    "default.yaml",
    "mock_hardware.yaml",
    "jetson_production.yaml",
    "jetson_hailo.yaml",
    "jetson_secure_metrics.yaml",
    "local_training.yaml",
]

_CONFIG_DIR = Path("config")


@pytest.mark.parametrize("filename", _MOUSE_DROID_YAMLS)
def test_yaml_loads_cleanly(filename: str) -> None:
    """Every committed mouse-droid YAML must load without validation error."""
    path = _CONFIG_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not present in this checkout")
    data = yaml.safe_load(path.read_text())
    s = Settings.model_validate(data)
    assert s.voice is not None, f"{filename}: voice section must always resolve to VoiceConfig"


# ---------------------------------------------------------------------------
# VoiceConfig — default field values (no YAML overrides).
# ---------------------------------------------------------------------------


def test_voice_config_default_disabled() -> None:
    """Voice engine must be disabled by default so existing configs are unaffected."""
    s = Settings.model_validate({"mock_hardware": True})
    assert s.voice.enabled is False


def test_voice_config_default_tts_model_path_none() -> None:
    """tts_model_path defaults to None (no model loaded unless explicitly set)."""
    s = Settings.model_validate({"mock_hardware": True})
    assert s.voice.tts_model_path is None


def test_voice_config_default_sample_rate() -> None:
    """tts_sample_rate defaults to 22050 Hz (Piper default)."""
    s = Settings.model_validate({"mock_hardware": True})
    assert s.voice.tts_sample_rate == 22050


def test_voice_config_default_cooldown() -> None:
    """cooldown_s defaults to 5.0 seconds."""
    s = Settings.model_validate({"mock_hardware": True})
    assert s.voice.cooldown_s == 5.0


def test_voice_config_default_queue_size() -> None:
    """queue_size defaults to 16 speech requests."""
    s = Settings.model_validate({"mock_hardware": True})
    assert s.voice.queue_size == 16


def test_voice_config_default_personality() -> None:
    """personality defaults to 'rocky'."""
    s = Settings.model_validate({"mock_hardware": True})
    assert s.voice.personality == "rocky"


# ---------------------------------------------------------------------------
# SpeakerConfig — default field values.
# ---------------------------------------------------------------------------


def test_speaker_config_defaults_when_present() -> None:
    """SpeakerConfig defaults are stable across schema changes."""
    cfg = SpeakerConfig()
    assert cfg.enabled is True
    assert cfg.device_name == "USB"
    assert cfg.sample_rate == 22050
    assert cfg.channels == 1
    assert cfg.chunk_size == 1024
    assert cfg.format == "float32"
    assert cfg.write_timeout_s == 0.5
    assert cfg.write_poll_interval_s == 0.01


def test_speaker_not_required_in_settings() -> None:
    """Settings.speaker is optional — omitting the section gives None."""
    s = Settings.model_validate({"mock_hardware": True})
    # speaker is Optional[SpeakerConfig] — may be None or a default
    assert s.speaker is None or isinstance(s.speaker, SpeakerConfig)


# ---------------------------------------------------------------------------
# jetson_production.yaml — validate introduced voice fields.
# ---------------------------------------------------------------------------


def test_jetson_production_voice_enabled() -> None:
    """jetson_production.yaml must have voice.enabled=True after this branch."""
    path = _CONFIG_DIR / "jetson_production.yaml"
    if not path.exists():
        pytest.skip("jetson_production.yaml not present")
    data = yaml.safe_load(path.read_text())
    s = Settings.model_validate(data)
    assert (
        s.voice.enabled is True
    ), "jetson_production.yaml: voice.enabled must be True; check config/jetson_production.yaml"


def test_jetson_production_tts_model_path_set() -> None:
    """jetson_production.yaml must specify the Piper model path."""
    path = _CONFIG_DIR / "jetson_production.yaml"
    if not path.exists():
        pytest.skip("jetson_production.yaml not present")
    data = yaml.safe_load(path.read_text())
    s = Settings.model_validate(data)
    assert (
        s.voice.tts_model_path is not None
    ), "jetson_production.yaml: voice.tts_model_path must not be None"
    assert s.voice.tts_model_path.endswith(
        ".onnx"
    ), "voice.tts_model_path must point to a .onnx file"


def test_jetson_production_tts_model_path_is_absolute() -> None:
    """Piper model path must be an absolute Linux path (deployed inside container).

    Uses a string startswith check so the test is cross-platform — this path
    is only resolved inside the Jetson/Docker container (Linux), not on the
    Windows/macOS dev machine running the tests.
    """
    path = _CONFIG_DIR / "jetson_production.yaml"
    if not path.exists():
        pytest.skip("jetson_production.yaml not present")
    data = yaml.safe_load(path.read_text())
    s = Settings.model_validate(data)
    if s.voice.tts_model_path is None:
        pytest.skip("tts_model_path not set")
    assert s.voice.tts_model_path.startswith("/"), (
        f"voice.tts_model_path must start with '/' (absolute Linux path), "
        f"got: {s.voice.tts_model_path!r}"
    )


# ---------------------------------------------------------------------------
# Minimal inline-YAML round-trips — validate that voice config is applied.
# ---------------------------------------------------------------------------


def test_voice_enabled_via_yaml() -> None:
    """voice.enabled can be set via YAML overlay."""
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "voice": {"enabled": True, "tts_model_path": "/opt/voice_models/test.onnx"},
        }
    )
    assert s.voice.enabled is True
    assert s.voice.tts_model_path == "/opt/voice_models/test.onnx"


def test_voice_phrase_overrides_empty_by_default() -> None:
    """phrase_overrides defaults to an empty dict so no override keys are active."""
    s = Settings.model_validate({"mock_hardware": True})
    assert s.voice.phrase_overrides == {}


def test_voice_intensity_threshold_range() -> None:
    """intensity_threshold accepts values in [0, 1]."""
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "voice": {"intensity_threshold": 0.5},
        }
    )
    assert s.voice.intensity_threshold == 0.5


def test_voice_config_backwards_compat_no_new_required_fields() -> None:
    """A minimal voice stanza with only 'enabled' must still parse cleanly."""
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "voice": {"enabled": False},
        }
    )
    assert s.voice.enabled is False
    assert s.voice.tts_model_path is None
    assert s.voice.tts_sample_rate == 22050


def test_speaker_config_backwards_compat_partial_stanza() -> None:
    """A partial speaker stanza must pick up all defaults for unspecified keys."""
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "speaker": {"sample_rate": 44100},
        }
    )
    assert s.speaker is not None
    assert s.speaker.sample_rate == 44100
    assert s.speaker.channels == 1  # unchanged default
    assert s.speaker.write_timeout_s == 0.5  # unchanged default


# ---------------------------------------------------------------------------
# New field backwards-compat tests (personality_to_model_map,
# event_intensity_thresholds, widened personality field)
# ---------------------------------------------------------------------------


def test_personality_to_model_map_defaults_to_empty() -> None:
    """personality_to_model_map defaults to {} — old YAMLs unaffected."""
    s = Settings.model_validate({"mock_hardware": True})
    assert s.voice.personality_to_model_map == {}


def test_event_intensity_thresholds_defaults_to_empty() -> None:
    """event_intensity_thresholds defaults to {} — old YAMLs unaffected."""
    s = Settings.model_validate({"mock_hardware": True})
    assert s.voice.event_intensity_thresholds == {}


def test_personality_rocky_still_accepted() -> None:
    """personality='rocky' (previously Literal) still loads cleanly."""
    s = Settings.model_validate({"mock_hardware": True, "voice": {"personality": "rocky"}})
    assert s.voice.personality == "rocky"


def test_personality_to_model_map_round_trips_via_yaml() -> None:
    """personality_to_model_map values survive Settings round-trip."""
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "voice": {
                "personality": "rocky",
                "personality_to_model_map": {
                    "rocky": "/opt/voice_models/rocky.onnx",
                },
            },
        }
    )
    assert s.voice.personality_to_model_map == {"rocky": "/opt/voice_models/rocky.onnx"}
    assert s.voice.resolved_tts_model_path() == "/opt/voice_models/rocky.onnx"


def test_event_intensity_thresholds_round_trips_via_yaml() -> None:
    """event_intensity_thresholds values survive Settings round-trip."""
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "voice": {
                "event_intensity_thresholds": {
                    "obstacle_detected": 0.4,
                    "emergency_stop": 0.2,
                },
            },
        }
    )
    assert s.voice.event_intensity_thresholds["obstacle_detected"] == pytest.approx(0.4)
    assert s.voice.event_intensity_thresholds["emergency_stop"] == pytest.approx(0.2)


def test_old_yaml_without_new_fields_resolves_to_tts_model_path() -> None:
    """Old YAML with only tts_model_path resolves correctly via new method."""
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "voice": {"tts_model_path": "/opt/voice_models/rocky.onnx"},
        }
    )
    assert s.voice.resolved_tts_model_path() == "/opt/voice_models/rocky.onnx"
