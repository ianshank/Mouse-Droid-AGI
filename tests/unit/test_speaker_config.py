"""Unit tests for SpeakerConfig and VoiceConfig validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import SpeakerConfig, VoiceConfig


class TestSpeakerConfig:
    """SpeakerConfig validation tests."""

    def test_defaults(self) -> None:
        """Default SpeakerConfig is valid."""
        cfg = SpeakerConfig()
        assert cfg.enabled is True
        assert cfg.sample_rate == 22050
        assert cfg.channels == 1
        assert cfg.chunk_size == 1024
        assert cfg.device_name == "USB"
        assert cfg.format == "float32"

    def test_custom_values(self) -> None:
        """Custom values are accepted."""
        cfg = SpeakerConfig(
            sample_rate=44100,
            channels=2,
            chunk_size=512,
            device_name="Speaker",
            format="int16",
        )
        assert cfg.sample_rate == 44100
        assert cfg.channels == 2
        assert cfg.format == "int16"

    def test_invalid_sample_rate(self) -> None:
        """Zero sample rate is rejected."""
        with pytest.raises(ValidationError):
            SpeakerConfig(sample_rate=0)

    def test_invalid_channels(self) -> None:
        """Channel count > 2 is rejected."""
        with pytest.raises(ValidationError):
            SpeakerConfig(channels=3)

    def test_invalid_format(self) -> None:
        """Unknown format is rejected."""
        with pytest.raises(ValidationError):
            SpeakerConfig(format="float64")  # type: ignore[arg-type]


class TestVoiceConfig:
    """VoiceConfig validation tests."""

    def test_defaults(self) -> None:
        """Default VoiceConfig is valid and disabled."""
        cfg = VoiceConfig()
        assert cfg.enabled is False
        assert cfg.cooldown_s == 5.0
        assert cfg.personality == "rocky"
        assert cfg.tts_model_path is None
        assert cfg.tts_sample_rate == 22050
        assert cfg.queue_size == 16
        assert cfg.phrase_overrides == {}

    def test_phrase_overrides(self) -> None:
        """Custom phrase overrides are stored."""
        overrides = {"startup": ["Hello!", "Hi hi!"]}
        cfg = VoiceConfig(phrase_overrides=overrides)
        assert cfg.phrase_overrides == overrides

    def test_invalid_cooldown(self) -> None:
        """Zero cooldown is rejected."""
        with pytest.raises(ValidationError):
            VoiceConfig(cooldown_s=0)

    def test_invalid_queue_size(self) -> None:
        """Zero queue size is rejected."""
        with pytest.raises(ValidationError):
            VoiceConfig(queue_size=0)
