from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import MicrophoneConfig, Settings


def test_default_values():
    cfg = MicrophoneConfig()
    assert cfg.enabled is True
    assert cfg.sample_rate == 16000
    assert cfg.channels == 1
    assert cfg.chunk_size == 1024
    assert cfg.device_name == "USB"
    assert cfg.device_index is None
    assert cfg.format == "float32"
    assert cfg.n_mels == 64
    assert cfg.n_fft == 512
    assert cfg.hop_length == 256


def test_custom_values():
    cfg = MicrophoneConfig(
        enabled=False,
        device_index=2,
        device_name="TestMic",
        sample_rate=44100,
        channels=2,
        chunk_size=2048,
        format="int16",
        n_mels=128,
        n_fft=1024,
        hop_length=512,
    )
    assert cfg.enabled is False
    assert cfg.device_index == 2
    assert cfg.device_name == "TestMic"
    assert cfg.sample_rate == 44100
    assert cfg.channels == 2
    assert cfg.chunk_size == 2048
    assert cfg.format == "int16"
    assert cfg.n_mels == 128
    assert cfg.n_fft == 1024
    assert cfg.hop_length == 512


def test_channels_max_2():
    with pytest.raises(ValidationError):
        MicrophoneConfig(channels=3)


def test_channels_min_1():
    with pytest.raises(ValidationError):
        MicrophoneConfig(channels=0)


def test_sample_rate_positive():
    with pytest.raises(ValidationError):
        MicrophoneConfig(sample_rate=0)


def test_chunk_size_positive():
    with pytest.raises(ValidationError):
        MicrophoneConfig(chunk_size=0)


def test_settings_without_microphone():
    settings = Settings(mock_hardware=True)
    assert settings.microphone is None


def test_settings_with_microphone():
    settings = Settings(
        mock_hardware=True,
        microphone=MicrophoneConfig(),
    )
    assert settings.microphone is not None
    assert settings.microphone.sample_rate == 16000


def test_settings_microphone_from_dict():
    settings = Settings(
        mock_hardware=True,
        microphone={"sample_rate": 44100, "channels": 2},
    )
    assert settings.microphone is not None
    assert settings.microphone.sample_rate == 44100
    assert settings.microphone.channels == 2


def test_enabled_defaults_true():
    cfg = MicrophoneConfig()
    assert cfg.enabled is True


def test_backwards_compat_no_enabled_field():
    """Existing configs without ``enabled`` still parse (defaults to True)."""
    settings = Settings(
        mock_hardware=True,
        microphone={"sample_rate": 16000},
    )
    assert settings.microphone is not None
    assert settings.microphone.enabled is True


def test_disabled_microphone_from_dict():
    settings = Settings(
        mock_hardware=True,
        microphone={"enabled": False},
    )
    assert settings.microphone is not None
    assert settings.microphone.enabled is False


def test_n_mels_positive():
    with pytest.raises(ValidationError):
        MicrophoneConfig(n_mels=0)


def test_n_fft_positive():
    with pytest.raises(ValidationError):
        MicrophoneConfig(n_fft=0)


def test_hop_length_positive():
    with pytest.raises(ValidationError):
        MicrophoneConfig(hop_length=0)
