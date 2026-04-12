from __future__ import annotations

from unittest.mock import MagicMock

from mousedroid.config.schema import MicrophoneConfig, Settings
from mousedroid.factory import build_microphone
from mousedroid.hardware.audio.mock_microphone import MockMicrophone


def test_build_microphone_none_when_disabled():
    cfg = Settings(mock_hardware=True)
    result = build_microphone(cfg)
    assert result is None


def test_build_microphone_mock():
    cfg = Settings(mock_hardware=True, microphone=MicrophoneConfig())
    result = build_microphone(cfg)
    assert isinstance(result, MockMicrophone)


def test_build_microphone_mock_preserves_config():
    mic_cfg = MicrophoneConfig(sample_rate=44100, channels=2)
    cfg = Settings(mock_hardware=True, microphone=mic_cfg)
    result = build_microphone(cfg)
    assert result is not None
    assert result.sample_rate == 44100
    assert result.channels == 2


def test_build_microphone_disabled_via_flag():
    """Microphone config present but ``enabled=False`` returns None."""
    cfg = Settings(
        mock_hardware=True,
        microphone=MicrophoneConfig(enabled=False),
    )
    result = build_microphone(cfg)
    assert result is None


def test_build_microphone_real_hardware(monkeypatch):
    cfg = Settings(
        mock_hardware=False,
        microphone=MicrophoneConfig(),
        ultrasonic={"trigger_pin": 23, "echo_pin": 24},
    )
    mock_usb_mic = MagicMock()
    mock_module = MagicMock()
    mock_module.UsbMicrophone = MagicMock(return_value=mock_usb_mic)
    monkeypatch.setitem(
        __import__("sys").modules,
        "mousedroid.hardware.audio.usb_microphone",
        mock_module,
    )
    # Re-import to pick up monkeypatched module
    import importlib

    import mousedroid.factory

    importlib.reload(mousedroid.factory)
    from mousedroid.factory import build_microphone as build_mic_reloaded

    result = build_mic_reloaded(cfg)
    assert result is mock_usb_mic
