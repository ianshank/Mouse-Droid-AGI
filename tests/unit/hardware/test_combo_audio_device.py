"""Tests for combo USB audio device (Wonrabai USB Sound Card).

Verifies that both mic and speaker correctly discover and open
separate input/output streams on the same physical USB audio device.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mousedroid.config.schema import MicrophoneConfig, SpeakerConfig
from mousedroid.hardware.audio.usb_microphone import UsbMicrophone
from mousedroid.hardware.audio.usb_speaker import UsbSpeaker


def _wonrabai_device_list() -> list[dict[str, object]]:
    """Simulated PyAudio device list with a Wonrabai combo device."""
    return [
        {
            "name": "Built-in Audio Analog Stereo",
            "maxInputChannels": 2,
            "maxOutputChannels": 2,
        },
        {
            "name": "USB Audio Device",
            "maxInputChannels": 1,
            "maxOutputChannels": 2,
        },
        {
            "name": "HDMI Audio Output",
            "maxInputChannels": 0,
            "maxOutputChannels": 8,
        },
    ]


def _make_mock_pyaudio(devices: list[dict[str, object]]) -> MagicMock:
    """Create a mock pyaudio module with the given device list.

    Each call to ``PyAudio()`` returns a fresh mock instance so that
    ``_find_device_index`` (which creates its own PyAudio instance)
    and ``start()`` both work independently.
    """

    def _new_pa_instance() -> MagicMock:
        inst = MagicMock()
        inst.get_device_count.return_value = len(devices)
        inst.get_device_info_by_index.side_effect = lambda i: devices[i]
        inst.open.return_value = MagicMock()
        return inst

    mock_pyaudio = MagicMock()
    mock_pyaudio.PyAudio = MagicMock(side_effect=lambda: _new_pa_instance())
    mock_pyaudio.paFloat32 = 1
    mock_pyaudio.paInt16 = 8
    return mock_pyaudio


def test_mic_and_speaker_find_same_wonrabai_device() -> None:
    """Both mic and speaker discover the same USB Audio Device by name."""
    devices = _wonrabai_device_list()
    mock_pyaudio = _make_mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        mic = UsbMicrophone(MicrophoneConfig(device_name="USB"))
        speaker = UsbSpeaker(SpeakerConfig(device_name="USB"))

        mic_idx = mic._find_device_index()
        speaker_idx = speaker._find_device_index()

        # Both find device index 1 ("USB Audio Device")
        assert mic_idx == 1
        assert speaker_idx == 1


def test_mic_ignores_output_only_device() -> None:
    """Mic skips devices with maxInputChannels=0."""
    devices = [
        {
            "name": "USB Audio Device",
            "maxInputChannels": 0,
            "maxOutputChannels": 2,
        },
    ]
    mock_pyaudio = _make_mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        mic = UsbMicrophone(MicrophoneConfig(device_name="USB"))
        assert mic._find_device_index() is None


def test_speaker_ignores_input_only_device() -> None:
    """Speaker skips devices with maxOutputChannels=0."""
    devices = [
        {
            "name": "USB Audio Device",
            "maxInputChannels": 2,
            "maxOutputChannels": 0,
        },
    ]
    mock_pyaudio = _make_mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        speaker = UsbSpeaker(SpeakerConfig(device_name="USB"))
        assert speaker._find_device_index() is None


async def test_mic_opens_input_stream_on_combo_device() -> None:
    """Mic starts successfully and has a stream on the combo device."""
    devices = _wonrabai_device_list()
    mock_pyaudio = _make_mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        mic = UsbMicrophone(MicrophoneConfig(device_name="USB"))
        await mic.start()

        # Stream was opened (not None)
        assert mic._stream is not None

        await mic.stop()
        assert mic._stream is None


async def test_speaker_opens_output_stream_on_combo_device() -> None:
    """Speaker starts successfully and has a stream on the combo device."""
    devices = _wonrabai_device_list()
    mock_pyaudio = _make_mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        speaker = UsbSpeaker(SpeakerConfig(device_name="USB"))
        await speaker.start()

        # Stream was opened (not None)
        assert speaker._stream is not None

        await speaker.stop()
        assert speaker._stream is None


def test_device_name_case_insensitive() -> None:
    """Device name matching is case-insensitive."""
    devices = [
        {
            "name": "usb audio device",
            "maxInputChannels": 1,
            "maxOutputChannels": 1,
        },
    ]
    mock_pyaudio = _make_mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        mic = UsbMicrophone(MicrophoneConfig(device_name="USB"))
        assert mic._find_device_index() == 0

        speaker = UsbSpeaker(SpeakerConfig(device_name="USB"))
        assert speaker._find_device_index() == 0
