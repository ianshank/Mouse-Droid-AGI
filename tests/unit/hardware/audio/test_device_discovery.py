"""Unit tests for the shared PyAudio device-discovery helper.

Drives :func:`find_pyaudio_device_index` directly with a fake ``pyaudio``
module (patched into ``sys.modules``) so no audio hardware is required. These
tests PIN the microphone (input) and speaker (output) discovery paths in sync:
the helper is the single implementation both ``UsbMicrophone`` and
``UsbSpeaker`` delegate to, so exercising both directions here prevents the
two drivers from drifting apart again.
"""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import structlog.testing

from mousedroid.hardware.audio._device_discovery import find_pyaudio_device_index


def _combo_device_list() -> list[dict[str, object]]:
    """Simulated PyAudio device list with input-only, combo, and output-only entries."""
    return [
        {"name": "Built-in Input", "maxInputChannels": 2, "maxOutputChannels": 0},
        {"name": "USB Audio Device", "maxInputChannels": 1, "maxOutputChannels": 2},
        {"name": "HDMI Output", "maxInputChannels": 0, "maxOutputChannels": 8},
    ]


def _mock_pyaudio(devices: Sequence[object]) -> MagicMock:
    """Build a mock ``pyaudio`` module whose ``PyAudio()`` enumerates *devices*."""
    pa_instance = MagicMock()
    pa_instance.get_device_count.return_value = len(devices)
    pa_instance.get_device_info_by_index.side_effect = lambda i: devices[i]

    mock_module = MagicMock()
    mock_module.PyAudio.return_value = pa_instance
    return mock_module


def test_input_path_finds_input_device() -> None:
    """want_input=True returns the first name-matching input-capable device."""
    devices = _combo_device_list()
    mock_pyaudio = _mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        idx = find_pyaudio_device_index("USB", want_input=True, log_event="usb_microphone_found")

    assert idx == 1


def test_output_path_finds_output_device() -> None:
    """want_input=False returns the first name-matching output-capable device."""
    devices = _combo_device_list()
    mock_pyaudio = _mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        idx = find_pyaudio_device_index("USB", want_input=False, log_event="usb_speaker_found")

    assert idx == 1


def test_input_path_ignores_output_only_device() -> None:
    """An output-only device is skipped when an input device is requested."""
    devices = [{"name": "USB Audio Device", "maxInputChannels": 0, "maxOutputChannels": 2}]
    mock_pyaudio = _mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        idx = find_pyaudio_device_index("USB", want_input=True, log_event="usb_microphone_found")

    assert idx is None


def test_output_path_ignores_input_only_device() -> None:
    """An input-only device is skipped when an output device is requested."""
    devices = [{"name": "USB Audio Device", "maxInputChannels": 2, "maxOutputChannels": 0}]
    mock_pyaudio = _mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        idx = find_pyaudio_device_index("USB", want_input=False, log_event="usb_speaker_found")

    assert idx is None


def test_returns_none_when_no_name_match() -> None:
    """No device whose name contains the substring returns None."""
    devices = [{"name": "HDMI Output", "maxInputChannels": 2, "maxOutputChannels": 2}]
    mock_pyaudio = _mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        idx = find_pyaudio_device_index("USB", want_input=True, log_event="usb_microphone_found")

    assert idx is None


def test_name_match_is_case_insensitive() -> None:
    """Substring matching against the device name is case-insensitive."""
    devices = [{"name": "usb audio device", "maxInputChannels": 1, "maxOutputChannels": 1}]
    mock_pyaudio = _mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        idx = find_pyaudio_device_index("USB", want_input=True, log_event="usb_microphone_found")

    assert idx == 0


def test_logs_the_supplied_event_on_match() -> None:
    """The supplied log_event is emitted with index and name on a successful match."""
    devices = _combo_device_list()
    mock_pyaudio = _mock_pyaudio(devices)

    with (
        patch.dict("sys.modules", {"pyaudio": mock_pyaudio}),
        structlog.testing.capture_logs() as logs,
    ):
        find_pyaudio_device_index("USB", want_input=True, log_event="usb_microphone_found")

    matches = [e for e in logs if e.get("event") == "usb_microphone_found"]
    assert len(matches) == 1
    assert matches[0]["index"] == 1
    assert matches[0]["name"] == "USB Audio Device"


def test_skips_none_device_info() -> None:
    """A None device-info row (e.g. mid-enumeration disconnect) is skipped, not crashed on."""
    devices: list[object] = [
        None,
        {"name": "USB Audio Device", "maxInputChannels": 2, "maxOutputChannels": 0},
    ]
    mock_pyaudio = _mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        idx = find_pyaudio_device_index("USB", want_input=True, log_event="usb_microphone_found")

    assert idx == 1


def test_handles_none_channel_value_as_zero() -> None:
    """A present-but-None channel count is treated as 0 (no TypeError from int(None))."""
    devices: list[object] = [
        {"name": "USB Audio Device", "maxInputChannels": None, "maxOutputChannels": None},
    ]
    mock_pyaudio = _mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        idx = find_pyaudio_device_index("USB", want_input=True, log_event="usb_microphone_found")

    assert idx is None


def test_terminates_pyaudio_instance() -> None:
    """The PyAudio instance is always terminated, even on a successful match."""
    devices = _combo_device_list()
    mock_pyaudio = _mock_pyaudio(devices)

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        find_pyaudio_device_index("USB", want_input=True, log_event="usb_microphone_found")

    mock_pyaudio.PyAudio.return_value.terminate.assert_called_once()
