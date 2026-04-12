"""Unit tests for UsbSpeaker — mocked PyAudio."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mousedroid.config.schema import SpeakerConfig


def _cfg(**overrides: object) -> SpeakerConfig:
    return SpeakerConfig(**overrides)  # type: ignore[arg-type]


def test_init_stores_config() -> None:
    """UsbSpeaker stores config values."""
    from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

    cfg = _cfg(sample_rate=44100, channels=2, chunk_size=512)
    speaker = UsbSpeaker(cfg)
    assert speaker.sample_rate == 44100
    assert speaker.channels == 2
    assert speaker.chunk_size == 512


def _mock_pyaudio() -> MagicMock:
    """Create a mock pyaudio module."""
    mock_module = MagicMock()
    mock_module.paFloat32 = 1
    mock_module.paInt16 = 8
    return mock_module


def test_find_device_index_matches_output_device() -> None:
    """_find_device_index finds output device by name substring."""
    mock_pyaudio = _mock_pyaudio()
    mock_pa = MagicMock()
    mock_pa.get_device_count.return_value = 3
    mock_pa.get_device_info_by_index.side_effect = [
        {"name": "Built-in Input", "maxOutputChannels": 0},
        {"name": "USB Audio Device", "maxOutputChannels": 2},
        {"name": "HDMI Output", "maxOutputChannels": 8},
    ]
    mock_pyaudio.PyAudio.return_value = mock_pa

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

        speaker = UsbSpeaker(_cfg(device_name="USB"))
        idx = speaker._find_device_index()

    assert idx == 1


def test_find_device_index_not_found() -> None:
    """_find_device_index returns None when no matching device found."""
    mock_pyaudio = _mock_pyaudio()
    mock_pa = MagicMock()
    mock_pa.get_device_count.return_value = 1
    mock_pa.get_device_info_by_index.return_value = {
        "name": "HDMI",
        "maxOutputChannels": 2,
    }
    mock_pyaudio.PyAudio.return_value = mock_pa

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

        speaker = UsbSpeaker(_cfg(device_name="USB"))
        idx = speaker._find_device_index()

    assert idx is None


@pytest.mark.asyncio
async def test_start_opens_output_stream() -> None:
    """start() opens a PyAudio output stream."""
    mock_pyaudio = _mock_pyaudio()
    mock_pa = MagicMock()
    mock_pyaudio.PyAudio.return_value = mock_pa

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

        speaker = UsbSpeaker(_cfg(device_index=0))
        await speaker.start()

    mock_pa.open.assert_called_once()
    call_kwargs = mock_pa.open.call_args
    assert call_kwargs.kwargs.get("output") is True or call_kwargs[1].get("output") is True


@pytest.mark.asyncio
async def test_stop_closes_stream() -> None:
    """stop() closes the stream and terminates PyAudio."""
    from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

    speaker = UsbSpeaker(_cfg())
    mock_stream = MagicMock()
    mock_pa = MagicMock()
    speaker._stream = mock_stream
    speaker._pa = mock_pa

    await speaker.stop()

    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()
    mock_pa.terminate.assert_called_once()
    assert speaker._stream is None
    assert speaker._pa is None


@pytest.mark.asyncio
async def test_stop_when_not_started() -> None:
    """stop() handles None stream gracefully."""
    from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

    speaker = UsbSpeaker(_cfg())
    await speaker.stop()  # Should not raise


@pytest.mark.asyncio
async def test_write_chunk_not_started_is_noop() -> None:
    """write_chunk() is a silent no-op when stream not started."""
    from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

    speaker = UsbSpeaker(_cfg())
    await speaker.write_chunk(np.zeros(1024, dtype=np.float32))  # Should not raise


@pytest.mark.asyncio
async def test_write_chunk_float32_format() -> None:
    """write_chunk() writes float32 data directly."""
    from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

    speaker = UsbSpeaker(_cfg(format="float32"))
    mock_stream = MagicMock()
    speaker._stream = mock_stream

    samples = np.ones(1024, dtype=np.float32)
    await speaker.write_chunk(samples)

    mock_stream.write.assert_called_once()


@pytest.mark.asyncio
async def test_write_chunk_int16_format() -> None:
    """write_chunk() converts float32 to int16 when format is int16."""
    from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

    speaker = UsbSpeaker(_cfg(format="int16"))
    mock_stream = MagicMock()
    speaker._stream = mock_stream

    samples = np.array([0.5, -0.5], dtype=np.float32)
    await speaker.write_chunk(samples)

    mock_stream.write.assert_called_once()
    raw = mock_stream.write.call_args[0][0]
    decoded = np.frombuffer(raw, dtype=np.int16)
    assert decoded[0] == 16384  # 0.5 * 32768
    assert decoded[1] == -16384


@pytest.mark.asyncio
async def test_write_chunk_int16_clamps_overflow() -> None:
    """write_chunk() clamps values >= 1.0 to prevent int16 overflow."""
    from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

    speaker = UsbSpeaker(_cfg(format="int16"))
    mock_stream = MagicMock()
    speaker._stream = mock_stream

    samples = np.array([1.0, -1.0, 1.5, -1.5], dtype=np.float32)
    await speaker.write_chunk(samples)

    raw = mock_stream.write.call_args[0][0]
    decoded = np.frombuffer(raw, dtype=np.int16)
    # 1.0 and 1.5 clamped to just below 1.0 -> 32767
    assert decoded[0] == 32767
    assert decoded[2] == 32767
    # -1.0 and -1.5 clamped to -1.0 -> -32768
    assert decoded[1] == -32768
    assert decoded[3] == -32768


@pytest.mark.asyncio
async def test_start_handles_missing_pyaudio() -> None:
    """start() is a no-op when pyaudio is not importable."""
    from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

    speaker = UsbSpeaker(_cfg())
    with patch.dict("sys.modules", {"pyaudio": None}):
        await speaker.start()

    assert speaker._stream is None
    assert speaker._pa is None


@pytest.mark.asyncio
async def test_start_handles_oserror() -> None:
    """start() leaves speaker disabled on OSError from ALSA."""
    mock_pyaudio = _mock_pyaudio()
    mock_pa = MagicMock()
    mock_pa.open.side_effect = OSError("No ALSA devices")
    mock_pyaudio.PyAudio.return_value = mock_pa

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

        speaker = UsbSpeaker(_cfg(device_index=0))
        await speaker.start()

    assert speaker._stream is None
    mock_pa.terminate.assert_called_once()
