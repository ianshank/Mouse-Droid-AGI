"""Tests for the real UsbMicrophone driver (mocked PyAudio)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from mousedroid.config.schema import MicrophoneConfig
from mousedroid.hardware.audio.usb_microphone import UsbMicrophone


def _default_cfg() -> MicrophoneConfig:
    return MicrophoneConfig()


def test_sample_rate():
    mic = UsbMicrophone(_default_cfg())
    assert mic.sample_rate == 16000


def test_channels():
    mic = UsbMicrophone(_default_cfg())
    assert mic.channels == 1


def test_chunk_size():
    mic = UsbMicrophone(_default_cfg())
    assert mic.chunk_size == 1024


def test_custom_config():
    cfg = MicrophoneConfig(sample_rate=44100, channels=2, chunk_size=2048)
    mic = UsbMicrophone(cfg)
    assert mic.sample_rate == 44100
    assert mic.channels == 2
    assert mic.chunk_size == 2048


async def test_read_chunk_returns_silence_when_not_started():
    """Reading before start returns silence (graceful degradation)."""
    mic = UsbMicrophone(_default_cfg())
    chunk = await mic.read_chunk()
    assert chunk.dtype == np.float32
    assert chunk.shape == (1024,)
    assert np.all(chunk == 0.0)


async def test_start_device_not_found():
    """When device auto-detect finds nothing, start still succeeds with warning."""
    mock_pa_instance = MagicMock()
    mock_pa_instance.get_device_count.return_value = 2
    mock_pa_instance.get_device_info_by_index.side_effect = [
        {"name": "Built-in Audio", "maxInputChannels": 2},
        {"name": "HDMI Output", "maxInputChannels": 0},
    ]
    mock_stream = MagicMock()
    mock_pa_instance.open.return_value = mock_stream

    mock_pa_class = MagicMock(return_value=mock_pa_instance)
    mock_pyaudio = MagicMock()
    mock_pyaudio.PyAudio = mock_pa_class
    mock_pyaudio.paFloat32 = 1
    mock_pyaudio.paInt16 = 8

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        cfg = MicrophoneConfig(device_name="NonExistentSuziePi")
        mic = UsbMicrophone(cfg)
        await mic.start()
        # open was called with device_index=None since no device matched
        mock_pa_instance.open.assert_called_once()
        call_kwargs = mock_pa_instance.open.call_args
        assert call_kwargs[1]["input_device_index"] is None


async def test_start_stop_lifecycle():
    """Start and stop correctly manage PyAudio resources."""
    mock_pa_instance = MagicMock()
    mock_stream = MagicMock()
    mock_pa_instance.open.return_value = mock_stream
    mock_pa_instance.get_device_count.return_value = 0

    mock_pa_class = MagicMock(return_value=mock_pa_instance)
    mock_pyaudio = MagicMock()
    mock_pyaudio.PyAudio = mock_pa_class
    mock_pyaudio.paFloat32 = 1

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        mic = UsbMicrophone(_default_cfg())
        await mic.start()
        await mic.stop()
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        # terminate is called by both _find_device_index and stop, so at least once
        assert mock_pa_instance.terminate.call_count >= 1


async def test_read_chunk_float32():
    """Read chunk converts float32 bytes correctly."""
    mock_pa_instance = MagicMock()
    raw_data = np.array([0.1, 0.5, -0.3], dtype=np.float32).tobytes()
    mock_stream = MagicMock()
    mock_stream.read.return_value = raw_data
    mock_pa_instance.open.return_value = mock_stream
    mock_pa_instance.get_device_count.return_value = 0

    mock_pa_class = MagicMock(return_value=mock_pa_instance)
    mock_pyaudio = MagicMock()
    mock_pyaudio.PyAudio = mock_pa_class
    mock_pyaudio.paFloat32 = 1

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        cfg = MicrophoneConfig(chunk_size=3)
        mic = UsbMicrophone(cfg)
        await mic.start()
        chunk = await mic.read_chunk()
        np.testing.assert_allclose(chunk, [0.1, 0.5, -0.3], atol=1e-6)


async def test_read_chunk_int16():
    """Read chunk converts int16 bytes to normalised float32."""
    mock_pa_instance = MagicMock()
    raw_data = np.array([16384, -16384], dtype=np.int16).tobytes()
    mock_stream = MagicMock()
    mock_stream.read.return_value = raw_data
    mock_pa_instance.open.return_value = mock_stream
    mock_pa_instance.get_device_count.return_value = 0

    mock_pa_class = MagicMock(return_value=mock_pa_instance)
    mock_pyaudio = MagicMock()
    mock_pyaudio.PyAudio = mock_pa_class
    mock_pyaudio.paInt16 = 8

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        cfg = MicrophoneConfig(chunk_size=2, format="int16")
        mic = UsbMicrophone(cfg)
        await mic.start()
        chunk = await mic.read_chunk()
        assert chunk.dtype == np.float32
        np.testing.assert_allclose(chunk, [0.5, -0.5], atol=1e-4)


async def test_start_handles_missing_pyaudio():
    """Start gracefully handles missing pyaudio (import failure)."""
    with patch.dict("sys.modules", {"pyaudio": None}):
        mic = UsbMicrophone(_default_cfg())
        await mic.start()
        # Stream should be None — no crash
        assert mic._stream is None
        # read_chunk returns silence
        chunk = await mic.read_chunk()
        assert np.all(chunk == 0.0)


async def test_start_handles_oserror():
    """Start gracefully handles ALSA/PyAudio OSError."""
    mock_pa_instance = MagicMock()
    mock_pa_instance.get_device_count.return_value = 0
    mock_pa_instance.open.side_effect = OSError("ALSA device busy")

    mock_pa_class = MagicMock(return_value=mock_pa_instance)
    mock_pyaudio = MagicMock()
    mock_pyaudio.PyAudio = mock_pa_class
    mock_pyaudio.paFloat32 = 1

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        mic = UsbMicrophone(_default_cfg())
        await mic.start()
        # Stream should be None — graceful degradation
        assert mic._stream is None
        assert mic._pa is None
        # read_chunk returns silence, not an exception
        chunk = await mic.read_chunk()
        assert chunk.dtype == np.float32
        assert np.all(chunk == 0.0)


async def test_read_chunk_handles_device_disconnect():
    """Read chunk catches OSError from unplugged device, returns silence, disables stream."""
    mock_pa_instance = MagicMock()
    mock_stream = MagicMock()
    mock_stream.read.side_effect = OSError("Device disconnected")
    mock_pa_instance.open.return_value = mock_stream
    mock_pa_instance.get_device_count.return_value = 0

    mock_pa_class = MagicMock(return_value=mock_pa_instance)
    mock_pyaudio = MagicMock()
    mock_pyaudio.PyAudio = mock_pa_class
    mock_pyaudio.paFloat32 = 1

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        mic = UsbMicrophone(_default_cfg())
        await mic.start()
        assert mic._stream is not None

        # First read hits OSError — returns silence, disables stream
        chunk = await mic.read_chunk()
        assert chunk.dtype == np.float32
        assert chunk.shape == (1024,)
        assert np.all(chunk == 0.0)
        assert mic._stream is None

        # Subsequent reads return silence without attempting I/O
        chunk2 = await mic.read_chunk()
        assert np.all(chunk2 == 0.0)
        # stream.read was only called once (the failed attempt)
        assert mock_stream.read.call_count == 1


async def test_read_chunk_stereo_silence_shape():
    """Silence for stereo mic has correct shape."""
    cfg = MicrophoneConfig(channels=2, chunk_size=512)
    mic = UsbMicrophone(cfg)
    chunk = await mic.read_chunk()
    assert chunk.shape == (1024,)  # 512 * 2 channels
    assert np.all(chunk == 0.0)
