"""Unit tests for PiperTTS — mocked piper imports."""

from __future__ import annotations

import io
import struct
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mousedroid.config.schema import VoiceConfig


def _cfg(**overrides: object) -> VoiceConfig:
    return VoiceConfig(enabled=True, **overrides)  # type: ignore[arg-type]


def test_init_logs_config() -> None:
    """PiperTTS logs model path and sample rate on init."""
    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg(tts_model_path="/tmp/voice.onnx"))
    assert tts._cfg.tts_model_path == "/tmp/voice.onnx"


def test_start_loads_model_when_path_provided() -> None:
    """start() loads the piper voice model from path."""
    mock_voice = MagicMock()
    mock_piper_module = MagicMock()
    mock_piper_module.PiperVoice.load.return_value = mock_voice

    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg(tts_model_path="/tmp/model.onnx"))
    with patch.dict("sys.modules", {"piper": mock_piper_module}):
        tts.start()

    mock_piper_module.PiperVoice.load.assert_called_once_with("/tmp/model.onnx")
    assert tts._voice is mock_voice


def test_start_warns_when_no_model_path() -> None:
    """start() logs warning when no model path configured."""
    mock_piper_module = MagicMock()

    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg(tts_model_path=None))
    with patch.dict("sys.modules", {"piper": mock_piper_module}):
        tts.start()

    assert tts._voice is None


def test_start_handles_missing_piper() -> None:
    """start() handles ImportError when piper is not installed."""
    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg())
    # piper is not installed in test env, so start() should handle gracefully
    tts.start()
    assert tts._voice is None


def test_stop_clears_voice() -> None:
    """stop() releases the voice model reference."""
    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg())
    tts._voice = MagicMock()  # Pretend a model was loaded
    tts.stop()
    assert tts._voice is None


@pytest.mark.asyncio
async def test_synthesize_no_voice_returns_silence() -> None:
    """synthesize() returns silence when no model is loaded."""
    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg(tts_sample_rate=16000))
    samples = await tts.synthesize("Hello")
    assert samples.dtype == np.float32
    assert len(samples) == 16000
    assert samples.sum() == 0.0


@pytest.mark.asyncio
async def test_synthesize_empty_text_returns_silence() -> None:
    """synthesize() with empty text returns silence when model not loaded."""
    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg(tts_sample_rate=22050))
    samples = await tts.synthesize("")
    assert len(samples) == 22050


@pytest.mark.asyncio
async def test_synthesize_runs_in_thread() -> None:
    """synthesize() offloads blocking work to a thread."""
    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg())
    # Without a loaded model, _synthesize_sync returns silence
    samples = await tts.synthesize("test")
    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float32


def _make_wav_bytes(n_samples: int, sample_rate: int = 22050) -> bytes:
    """Generate a valid WAV byte buffer with int16 samples."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Write a simple sine-ish pattern as int16
        data = b"".join(struct.pack("<h", i % 1000) for i in range(n_samples))
        wf.writeframes(data)
    return buf.getvalue()


def test_synthesize_sync_with_loaded_voice_int16() -> None:
    """_synthesize_sync decodes int16 WAV written by Piper's binary-buffer API."""
    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg(tts_sample_rate=22050))
    n_samples = 100
    wav_bytes = _make_wav_bytes(n_samples)

    mock_voice = MagicMock()

    def write_wav_buffer(text: str, buf: io.BytesIO) -> None:
        assert text == "hello"
        assert hasattr(buf, "write")
        buf.write(wav_bytes)

    mock_voice.synthesize_wav = MagicMock(side_effect=write_wav_buffer)
    mock_voice.synthesize = MagicMock()
    tts._voice = mock_voice

    samples = tts._synthesize_sync("hello")
    assert samples.dtype == np.float32
    assert len(samples) == n_samples
    # int16 values should be normalised to [-1, 1]
    assert np.all(np.abs(samples) <= 1.0)
    mock_voice.synthesize_wav.assert_called_once()
    mock_voice.synthesize.assert_not_called()


def test_synthesize_sync_legacy_synthesize_only() -> None:
    """_synthesize_sync falls back to legacy synthesize(text, wav_file) when
    synthesize_wav is absent (piper-tts <1.3)."""
    from types import SimpleNamespace

    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg(tts_sample_rate=22050))
    n_samples = 50
    wav_bytes = _make_wav_bytes(n_samples)

    def write_wav(text: str, wav_file: wave.Wave_write) -> None:
        src = io.BytesIO(wav_bytes)
        with wave.open(src, "rb") as rf:
            wav_file.setnchannels(rf.getnchannels())
            wav_file.setsampwidth(rf.getsampwidth())
            wav_file.setframerate(rf.getframerate())
            wav_file.writeframes(rf.readframes(rf.getnframes()))

    # SimpleNamespace lacks synthesize_wav, so getattr(..., None) returns None
    tts._voice = SimpleNamespace(synthesize=write_wav)

    samples = tts._synthesize_sync("hi")
    assert samples.dtype == np.float32
    assert len(samples) == n_samples


def test_start_handles_generic_exception() -> None:
    """start() catches non-ImportError exceptions from piper loading."""
    from mousedroid.voice.tts import PiperTTS

    mock_piper_module = MagicMock()
    mock_piper_module.PiperVoice.load.side_effect = OSError("model corrupt")

    tts = PiperTTS(_cfg(tts_model_path="/tmp/corrupt.onnx"))
    with patch.dict("sys.modules", {"piper": mock_piper_module}):
        tts.start()  # Should not raise

    assert tts._voice is None
