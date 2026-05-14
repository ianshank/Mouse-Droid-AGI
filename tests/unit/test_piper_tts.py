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
    """_synthesize_sync decodes int16 WAV returned by Piper's modern API."""
    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg(tts_sample_rate=22050))
    n_samples = 100
    wav_bytes = _make_wav_bytes(n_samples)

    mock_voice = MagicMock()
    mock_voice.synthesize_wav = MagicMock(return_value=wav_bytes)
    mock_voice.synthesize = MagicMock()
    tts._voice = mock_voice
    tts._use_wav_api = True  # simulate detection of the modern API in start()

    samples = tts._synthesize_sync("hello")
    assert samples.dtype == np.float32
    assert len(samples) == n_samples
    # int16 values should be normalised to [-1, 1]
    assert np.all(np.abs(samples) <= 1.0)
    mock_voice.synthesize_wav.assert_called_once_with("hello")
    mock_voice.synthesize.assert_not_called()


def test_synthesize_sync_legacy_synthesize_only() -> None:
    """_synthesize_sync falls back to legacy synthesize(text, wav_file) when
    synthesize_wav is absent (piper-tts <1.3)."""
    from types import SimpleNamespace

    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg(tts_sample_rate=22050))
    n_samples = 50
    raw_pcm = np.arange(n_samples, dtype=np.int16).tobytes()

    def write_wav(text: str, wav_file: wave.Wave_write) -> None:
        assert text == "hi"
        wav_file.writeframes(raw_pcm)

    # SimpleNamespace lacks synthesize_wav, so getattr(..., None) returns None
    tts._voice = SimpleNamespace(synthesize=write_wav)

    samples = tts._synthesize_sync("hi")
    assert samples.dtype == np.float32
    assert len(samples) == n_samples
    assert np.isclose(samples[1], 1.0 / 32768.0)


def test_start_handles_generic_exception() -> None:
    """start() catches non-ImportError exceptions from piper loading."""
    from mousedroid.voice.tts import PiperTTS

    mock_piper_module = MagicMock()
    mock_piper_module.PiperVoice.load.side_effect = OSError("model corrupt")

    tts = PiperTTS(_cfg(tts_model_path="/tmp/corrupt.onnx"))
    with patch.dict("sys.modules", {"piper": mock_piper_module}):
        tts.start()  # Should not raise

    assert tts._voice is None


def test_start_uses_resolved_path_from_personality_map() -> None:
    """start() calls PiperVoice.load with the personality-map path, not tts_model_path."""
    mock_voice = MagicMock()
    mock_piper_module = MagicMock()
    mock_piper_module.PiperVoice.load.return_value = mock_voice

    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(
        _cfg(
            personality="rocky",
            tts_model_path="/models/default.onnx",
            personality_to_model_map={"rocky": "/models/rocky_hd.onnx"},
        )
    )
    with patch.dict("sys.modules", {"piper": mock_piper_module}):
        tts.start()

    # Must load the map-resolved path, NOT tts_model_path
    mock_piper_module.PiperVoice.load.assert_called_once_with("/models/rocky_hd.onnx")
    assert tts._voice is mock_voice


def test_start_falls_back_to_tts_model_path_when_map_empty() -> None:
    """start() uses tts_model_path when personality_to_model_map is empty."""
    mock_voice = MagicMock()
    mock_piper_module = MagicMock()
    mock_piper_module.PiperVoice.load.return_value = mock_voice

    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg(tts_model_path="/models/default.onnx"))
    with patch.dict("sys.modules", {"piper": mock_piper_module}):
        tts.start()

    mock_piper_module.PiperVoice.load.assert_called_once_with("/models/default.onnx")


def test_synthesize_sync_applies_output_volume_gain() -> None:
    """_synthesize_sync scales float32 output by output_volume and preserves dtype."""
    from types import SimpleNamespace

    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg(tts_sample_rate=4, output_volume=0.5))
    samples_in = np.array([0, 13107, -19661, 32767], dtype=np.int16)

    def write_wav(text: str, wav_file: wave.Wave_write) -> None:
        assert text == "gain"
        wav_file.writeframes(samples_in.tobytes())

    tts._voice = SimpleNamespace(synthesize=write_wav)

    samples = tts._synthesize_sync("gain")

    np.testing.assert_allclose(
        samples,
        np.array([0.0, 0.2, -0.3, 0.5], dtype=np.float32),
        atol=5e-4,
    )
    assert samples.dtype == np.float32


def test_synthesize_sync_clips_after_gain() -> None:
    """Gain is clipped into the speaker-safe [-1, 1] interval."""
    from types import SimpleNamespace

    from mousedroid.voice.tts import PiperTTS

    tts = PiperTTS(_cfg(tts_sample_rate=3, output_volume=2.0))
    samples_in = np.array([26214, -29491, 6553], dtype=np.int16)

    def write_wav(text: str, wav_file: wave.Wave_write) -> None:
        assert text == "clip"
        wav_file.writeframes(samples_in.tobytes())

    tts._voice = SimpleNamespace(synthesize=write_wav)

    samples = tts._synthesize_sync("clip")

    np.testing.assert_allclose(samples, np.array([1.0, -1.0, 0.4], dtype=np.float32), atol=5e-4)
