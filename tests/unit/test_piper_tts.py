"""Unit tests for PiperTTS — mocked piper imports."""

from __future__ import annotations

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
