"""Integration tests for PiperTTS with a loaded voice object."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from mousedroid.config.schema import VoiceConfig
from mousedroid.voice.tts import PiperTTS


def _cfg(**overrides: object) -> VoiceConfig:
    return VoiceConfig(enabled=True, **overrides)  # type: ignore[arg-type]


class _FakePiperVoice:
    def __init__(self, samples: np.ndarray, sample_rate: int) -> None:
        self._samples = samples.astype(np.int16, copy=False)
        self._sample_rate = sample_rate

    def synthesize_wav(self, text: str, wav_file: io.BytesIO) -> None:
        with wave.open(wav_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(self._samples.tobytes())

    def synthesize(self, text: str, wav_file: wave.Wave_write) -> None:
        wav_file.writeframes(self._samples.tobytes())


@pytest.mark.asyncio
async def test_tts_synthesize_with_loaded_voice_produces_scaled_audio() -> None:
    cfg = _cfg(tts_sample_rate=8, output_volume=0.5)
    tts = PiperTTS(cfg)
    raw_samples = np.array([0, 32767, -16384, 8192], dtype=np.int16)
    tts._voice = _FakePiperVoice(raw_samples, sample_rate=cfg.tts_sample_rate)

    samples = await tts.synthesize("integration")

    assert samples.dtype == np.float32
    assert samples.shape == (4,)
    np.testing.assert_allclose(
        samples,
        np.array([0.0, 0.5, -0.25, 0.125], dtype=np.float32),
        atol=5e-4,
    )


@pytest.mark.asyncio
async def test_tts_synthesize_clips_when_gain_exceeds_unity() -> None:
    cfg = _cfg(tts_sample_rate=4, output_volume=2.0)
    tts = PiperTTS(cfg)
    raw_samples = np.array([32767, -32768, 4096], dtype=np.int16)
    tts._voice = _FakePiperVoice(raw_samples, sample_rate=cfg.tts_sample_rate)

    samples = await tts.synthesize("integration")

    np.testing.assert_allclose(samples, np.array([1.0, -1.0, 0.25], dtype=np.float32), atol=5e-4)
