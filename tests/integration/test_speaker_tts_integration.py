"""Integration tests for Rocky voice playback with PiperTTS and MockSpeaker."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from mousedroid.config.schema import SpeakerConfig, VoiceConfig
from mousedroid.hardware.audio.mock_speaker import MockSpeaker
from mousedroid.voice.rocky import RockyVoiceEngine
from mousedroid.voice.tts import PiperTTS


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
async def test_play_phrase_routes_piper_audio_into_mock_speaker() -> None:
    voice_cfg = VoiceConfig(enabled=True, tts_sample_rate=8, output_volume=0.5, queue_size=4)
    speaker = MockSpeaker(SpeakerConfig(sample_rate=8, chunk_size=4))
    tts = PiperTTS(voice_cfg)
    tts._voice = _FakePiperVoice(np.array([0, 32767, -16384, 8192], dtype=np.int16), 8)
    engine = RockyVoiceEngine(voice_cfg, speaker, tts)

    await engine.start()
    try:
        samples_written, peak_abs = await engine.play_phrase("diagnostic")
    finally:
        await engine.stop()

    chunks = speaker.get_written_chunks()
    assert samples_written == 4
    assert peak_abs == pytest.approx(0.5, abs=5e-4)
    assert len(chunks) == 1
    np.testing.assert_allclose(chunks[0], np.array([0.0, 0.5, -0.25, 0.125], dtype=np.float32), atol=5e-4)


@pytest.mark.asyncio
async def test_play_phrase_reports_speaker_unavailable() -> None:
    class _UnavailableSpeaker(MockSpeaker):
        def __init__(self, cfg: SpeakerConfig) -> None:
            super().__init__(cfg)
            self._stream = None

    voice_cfg = VoiceConfig(enabled=True, tts_sample_rate=8, queue_size=4)
    speaker = _UnavailableSpeaker(SpeakerConfig(sample_rate=8, chunk_size=4))
    tts = PiperTTS(voice_cfg)
    tts._voice = _FakePiperVoice(np.array([0, 1000], dtype=np.int16), 8)
    engine = RockyVoiceEngine(voice_cfg, speaker, tts)

    with pytest.raises(RuntimeError, match="speaker device unavailable"):
        await engine.play_phrase("diagnostic")