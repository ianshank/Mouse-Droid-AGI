"""Integration test for the full Rocky voice pipeline.

Tests the end-to-end flow: event -> phrase bank -> TTS -> speaker output.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from mousedroid.config.schema import SpeakerConfig, VoiceConfig
from mousedroid.hardware.audio.mock_speaker import MockSpeaker
from mousedroid.voice.mock_tts import MockTTS
from mousedroid.voice.rocky import RockyVoiceEngine


def _build_pipeline() -> tuple[RockyVoiceEngine, MockSpeaker, MockTTS]:
    """Build a complete voice pipeline with mocks."""
    voice_cfg = VoiceConfig(
        enabled=True,
        cooldown_s=0.1,
        tts_sample_rate=22050,
        queue_size=8,
    )
    speaker_cfg = SpeakerConfig(sample_rate=22050, chunk_size=512)
    speaker = MockSpeaker(speaker_cfg)
    tts = MockTTS(voice_cfg)
    engine = RockyVoiceEngine(voice_cfg, speaker, tts)
    return engine, speaker, tts


@pytest.mark.asyncio
async def test_event_to_audio_output() -> None:
    """Full pipeline: event -> phrase -> TTS -> speaker chunks."""
    engine, speaker, tts = _build_pipeline()
    await engine.start()
    try:
        await engine.speak("task_complete")
        await asyncio.sleep(0.5)

        # TTS was called with a phrase from the bank
        calls = tts.get_calls()
        assert len(calls) == 1

        # Speaker received audio chunks
        chunks = speaker.get_written_chunks()
        assert len(chunks) > 0

        # Total samples match TTS output (22050 samples / 512 chunk = ~44 chunks)
        total_samples = sum(c.shape[0] for c in chunks)
        assert total_samples >= 22050  # At least one second of audio
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_multiple_events_sequential() -> None:
    """Multiple events are spoken sequentially with cooldown."""
    engine, _speaker, tts = _build_pipeline()
    await engine.start()
    try:
        await engine.speak("startup")
        await asyncio.sleep(0.3)

        await engine.speak("path_clear")
        await asyncio.sleep(0.3)

        calls = tts.get_calls()
        assert len(calls) == 2
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_pipeline_start_stop() -> None:
    """Pipeline starts and stops without errors."""
    engine, speaker, _ = _build_pipeline()
    await engine.start()
    assert speaker.started
    await engine.stop()
    assert not speaker.started


@pytest.mark.asyncio
async def test_all_chunk_shapes_match_config() -> None:
    """Every chunk written matches the configured chunk_size."""
    engine, speaker, _ = _build_pipeline()
    await engine.start()
    try:
        await engine.speak("error")
        await asyncio.sleep(0.3)
        for chunk in speaker.get_written_chunks():
            assert chunk.shape == (512,)
            assert chunk.dtype == np.float32
    finally:
        await engine.stop()
