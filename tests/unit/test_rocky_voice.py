"""Unit tests for the Rocky voice engine."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from mousedroid.config.schema import SpeakerConfig, VoiceConfig
from mousedroid.hardware.audio.mock_speaker import MockSpeaker
from mousedroid.voice.mock_tts import MockTTS
from mousedroid.voice.phrase_bank import DEFAULT_PHRASES
from mousedroid.voice.rocky import Priority, RockyVoiceEngine, rocky_transform


class TestRockyTransform:
    """Tests for the rocky_transform grammar function."""

    def test_strips_articles(self) -> None:
        """Articles (the, a, an) are removed."""
        assert rocky_transform("the path is clear", intensity=0.0) == "path is clear"
        assert rocky_transform("a red object", intensity=0.0) == "red object"
        assert rocky_transform("an error occurred", intensity=0.0) == "error occurred"

    def test_case_insensitive_articles(self) -> None:
        """Articles are stripped and capitalisation preserved."""
        assert rocky_transform("The path", intensity=0.0) == "Path"

    def test_high_intensity_adds_exclamation(self) -> None:
        """High intensity adds exclamation mark."""
        result = rocky_transform("path is clear", intensity=0.9)
        assert result.endswith("!")

    def test_low_intensity_no_exclamation(self) -> None:
        """Low intensity does not add exclamation."""
        result = rocky_transform("path is clear", intensity=0.3)
        assert not result.endswith("!")

    def test_existing_exclamation_not_doubled(self) -> None:
        """Already-exclaimed text is not double-exclaimed."""
        result = rocky_transform("Good!", intensity=0.9)
        assert result.count("!") == 1

    def test_empty_string(self) -> None:
        """Empty input returns empty output."""
        assert rocky_transform("") == ""

    def test_adjective_repetition_high_intensity(self) -> None:
        """Known adjectives are repeated at high intensity."""
        result = rocky_transform("path is good", intensity=0.95)
        assert result.count("good") == 3  # 3x at intensity >= 0.9

    def test_adjective_repetition_medium_intensity(self) -> None:
        """Known adjectives are repeated 2x at medium-high intensity."""
        result = rocky_transform("path is good", intensity=0.75)
        assert result.count("good") == 2

    def test_capitalisation_preserved_after_article_strip(self) -> None:
        """First word capitalised when leading article was capitalised."""
        assert rocky_transform("The path ahead", intensity=0.0) == "Path ahead"
        assert rocky_transform("A strange thing", intensity=0.0) == "Strange thing"


class TestPhraseBankCoverage:
    """Verify the phrase bank has expected events."""

    def test_required_events_present(self) -> None:
        """All critical robot events have phrases."""
        required = {
            "task_complete",
            "obstacle_detected",
            "emergency_stop",
            "path_clear",
            "low_battery",
            "new_object",
            "navigation_success",
            "error",
            "idle",
            "startup",
            "shutdown",
        }
        assert required.issubset(set(DEFAULT_PHRASES.keys()))

    def test_all_events_have_phrases(self) -> None:
        """Every event has at least one phrase."""
        for event, phrases in DEFAULT_PHRASES.items():
            assert len(phrases) > 0, f"Event {event!r} has no phrases"


def _make_engine(
    cooldown_s: float = 0.1,
    queue_size: int = 16,
) -> tuple[RockyVoiceEngine, MockSpeaker, MockTTS]:
    """Create a test engine with mock speaker and TTS."""
    voice_cfg = VoiceConfig(
        enabled=True,
        cooldown_s=cooldown_s,
        queue_size=queue_size,
        tts_sample_rate=22050,
    )
    speaker_cfg = SpeakerConfig(sample_rate=22050, chunk_size=1024)
    speaker = MockSpeaker(speaker_cfg)
    tts = MockTTS(voice_cfg)
    engine = RockyVoiceEngine(voice_cfg, speaker, tts)
    return engine, speaker, tts


@pytest.mark.asyncio
async def test_speak_queues_known_event() -> None:
    """Speaking a known event sends transformed text to TTS."""
    engine, _speaker, tts = _make_engine()
    await engine.start()
    try:
        await engine.speak("startup")
        # Give the worker time to process
        await asyncio.sleep(0.3)
        calls = tts.get_calls()
        assert len(calls) == 1
        # Text is rocky_transform'd — verify it's a non-empty string
        assert len(calls[0]) > 0
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_speak_unknown_event_ignored() -> None:
    """Unknown events are silently ignored."""
    engine, _speaker, tts = _make_engine()
    await engine.start()
    try:
        await engine.speak("totally_unknown_event")
        await asyncio.sleep(0.2)
        assert len(tts.get_calls()) == 0
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_cooldown_prevents_rapid_speech() -> None:
    """Cooldown prevents speaking too frequently."""
    engine, _speaker, tts = _make_engine(cooldown_s=10.0)
    await engine.start()
    try:
        await engine.speak("startup")
        await asyncio.sleep(0.2)
        await engine.speak("task_complete")
        await asyncio.sleep(0.2)
        # Only the first should have been spoken (cooldown blocks second)
        assert len(tts.get_calls()) == 1
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_emergency_bypasses_cooldown() -> None:
    """Emergency events bypass the cooldown timer."""
    engine, _speaker, tts = _make_engine(cooldown_s=10.0)
    await engine.start()
    try:
        await engine.speak("startup")
        await asyncio.sleep(0.2)
        await engine.speak("emergency_stop")
        await asyncio.sleep(0.2)
        # Both should have been spoken: startup + emergency
        assert len(tts.get_calls()) == 2
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_synthesised_audio_written_to_speaker() -> None:
    """TTS output is written to the speaker in chunks."""
    engine, speaker, _tts = _make_engine()
    await engine.start()
    try:
        await engine.speak("startup")
        await asyncio.sleep(0.3)
        # MockTTS returns 22050 samples (1 second at 22050 Hz)
        # With chunk_size=1024, that's ceil(22050/1024) = 22 chunks
        chunks = speaker.get_written_chunks()
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.shape == (1024,)
            assert chunk.dtype == np.float32
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_phrase_overrides() -> None:
    """User-provided phrase overrides replace defaults."""
    voice_cfg = VoiceConfig(
        enabled=True,
        cooldown_s=0.1,
        tts_sample_rate=22050,
        phrase_overrides={"startup": ["Custom hello!"]},
    )
    speaker_cfg = SpeakerConfig(sample_rate=22050, chunk_size=1024)
    speaker = MockSpeaker(speaker_cfg)
    tts = MockTTS(voice_cfg)
    engine = RockyVoiceEngine(voice_cfg, speaker, tts)
    await engine.start()
    try:
        await engine.speak("startup")
        await asyncio.sleep(0.3)
        assert tts.get_calls() == ["Custom hello!"]
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_start_stop_lifecycle() -> None:
    """Engine starts and stops cleanly."""
    engine, speaker, _ = _make_engine()
    await engine.start()
    assert speaker.started
    await engine.stop()
    assert not speaker.started


class TestPriority:
    """Priority enum tests."""

    def test_ordering(self) -> None:
        """Emergency > High > Normal."""
        assert Priority.EMERGENCY > Priority.HIGH > Priority.NORMAL

    def test_values(self) -> None:
        """Priority values are as expected."""
        assert Priority.NORMAL == 0
        assert Priority.HIGH == 1
        assert Priority.EMERGENCY == 2


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_samples_empty_array() -> None:
    """Engine handles empty audio samples gracefully."""
    engine, speaker, _tts = _make_engine()
    await engine.start()
    try:
        await engine._write_samples(np.array([], dtype=np.float32))
        assert len(speaker.get_written_chunks()) == 0
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_write_samples_smaller_than_chunk() -> None:
    """Samples smaller than chunk_size are zero-padded."""
    engine, speaker, _tts = _make_engine()
    await engine.start()
    try:
        small = np.ones(100, dtype=np.float32)
        await engine._write_samples(small)
        chunks = speaker.get_written_chunks()
        assert len(chunks) == 1
        assert chunks[0].shape == (1024,)
        # First 100 samples are 1.0, rest are 0.0
        assert chunks[0][99] == 1.0
        assert chunks[0][100] == 0.0
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_queue_full_normal_priority_dropped() -> None:
    """Normal-priority events are dropped when queue is full."""
    engine, _speaker, _tts = _make_engine(cooldown_s=100.0, queue_size=1)
    await engine.start()
    try:
        await engine.speak("startup")
        await engine.speak("task_complete")  # Should be dropped (queue full)
        # Only 1 item should be in queue
        assert engine._queue.qsize() == 1
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_drain_queue_empty() -> None:
    """Draining an empty queue does not raise."""
    engine, _speaker, _tts = _make_engine()
    await engine._drain_queue()  # Should not raise


@pytest.mark.asyncio
async def test_drain_queue_clears_all() -> None:
    """Draining removes all queued items."""
    engine, _speaker, _tts = _make_engine(cooldown_s=100.0)
    # Fill the queue without starting the worker
    await engine.speak("startup")
    await engine.speak("task_complete")
    assert engine._queue.qsize() > 0
    await engine._drain_queue()
    assert engine._queue.qsize() == 0


@pytest.mark.asyncio
async def test_stop_then_restart() -> None:
    """Engine can be stopped and restarted cleanly."""
    engine, speaker, tts = _make_engine()
    await engine.start()
    await engine.stop()
    assert not speaker.started

    tts.clear()
    speaker.clear()
    await engine.start()
    assert speaker.started
    await engine.speak("startup")
    await asyncio.sleep(0.3)
    assert len(tts.get_calls()) == 1
    await engine.stop()


@pytest.mark.asyncio
async def test_speak_with_context_parameter() -> None:
    """Context parameter is accepted without error."""
    engine, _speaker, tts = _make_engine()
    await engine.start()
    try:
        await engine.speak("startup", context={"distance_m": 1.5})
        await asyncio.sleep(0.3)
        assert len(tts.get_calls()) == 1
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_worker_handles_tts_exception() -> None:
    """Worker continues after TTS synthesis failure."""
    engine, _speaker, tts = _make_engine()
    original_synthesize = tts.synthesize

    call_count = 0

    async def failing_then_ok(text: str) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("TTS failed")
        return await original_synthesize(text)

    tts.synthesize = failing_then_ok  # type: ignore[assignment]
    await engine.start()
    try:
        await engine.speak("startup")  # Will fail
        await asyncio.sleep(0.3)
        await engine.speak("task_complete")  # Should succeed
        await asyncio.sleep(0.3)
        assert call_count == 2
    finally:
        await engine.stop()


def test_rocky_transform_whitespace_only() -> None:
    """Transform handles whitespace-only input."""
    result = rocky_transform("   ", intensity=0.0)
    assert result == ""


def test_rocky_transform_at_intensity_boundary() -> None:
    """Transform at exactly 0.7 intensity does NOT add exclamation."""
    result = rocky_transform("hello", intensity=0.7)
    assert not result.endswith("!")


@pytest.mark.asyncio
async def test_empty_phrase_override_ignored() -> None:
    """Empty phrase list from override causes speak() to no-op."""
    voice_cfg = VoiceConfig(
        enabled=True,
        cooldown_s=0.1,
        tts_sample_rate=22050,
        phrase_overrides={"startup": []},
    )
    speaker_cfg = SpeakerConfig(sample_rate=22050, chunk_size=1024)
    speaker = MockSpeaker(speaker_cfg)
    tts = MockTTS(voice_cfg)
    engine = RockyVoiceEngine(voice_cfg, speaker, tts)
    await engine.start()
    try:
        await engine.speak("startup")
        await asyncio.sleep(0.2)
        assert len(tts.get_calls()) == 0  # Empty phrase list -> no speech
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_emergency_clears_full_queue() -> None:
    """Emergency event clears full queue and inserts itself."""
    engine, _speaker, _tts = _make_engine(cooldown_s=100.0, queue_size=1)
    await engine.start()
    try:
        # Fill the queue with a normal event (cooldown blocks worker)
        await engine.speak("startup")
        assert engine._queue.qsize() == 1
        # Emergency should clear and insert
        await engine.speak("emergency_stop")
        assert engine._queue.qsize() == 1
        # The queued item should be the emergency
        item = engine._queue.get_nowait()
        assert item.priority == -Priority.EMERGENCY
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_write_samples_stereo_duplicates_channels() -> None:
    """Stereo speaker receives duplicated mono samples."""
    voice_cfg = VoiceConfig(
        enabled=True,
        cooldown_s=0.1,
        tts_sample_rate=22050,
    )
    speaker_cfg = SpeakerConfig(sample_rate=22050, chunk_size=4, channels=2)
    speaker = MockSpeaker(speaker_cfg)
    tts = MockTTS(voice_cfg)
    engine = RockyVoiceEngine(voice_cfg, speaker, tts)
    await engine.start()
    try:
        # 4 mono samples -> 8 stereo samples -> 1 chunk of 8 (4*2)
        mono = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        await engine._write_samples(mono)
        chunks = speaker.get_written_chunks()
        assert len(chunks) == 1
        # Each mono sample duplicated: [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
        expected = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4], dtype=np.float32)
        np.testing.assert_allclose(chunks[0], expected)
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_worker_timeout_continues_loop() -> None:
    """Worker handles queue.get timeout without crashing."""
    # Use very short poll timeout so it triggers quickly
    voice_cfg = VoiceConfig(
        enabled=True,
        cooldown_s=0.1,
        queue_size=16,
        tts_sample_rate=22050,
        queue_poll_timeout_s=0.05,
    )
    speaker_cfg = SpeakerConfig(sample_rate=22050, chunk_size=1024)
    speaker = MockSpeaker(speaker_cfg)
    tts = MockTTS(voice_cfg)
    engine = RockyVoiceEngine(voice_cfg, speaker, tts)
    await engine.start()
    try:
        # Let the worker spin through a few timeout cycles
        await asyncio.sleep(0.2)
        # Then queue something — it should still be processed
        await engine.speak("startup")
        await asyncio.sleep(0.3)
        assert len(tts.get_calls()) == 1
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_speak_uses_valence_from_context() -> None:
    """Context valence controls rocky_transform intensity."""
    # Use phrase override with a plain sentence (no trailing !)
    # so we can verify exclamation behaviour from intensity alone
    voice_cfg = VoiceConfig(
        enabled=True,
        cooldown_s=0.1,
        tts_sample_rate=22050,
        phrase_overrides={"startup": ["hello world"]},
    )
    speaker_cfg = SpeakerConfig(sample_rate=22050, chunk_size=1024)
    speaker = MockSpeaker(speaker_cfg)
    tts = MockTTS(voice_cfg)
    engine = RockyVoiceEngine(voice_cfg, speaker, tts)
    await engine.start()
    try:
        # Low valence -> no exclamation
        await engine.speak("startup", context={"valence": 0.1})
        await asyncio.sleep(0.3)
        calls = tts.get_calls()
        assert len(calls) == 1
        assert not calls[0].endswith("!")
    finally:
        await engine.stop()
