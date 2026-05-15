"""Unit tests for per-event cooldown + token-bucket fairness in RockyVoiceEngine.

These tests cover PR #6 (V2 + V5) acceptance:

- Burst suppression by per-event cooldown emits exactly one spoken phrase
  and ``N-1`` recorded ``event_dropped_cooldown`` events.
- Token bucket drains for non-emergency priority classes and refills over
  time at the configured rate.
- Emergency events bypass both the cooldown gate and the token bucket.
- Every drop is observed via the injected ``FailureRecorder``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

from mousedroid.common.rate_limit import TokenBucket
from mousedroid.common.time.protocol import MockClock
from mousedroid.config.schema import SpeakerConfig, VoiceConfig
from mousedroid.hardware.audio.mock_speaker import MockSpeaker
from mousedroid.telemetry.failure_recorder import SeverityLevel
from mousedroid.voice.mock_tts import MockTTS
from mousedroid.voice.rocky import RockyVoiceEngine


@dataclass
class _RecordedFailure:
    """One observed FailureRecorder.record() call."""

    subsystem: str
    reason: str
    level: SeverityLevel
    extra: dict[str, str | int | float] = field(default_factory=dict)


class _RecordingFailureRecorder:
    """Test double that captures every record() call."""

    def __init__(self) -> None:
        self.events: list[_RecordedFailure] = []

    def record(
        self,
        subsystem: str,
        reason: str,
        *,
        level: SeverityLevel = "warning",
        extra: Mapping[str, str | int | float] | None = None,
    ) -> None:
        self.events.append(
            _RecordedFailure(
                subsystem=subsystem,
                reason=reason,
                level=level,
                extra=dict(extra) if extra else {},
            )
        )

    def reasons(self) -> list[str]:
        return [e.reason for e in self.events]

    def reason_count(self, reason: str) -> int:
        return sum(1 for e in self.events if e.reason == reason)


def _make_engine(
    *,
    cooldown_s: float = 5.0,
    cooldown_per_event: dict[str, float] | None = None,
    token_bucket_capacity: int = 3,
    token_bucket_refill_rate: float = 1.0,
    queue_size: int = 64,
) -> tuple[RockyVoiceEngine, MockSpeaker, MockTTS, _RecordingFailureRecorder]:
    """Construct a RockyVoiceEngine wired with a recording failure recorder."""
    voice_cfg = VoiceConfig(
        enabled=True,
        cooldown_s=cooldown_s,
        cooldown_per_event=cooldown_per_event or {},
        token_bucket_capacity=token_bucket_capacity,
        token_bucket_refill_rate=token_bucket_refill_rate,
        queue_size=queue_size,
        tts_sample_rate=22050,
    )
    speaker_cfg = SpeakerConfig(sample_rate=22050, chunk_size=1024)
    speaker = MockSpeaker(speaker_cfg)
    tts = MockTTS(voice_cfg)
    recorder = _RecordingFailureRecorder()
    engine = RockyVoiceEngine(voice_cfg, speaker, tts, failure_recorder=recorder)
    return engine, speaker, tts, recorder


# ---------------------------------------------------------------------------
# Token bucket primitive
# ---------------------------------------------------------------------------


class TestTokenBucket:
    """Verify the shared :class:`TokenBucket` from ``mousedroid.common.rate_limit``.

    PR #76 follow-up: the voice engine used to have a redundant
    ``_TokenBucket``. It now shares :class:`TokenBucket` with the
    MCP and REST control planes, with :class:`ClockProtocol` injected
    so tests can drive time deterministically (no wall-clock waits).
    Addresses Gemini medium DRY review on PR #76.
    """

    async def test_consumes_until_empty(self) -> None:
        clock = MockClock(start=0.0)
        bucket = TokenBucket(rate_per_s=0.0, capacity=3.0, clock=clock)
        taken1, _ = await bucket.take()
        taken2, _ = await bucket.take()
        taken3, _ = await bucket.take()
        taken4, _ = await bucket.take()
        assert taken1
        assert taken2
        assert taken3
        assert not taken4

    async def test_refills_over_time(self) -> None:
        clock = MockClock(start=0.0)
        bucket = TokenBucket(rate_per_s=1000.0, capacity=2.0, clock=clock)
        taken1, _ = await bucket.take()
        taken2, _ = await bucket.take()
        taken3, _ = await bucket.take()
        assert taken1
        assert taken2
        assert not taken3  # exhausted at capacity=2
        # Advance simulated time by 10 ms — at 1000 tok/s that's >= 1 token.
        clock._now = 0.01  # type: ignore[attr-defined]
        taken4, _ = await bucket.take()
        assert taken4

    async def test_capped_at_capacity(self) -> None:
        clock = MockClock(start=0.0)
        bucket = TokenBucket(rate_per_s=1000.0, capacity=2.0, clock=clock)
        # Drain initial 2 tokens, then jump simulated time 10 s ahead;
        # refill would mathematically add 10 000 tokens but the cap
        # holds at 2 — so we should get exactly 2 takes, then a drop.
        await bucket.take()
        await bucket.take()
        clock._now = 10.0  # type: ignore[attr-defined]
        taken1, _ = await bucket.take()
        taken2, _ = await bucket.take()
        taken3, _ = await bucket.take()
        assert taken1
        assert taken2
        assert not taken3


# ---------------------------------------------------------------------------
# Per-event cooldown — burst suppression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_burst_obstacle_events_per_event_cooldown_drops_logged() -> None:
    """10 obstacle events in tight burst with 5 s cooldown → 1 spoken + 9 drops."""
    engine, _speaker, tts, recorder = _make_engine(
        cooldown_s=5.0,
        cooldown_per_event={"obstacle_detected": 5.0},
        # Capacity high so the bucket never gates this test.
        token_bucket_capacity=100,
        token_bucket_refill_rate=100.0,
    )
    await engine.start()
    try:
        for _ in range(10):
            await engine.speak("obstacle_detected")
        # Drain the worker queue.
        await asyncio.sleep(0.5)
        assert len(tts.get_calls()) == 1
        assert recorder.reason_count("event_dropped_cooldown") == 9
        assert recorder.reason_count("event_dropped_rate_limit") == 0
        # Every drop should target the voice subsystem.
        assert all(e.subsystem == "voice" for e in recorder.events)
        # Each drop carries the event name in its extra payload.
        for event in recorder.events:
            assert event.extra.get("event") == "obstacle_detected"
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_per_event_cooldown_isolated_per_event() -> None:
    """Two different events under per-event cooldown each speak independently."""
    engine, _speaker, tts, recorder = _make_engine(
        cooldown_s=5.0,
        cooldown_per_event={
            "obstacle_detected": 5.0,
            "low_battery": 5.0,
        },
        token_bucket_capacity=10,
        token_bucket_refill_rate=10.0,
    )
    await engine.start()
    try:
        await engine.speak("obstacle_detected")
        await engine.speak("low_battery")
        await asyncio.sleep(0.3)
        assert len(tts.get_calls()) == 2
        assert recorder.reason_count("event_dropped_cooldown") == 0
    finally:
        await engine.stop()


# ---------------------------------------------------------------------------
# Token-bucket backpressure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_bucket_drains_for_high_priority() -> None:
    """HIGH-priority bucket: capacity=2 → 3rd event dropped + recorded.

    Per-event cooldowns are set well below the test window so the cooldown
    gate doesn't suppress events before the token bucket has a chance to.
    """
    engine, _speaker, tts, recorder = _make_engine(
        cooldown_s=5.0,
        cooldown_per_event={
            "obstacle_detected": 0.001,
            "low_battery": 0.001,
            "error": 0.001,
        },
        token_bucket_capacity=2,
        # Tiny positive refill keeps Pydantic happy (cfg requires > 0) while
        # ensuring no token refills during the synchronous test window.
        token_bucket_refill_rate=1e-9,
    )
    await engine.start()
    try:
        await engine.speak("obstacle_detected")  # HIGH
        await engine.speak("low_battery")  # HIGH
        await engine.speak("error")  # HIGH — bucket empty, drop
        await asyncio.sleep(0.3)
        assert len(tts.get_calls()) == 2
        assert recorder.reason_count("event_dropped_rate_limit") == 1
        assert recorder.reason_count("event_dropped_cooldown") == 0
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_token_bucket_separate_per_priority_class() -> None:
    """HIGH and NORMAL each have their own bucket — draining HIGH doesn't gate NORMAL."""
    engine, _speaker, tts, recorder = _make_engine(
        cooldown_s=5.0,
        cooldown_per_event={
            "obstacle_detected": 0.001,
            "error": 0.001,
            "idle": 0.001,
        },
        token_bucket_capacity=1,
        token_bucket_refill_rate=1e-9,
    )
    await engine.start()
    try:
        await engine.speak("obstacle_detected")  # HIGH — consumes HIGH bucket
        await engine.speak("error")  # HIGH — bucket empty, drop
        await engine.speak("idle")  # NORMAL — own bucket, allowed
        await asyncio.sleep(0.3)
        assert len(tts.get_calls()) == 2
        assert recorder.reason_count("event_dropped_rate_limit") == 1
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_emergency_bypasses_token_bucket_and_cooldown() -> None:
    """Emergency events are never gated by cooldown or token bucket."""
    engine, _speaker, tts, recorder = _make_engine(
        cooldown_s=5.0,
        cooldown_per_event={"emergency_stop": 5.0},
        token_bucket_capacity=1,
        token_bucket_refill_rate=1e-9,
    )
    await engine.start()
    try:
        # Repeated emergencies all enqueue; none are dropped.
        for _ in range(5):
            await engine.speak("emergency_stop")
        await asyncio.sleep(0.3)
        assert len(tts.get_calls()) == 5
        assert recorder.events == []
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_token_bucket_refills() -> None:
    """After waiting > 1/refill_rate, the bucket admits another event."""
    engine, _speaker, tts, recorder = _make_engine(
        cooldown_s=5.0,
        cooldown_per_event={
            "obstacle_detected": 0.001,
            "error": 0.001,
        },
        token_bucket_capacity=1,
        # 50 tokens/sec ⇒ ~20 ms per token.
        token_bucket_refill_rate=50.0,
    )
    await engine.start()
    try:
        await engine.speak("obstacle_detected")  # HIGH — bucket emptied
        # Burst dropped: bucket empty until refill.
        await engine.speak("error")  # dropped
        assert recorder.reason_count("event_dropped_rate_limit") == 1
        # Wait long enough for the bucket to refill.
        await asyncio.sleep(0.1)
        await engine.speak("error")
        await asyncio.sleep(0.3)
        assert len(tts.get_calls()) == 2
    finally:
        await engine.stop()


# ---------------------------------------------------------------------------
# Drops are recorded with the right subsystem/reason
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dropped_event_failure_record_payload() -> None:
    """Recorded payload identifies subsystem=voice and carries event/priority."""
    engine, _speaker, _tts, recorder = _make_engine(
        cooldown_s=5.0,
        cooldown_per_event={"obstacle_detected": 5.0},
        token_bucket_capacity=100,
        token_bucket_refill_rate=100.0,
    )
    await engine.start()
    try:
        await engine.speak("obstacle_detected")
        await engine.speak("obstacle_detected")  # dropped by cooldown
        await asyncio.sleep(0.1)
        assert len(recorder.events) == 1
        drop = recorder.events[0]
        assert drop.subsystem == "voice"
        assert drop.reason == "event_dropped_cooldown"
        assert drop.extra.get("event") == "obstacle_detected"
        assert drop.extra.get("priority") == "HIGH"
        assert drop.level in ("warning", "error", "critical")
    finally:
        await engine.stop()


# ---------------------------------------------------------------------------
# Backwards compatibility — default cooldown_per_event/buckets don't break
# existing global cooldown semantics.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_global_cooldown_still_applies_when_no_per_event_entries() -> None:
    """With cooldown_per_event empty, the global cooldown still gates back-to-back speech."""
    engine, _speaker, tts, recorder = _make_engine(
        cooldown_s=10.0,
        cooldown_per_event={},
        token_bucket_capacity=10,
        token_bucket_refill_rate=10.0,
    )
    await engine.start()
    try:
        await engine.speak("startup")
        await engine.speak("task_complete")  # different event, but global cooldown applies
        await asyncio.sleep(0.3)
        assert len(tts.get_calls()) == 1
        assert recorder.reason_count("event_dropped_cooldown") == 1
    finally:
        await engine.stop()
