"""Unit tests for CommentaryEngine gating, EW novelty stats, and resilience."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from mousedroid.commentary.engine import SUPPRESSION_REASONS, CommentaryEngine
from mousedroid.commentary.protocol import CommentaryFacts
from mousedroid.common.time.protocol import MockClock, RealClock
from mousedroid.config.schema import CommentaryConfig, MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.voice.protocol import VoiceEngineProtocol


def _idle_facts(*, novelty: float | None, is_emergency: bool = False) -> CommentaryFacts:
    return CommentaryFacts(
        min_clearance_m=5.0,
        forward_distance_m=5.0,
        audio_rms=0.0,
        speed_mps=0.0,
        turn_rate=0.0,
        battery_v=12.0,
        novelty=novelty,
        is_emergency=is_emergency,
        lidar_valid=True,
        audio_valid=True,
        timestamp=0.0,
    )


class _StubComposer:
    def __init__(self, text: str = "hello there") -> None:
        self._text = text
        self.calls = 0

    async def compose(self, facts: CommentaryFacts) -> str:
        self.calls += 1
        return self._text


class _BusyVoice:
    """VoiceEngineProtocol + SpeakerBusyProtocol (always speaking)."""

    is_speaking = True

    async def speak(self, event: str, context: dict[str, float] | None = None) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def play_phrase(self, text: str) -> tuple[int, float]:
        return (1, 0.0)


def _voice() -> AsyncMock:
    v = AsyncMock(spec=VoiceEngineProtocol)
    v.play_phrase = AsyncMock(return_value=(100, 0.5))
    return v


def _engine(
    cfg: CommentaryConfig,
    *,
    voice: object | None = None,
    composer: object | None = None,
    clock: MockClock | None = None,
    reg: MetricsRegistry | None = None,
) -> CommentaryEngine:
    return CommentaryEngine(
        cfg,
        voice_engine=voice if voice is not None else _voice(),  # type: ignore[arg-type]
        composer=composer if composer is not None else _StubComposer(),  # type: ignore[arg-type]
        metrics=reg,
        clock=clock if clock is not None else MockClock(),
    )


def _cfg(**over: object) -> CommentaryConfig:
    base: dict[str, object] = {
        "enabled": True,
        "composer": "template",
        "novelty_warmup_n": 2,
        "novelty_gate_alpha": 0.3,
        "novelty_sigma": 2.0,
        "min_interval_s": 0.0,
        "cadence_s": 0.01,
    }
    base.update(over)
    return CommentaryConfig(**base)  # type: ignore[arg-type]


def _reason(reg: MetricsRegistry) -> str | None:
    snap = reg._commentary_suppressed.snapshot()
    return next(iter(snap), None) if snap else None


# --------------------------------------------------------------------------- #
# Fire path + novelty statistics
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_novelty_spike_fires_once() -> None:
    reg = MetricsRegistry(MetricsConfig())
    voice = _voice()
    eng = _engine(_cfg(), voice=voice, composer=_StubComposer("open space here"), reg=reg)
    # Build a small varied distribution past warmup, then a clear outlier.
    eng.observe(1.0, _idle_facts(novelty=1.0))
    eng.observe(2.0, _idle_facts(novelty=2.0))
    eng.observe(100.0, _idle_facts(novelty=100.0))
    await eng._evaluate_and_speak()
    assert voice.play_phrase.await_count == 1
    assert reg._commentary_emitted.value == 1


@pytest.mark.asyncio
async def test_compare_before_update_and_peak_reset() -> None:
    """A lone spike fires once; the next window (no new spike) does not."""
    voice = _voice()
    eng = _engine(_cfg(), voice=voice)
    eng.observe(1.0, _idle_facts(novelty=1.0))
    eng.observe(2.0, _idle_facts(novelty=2.0))
    eng.observe(100.0, _idle_facts(novelty=100.0))
    await eng._evaluate_and_speak()
    assert voice.play_phrase.await_count == 1
    # Next window: only an in-distribution sample (no spike) -> peak reset means
    # the old spike does not re-fire.
    eng.observe(2.0, _idle_facts(novelty=2.0))
    await eng._evaluate_and_speak()
    assert voice.play_phrase.await_count == 1


@pytest.mark.asyncio
async def test_ew_stats_track_and_std_nonnegative() -> None:
    eng = _engine(_cfg())
    for v in (1.0, 1.0, 1.0, 1.0):
        eng.observe(v, _idle_facts(novelty=v))
    assert eng._mean == pytest.approx(1.0, abs=1e-6)
    assert eng._std() >= 0.0


# --------------------------------------------------------------------------- #
# Suppression branches (each asserts its reason)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_suppress_no_facts() -> None:
    reg = MetricsRegistry(MetricsConfig())
    eng = _engine(_cfg(), reg=reg)
    await eng._evaluate_and_speak()
    assert _reason(reg) == "no_facts"


@pytest.mark.asyncio
async def test_suppress_below_threshold() -> None:
    reg = MetricsRegistry(MetricsConfig())
    eng = _engine(_cfg(), reg=reg)
    eng.observe(1.0, _idle_facts(novelty=1.0))
    eng.observe(2.0, _idle_facts(novelty=2.0))
    eng.observe(2.1, _idle_facts(novelty=2.1))  # in-distribution, not an outlier
    await eng._evaluate_and_speak()
    assert _reason(reg) == "below_threshold"


@pytest.mark.asyncio
async def test_suppress_warmup() -> None:
    reg = MetricsRegistry(MetricsConfig())
    eng = _engine(_cfg(novelty_warmup_n=100), reg=reg)
    eng.observe(100.0, _idle_facts(novelty=100.0))  # spike, but below warmup
    await eng._evaluate_and_speak()
    assert _reason(reg) == "below_threshold"


@pytest.mark.asyncio
async def test_suppress_std_floor_degenerate_variance() -> None:
    reg = MetricsRegistry(MetricsConfig())
    eng = _engine(_cfg(novelty_std_floor=1.0), reg=reg)
    for _ in range(5):
        eng.observe(5.0, _idle_facts(novelty=5.0))  # constant -> std collapses
    await eng._evaluate_and_speak()
    assert _reason(reg) == "below_threshold"


@pytest.mark.asyncio
async def test_suppress_not_idle_too_fast() -> None:
    reg = MetricsRegistry(MetricsConfig())
    eng = _engine(_cfg(allow_without_novelty=True), reg=reg)
    facts = CommentaryFacts(
        min_clearance_m=5.0,
        forward_distance_m=5.0,
        audio_rms=0.0,
        speed_mps=1.0,
        turn_rate=0.0,
        battery_v=12.0,
        novelty=None,
        is_emergency=False,
        lidar_valid=True,
        audio_valid=True,
        timestamp=0.0,
    )
    eng.observe(None, facts)
    await eng._evaluate_and_speak()
    assert _reason(reg) == "not_idle"


@pytest.mark.asyncio
async def test_suppress_not_idle_too_close() -> None:
    reg = MetricsRegistry(MetricsConfig())
    eng = _engine(_cfg(allow_without_novelty=True), reg=reg)
    facts = CommentaryFacts(
        min_clearance_m=0.1,
        forward_distance_m=0.1,
        audio_rms=0.0,
        speed_mps=0.0,
        turn_rate=0.0,
        battery_v=12.0,
        novelty=None,
        is_emergency=False,
        lidar_valid=True,
        audio_valid=True,
        timestamp=0.0,
    )
    eng.observe(None, facts)
    await eng._evaluate_and_speak()
    assert _reason(reg) == "not_idle"


@pytest.mark.asyncio
async def test_suppress_emergency_active() -> None:
    reg = MetricsRegistry(MetricsConfig())
    eng = _engine(_cfg(allow_without_novelty=True), reg=reg)
    eng.observe(None, _idle_facts(novelty=None))
    eng.observe_emergency(True)
    await eng._evaluate_and_speak()
    assert _reason(reg) == "emergency"


@pytest.mark.asyncio
async def test_suppress_post_emergency_quiet_window() -> None:
    reg = MetricsRegistry(MetricsConfig())
    clock = MockClock()
    eng = _engine(
        _cfg(allow_without_novelty=True, post_emergency_quiet_s=10.0), reg=reg, clock=clock
    )
    eng.observe_emergency(True)
    eng.observe_emergency(False)  # emergency cleared...
    clock.advance(5.0)  # ...but still inside the 10s quiet window
    eng.observe(None, _idle_facts(novelty=None))
    await eng._evaluate_and_speak()
    assert _reason(reg) == "emergency"


@pytest.mark.asyncio
async def test_suppress_busy() -> None:
    reg = MetricsRegistry(MetricsConfig())
    eng = _engine(_cfg(allow_without_novelty=True), voice=_BusyVoice(), reg=reg)
    eng.observe(None, _idle_facts(novelty=None))
    await eng._evaluate_and_speak()
    assert _reason(reg) == "busy"


@pytest.mark.asyncio
async def test_suppress_no_novelty_signal() -> None:
    reg = MetricsRegistry(MetricsConfig())
    eng = _engine(_cfg(allow_without_novelty=False), reg=reg)
    eng.observe(None, _idle_facts(novelty=None))
    await eng._evaluate_and_speak()
    assert _reason(reg) == "no_novelty_signal"


@pytest.mark.asyncio
async def test_allow_without_novelty_fires_on_cadence() -> None:
    voice = _voice()
    eng = _engine(_cfg(allow_without_novelty=True), voice=voice)
    eng.observe(None, _idle_facts(novelty=None))
    await eng._evaluate_and_speak()
    assert voice.play_phrase.await_count == 1


@pytest.mark.asyncio
async def test_suppress_cooldown() -> None:
    reg = MetricsRegistry(MetricsConfig())
    voice = _voice()
    eng = _engine(_cfg(allow_without_novelty=True, min_interval_s=10.0), voice=voice, reg=reg)
    eng.observe(None, _idle_facts(novelty=None))
    await eng._evaluate_and_speak()  # fires (last_fire_t = 0)
    eng.observe(None, _idle_facts(novelty=None))
    await eng._evaluate_and_speak()  # immediate -> cooldown
    assert voice.play_phrase.await_count == 1
    assert _reason(reg) == "cooldown"


@pytest.mark.asyncio
async def test_suppress_empty() -> None:
    reg = MetricsRegistry(MetricsConfig())
    eng = _engine(_cfg(allow_without_novelty=True), composer=_StubComposer(""), reg=reg)
    eng.observe(None, _idle_facts(novelty=None))
    await eng._evaluate_and_speak()
    assert _reason(reg) == "empty"


@pytest.mark.asyncio
async def test_suppress_empty_after_transform() -> None:
    reg = MetricsRegistry(MetricsConfig())
    # All-article text -> rocky_transform strips everything -> "".
    eng = _engine(_cfg(allow_without_novelty=True), composer=_StubComposer("the a an"), reg=reg)
    eng.observe(None, _idle_facts(novelty=None))
    await eng._evaluate_and_speak()
    assert _reason(reg) == "empty_after_transform"


@pytest.mark.asyncio
async def test_facts_emergency_flag_suppresses() -> None:
    reg = MetricsRegistry(MetricsConfig())
    eng = _engine(_cfg(allow_without_novelty=True), reg=reg)
    eng.observe(None, _idle_facts(novelty=None, is_emergency=True))
    await eng._evaluate_and_speak()
    assert _reason(reg) == "emergency"


# --------------------------------------------------------------------------- #
# run() loop: cancellation + resilience
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_run_cancellation_propagates() -> None:
    eng = _engine(_cfg(cadence_s=60.0), clock=MockClock())
    task = asyncio.create_task(eng.run())
    await asyncio.sleep(0)  # let it reach the sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_run_swallows_composer_exception() -> None:
    class _Boom:
        async def compose(self, facts: CommentaryFacts) -> str:
            raise RuntimeError("boom")

    clock = MockClock()
    eng = _engine(_cfg(allow_without_novelty=True, cadence_s=1.0), composer=_Boom(), clock=clock)
    eng.observe(None, _idle_facts(novelty=None))
    task = asyncio.create_task(eng.run())
    await asyncio.sleep(0)
    clock.advance(1.0)  # wake the loop -> composer raises -> logged, loop survives
    await asyncio.sleep(0)
    assert not task.done()  # loop did not crash
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_play_phrase_runtimeerror_caught() -> None:
    voice = _voice()
    voice.play_phrase = AsyncMock(side_effect=RuntimeError("speaker gone"))
    clock = MockClock()
    eng = _engine(_cfg(allow_without_novelty=True, cadence_s=1.0), voice=voice, clock=clock)
    eng.observe(None, _idle_facts(novelty=None))
    task = asyncio.create_task(eng.run())
    await asyncio.sleep(0)
    clock.advance(1.0)
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_stop_sets_stopped_flag() -> None:
    eng = _engine(_cfg())
    await eng.stop()
    assert eng._stopped is True


@pytest.mark.asyncio
async def test_no_voice_suppressed() -> None:
    reg = MetricsRegistry(MetricsConfig())
    eng = CommentaryEngine(
        _cfg(allow_without_novelty=True),
        voice_engine=None,  # type: ignore[arg-type]
        composer=_StubComposer(),
        metrics=reg,
        clock=MockClock(),
    )
    eng.observe(None, _idle_facts(novelty=None))
    await eng._evaluate_and_speak()
    assert _reason(reg) == "no_voice"


@pytest.mark.asyncio
async def test_cancellation_during_compose_propagates() -> None:
    """Cancelling while composing re-raises (the in-try CancelledError guard)."""
    blocker = asyncio.Event()

    class _Block:
        async def compose(self, facts: CommentaryFacts) -> str:
            await blocker.wait()
            return "x"

    eng = CommentaryEngine(
        _cfg(allow_without_novelty=True, cadence_s=0.001),
        voice_engine=_voice(),
        composer=_Block(),
        clock=RealClock(),
    )
    eng.observe(None, _idle_facts(novelty=None))
    task = asyncio.create_task(eng.run())
    await asyncio.sleep(0.02)  # reach compose and block
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_intensity_threshold_forwarded_to_transform() -> None:
    """An operator intensity_threshold is threaded into rocky_transform on fire."""
    voice = _voice()
    eng = CommentaryEngine(
        _cfg(allow_without_novelty=True),
        voice_engine=voice,
        composer=_StubComposer("good open space"),
        clock=MockClock(),
        intensity_threshold=0.5,
    )
    eng.observe(None, _idle_facts(novelty=None))
    await eng._evaluate_and_speak()
    assert voice.play_phrase.await_count == 1
    # excitement_intensity (0.6) > threshold (0.5) -> adjective repetition fires.
    spoken = voice.play_phrase.await_args.args[0]
    assert "good good" in spoken


@pytest.mark.asyncio
async def test_debug_logs_evaluation_and_suppression_reason() -> None:
    """Debug logs surface the gate inputs + reason (operator debuggability)."""
    from structlog.testing import capture_logs

    eng = _engine(_cfg())  # no facts observed -> suppressed no_facts
    with capture_logs() as logs:
        await eng._evaluate_and_speak()
    events = {e.get("event"): e for e in logs}
    assert "commentary_evaluating" in events
    assert events["commentary_evaluating"]["has_facts"] is False
    assert events["commentary_suppressed"]["reason"] == "no_facts"


def test_suppression_reasons_complete() -> None:
    """Every reason the engine emits is in its exported frozenset."""
    assert len(SUPPRESSION_REASONS) == 10
