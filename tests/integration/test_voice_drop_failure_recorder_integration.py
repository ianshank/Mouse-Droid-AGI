"""Integration pin — voice drops reach a *real* PrometheusFailureRecorder.

``RockyVoiceEngine._record_drop`` passes ``extra={"event": ...}`` to the
injected :class:`FailureRecorder`.  Every pre-existing test of that path
injected a recording test double, which simply stores the mapping — so the
production splat into structlog was never exercised and a ``TypeError``
("got multiple values for argument 'event'") shipped, observed during
orchestrator shutdown in ``python -m mousedroid.main --mock-hardware``.

These tests wire the factory-built voice engine to the concrete
:class:`PrometheusFailureRecorder` so the real log call runs.
"""

from __future__ import annotations

import asyncio

import pytest
import structlog.testing

from mousedroid.common.time.protocol import MockClock
from mousedroid.config.schema import MetricsConfig, Settings, SpeakerConfig, VoiceConfig
from mousedroid.factory import build_voice_engine
from mousedroid.hardware.audio.mock_speaker import MockSpeaker
from mousedroid.telemetry.failure_recorder import PrometheusFailureRecorder
from mousedroid.telemetry.metrics import MetricsRegistry

_SAMPLE_RATE = 22050


def _settings() -> Settings:
    """Build Settings with a burst-suppressing per-event cooldown."""
    return Settings(  # type: ignore[call-arg]
        mock_hardware=True,
        voice=VoiceConfig(
            enabled=True,
            cooldown_s=5.0,
            cooldown_per_event={"obstacle_detected": 5.0},
            # Capacity high so the token bucket never gates these tests.
            token_bucket_capacity=100,
            token_bucket_refill_rate=100.0,
            queue_size=64,
            tts_sample_rate=_SAMPLE_RATE,
        ),
    )


async def _burst_drop() -> tuple[list[dict[str, object]], MetricsRegistry]:
    """Emit a 3-event burst through the real recorder; return logs + registry."""
    registry = MetricsRegistry(MetricsConfig())
    recorder = PrometheusFailureRecorder(registry)
    speaker = MockSpeaker(SpeakerConfig(sample_rate=_SAMPLE_RATE, chunk_size=1024))

    engine = build_voice_engine(
        _settings(),
        speaker=speaker,
        failure_recorder=recorder,
        clock=MockClock(start=0.0),
        metrics=registry,
    )
    if engine is None:
        pytest.fail("build_voice_engine returned None — factory wiring changed")

    await engine.start()
    try:
        with structlog.testing.capture_logs() as logs:
            for _ in range(3):
                await engine.speak("obstacle_detected")
            await asyncio.sleep(0.1)
    finally:
        await engine.stop()

    return list(logs), registry


@pytest.mark.asyncio
async def test_cooldown_drop_emits_failure_log_without_raising() -> None:
    """The drop path logs subsystem_failure_recorded instead of raising TypeError."""
    logs, _registry = await _burst_drop()

    recorded = [entry for entry in logs if entry["event"] == "subsystem_failure_recorded"]
    assert recorded, "no subsystem_failure_recorded log line was emitted"


@pytest.mark.asyncio
async def test_cooldown_drop_log_carries_the_voice_event_name() -> None:
    """The colliding caller key is namespaced, not lost."""
    logs, _registry = await _burst_drop()

    recorded = [entry for entry in logs if entry["event"] == "subsystem_failure_recorded"]
    assert recorded[0]["extra_event"] == "obstacle_detected"


@pytest.mark.asyncio
async def test_cooldown_drop_log_keeps_recorder_owned_fields() -> None:
    """subsystem/reason/log_level survive the caller's extra payload."""
    logs, _registry = await _burst_drop()

    entry = next(e for e in logs if e["event"] == "subsystem_failure_recorded")
    assert entry["subsystem"] == "voice"
    assert entry["reason"] == "event_dropped_cooldown"
    assert entry["log_level"] == "warning"


@pytest.mark.asyncio
async def test_counter_and_log_line_counts_agree() -> None:
    """Every counted drop also produced a log line.

    This is the exact invariant the bug broke: ``record()`` incremented the
    counter on its first statement and *then* raised, so drops were counted
    but never logged.  Comparing the two counts fails loudly if that
    ordering hazard ever returns.
    """
    logs, registry = await _burst_drop()

    logged = sum(1 for e in logs if e["event"] == "subsystem_failure_recorded")
    counted = registry._subsystem_failures.snapshot().get(
        ("voice", "event_dropped_cooldown", "warning")
    )

    assert counted == logged
    assert counted > 0
