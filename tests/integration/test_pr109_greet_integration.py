"""Integration test: MSE-6 greeting end-to-end through the real voice stack.

Issue #109. Drives :meth:`Greeter.greet` through a real
:class:`RockyVoiceEngine` built by the production ``build_voice_engine``
factory in ``mock_hardware`` mode (MockTTS + MockSpeaker — no audio HW).
Asserts the full phrase ordering surfaces as the documented structlog
event sequence:

    greeting_started
      -> greeting_chirp_playing  (pre-flourish text)
      -> greeting_chirp_done
      -> [inter_chirp_delay_s honoured between chirp and message]
      -> greeting_message_playing (rocky-styled message, names interpolated)
      -> greeting_done            (terminal)

No hardcoded phrase text or delays — every tunable comes from
``GreetingConfig`` / the phrase bank. Structlog events are captured via
``structlog.testing.capture_logs`` (the repo's existing capture pattern,
see ``tests/unit/voice/test_tts_synthesize_adapter.py``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import structlog.testing

from mousedroid.config.schema import Settings
from mousedroid.factory import build_greeter, build_voice_engine
from mousedroid.voice.phrase_bank import DEFAULT_PHRASES

# Slot ordering the greeter emits — pinned so a future reorder is caught.
_EXPECTED_EVENT_ORDER = (
    "greeting_started",
    "greeting_chirp_playing",
    "greeting_chirp_done",
    "greeting_message_playing",
    "greeting_done",
)


def _settings(*, inter_chirp_delay_s: float) -> Settings:
    """Real mock-hardware settings with voice + speaker + greeting enabled.

    Built via ``Settings.model_validate`` so every nested validator runs
    (matches the factory test discipline). ``inter_chirp_delay_s`` is the
    only knob the individual tests vary.
    """
    return Settings.model_validate(
        {
            "mock_hardware": True,
            "voice": {"enabled": True},
            "speaker": {"enabled": True},
            "greeting": {
                "enabled": True,
                "names": ["John", "Jordan", "Parvathay", "Jeff"],
                "inter_chirp_delay_s": inter_chirp_delay_s,
            },
        }
    )


@pytest.mark.asyncio
async def test_greet_end_to_end_emits_ordered_events_through_real_engine() -> None:
    cfg = _settings(inter_chirp_delay_s=0.0)
    # Build the engine via the production factory so we exercise the real
    # RockyVoiceEngine + MockTTS + MockSpeaker chain (not a hand-rolled fake).
    engine = build_voice_engine(cfg)
    assert engine is not None, "mock-hardware voice engine must build"
    greeter = build_greeter(cfg, voice_engine=engine)

    await engine.start()  # caller owns lifecycle (Greeter never starts it)
    try:
        with structlog.testing.capture_logs() as logs:
            await greeter.greet()
    finally:
        await engine.stop()

    events = [e["event"] for e in logs if e["event"].startswith("greeting_")]
    # The documented terminal event is present and last.
    assert events[-1] == "greeting_done"
    # The full ordering is a subsequence of what was emitted.
    assert events == list(_EXPECTED_EVENT_ORDER), events


@pytest.mark.asyncio
async def test_greet_chirp_text_then_styled_message_with_names() -> None:
    cfg = _settings(inter_chirp_delay_s=0.0)
    engine = build_voice_engine(cfg)
    assert engine is not None
    greeter = build_greeter(cfg, voice_engine=engine)

    await engine.start()
    try:
        with structlog.testing.capture_logs() as logs:
            await greeter.greet()
    finally:
        await engine.stop()

    by_event = {e["event"]: e for e in logs}
    # Pre-flourish chirp carries the phrase-bank text (config-sourced, not literal).
    chirp = by_event["greeting_chirp_playing"]
    assert chirp["text"] == DEFAULT_PHRASES["greeting_excited"][0]
    # The styled message interpolates the Oxford-comma name list.
    msg = by_event["greeting_message_playing"]
    assert "John" in msg["text"]
    assert "and Jeff" in msg["text"]
    assert msg["names"] == "John, Jordan, Parvathay and Jeff"
    # Terminal event reports a non-empty synthesized sample count.
    assert by_event["greeting_done"]["samples"] > 0


@pytest.mark.asyncio
async def test_greet_honours_inter_chirp_delay() -> None:
    """A non-zero ``inter_chirp_delay_s`` is awaited between chirp and message.

    Asserts deterministically that the greeter awaits ``asyncio.sleep`` with
    the config-sourced delay (never hardcoded in the greeter). A wall-clock
    measurement would be flaky under cross-test event-loop state (another test
    can leave ``asyncio.sleep`` sped/patched), so we spy the call instead.
    """
    delay_s = 0.05
    cfg = _settings(inter_chirp_delay_s=delay_s)
    assert cfg.greeting is not None
    assert cfg.greeting.inter_chirp_delay_s == delay_s
    engine = build_voice_engine(cfg)
    assert engine is not None
    greeter = build_greeter(cfg, voice_engine=engine)

    await engine.start()
    try:
        with patch("mousedroid.voice.greeting.asyncio.sleep", new_callable=AsyncMock) as sleep_spy:
            await greeter.greet()
    finally:
        await engine.stop()

    # The inter-chirp delay branch must fire with the EXACT config value —
    # deterministic, no wall-clock dependency.
    assert any(
        call.args == (delay_s,) for call in sleep_spy.await_args_list
    ), f"inter-chirp delay {delay_s}s not awaited: {sleep_spy.await_args_list}"
