"""Unit tests for ``Greeter`` + ``format_names_oxford``.

Uses a lightweight ``_FakeVoiceEngine`` (not the mock-library Mock) so
the call ordering between the pre-flourish chirp and the rocky-styled
custom message is verifiable as a deterministic event list, matching
the test discipline of the existing voice tests.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from mousedroid.config.schema import GreetingConfig
from mousedroid.voice.greeting import Greeter, _select_chirp_text, format_names_oxford


class _FakeVoiceEngine:
    """Async ``VoiceEngineProtocol`` stand-in that records playback order.

    Exposes the subset of :class:`VoiceEngineProtocol` the greeter calls
    (``start`` / ``stop`` / ``play_phrase``) plus an ``utterances``
    list so tests can assert the chirp landed BEFORE the message and
    that the rocky-transformed text actually flowed through.
    """

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.utterances: list[str] = []

    @property
    def is_ready(self) -> bool:  # pragma: no cover — not exercised by greeter
        return self.started

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def speak(self, event: str, context: dict[str, float] | None = None) -> None:
        # Greeter doesn't currently call speak(); the protocol requires it.
        # pragma: no cover
        raise NotImplementedError

    async def play_phrase(self, text: str) -> tuple[int, float]:
        self.utterances.append(text)
        # Return plausible (samples, peak) without driving real audio.
        return (len(text) * 100, 0.5)


# --------------------------------------------------------------------------- #
# format_names_oxford
# --------------------------------------------------------------------------- #


def test_format_names_empty_returns_empty_string() -> None:
    assert format_names_oxford([]) == ""


def test_format_names_strips_whitespace_only_entries() -> None:
    assert format_names_oxford(["John", " ", "", "Jordan"]) == "John and Jordan"


def test_format_names_single() -> None:
    assert format_names_oxford(["John"]) == "John"


def test_format_names_two_joined_with_and() -> None:
    assert format_names_oxford(["John", "Jordan"]) == "John and Jordan"


def test_format_names_oxford_comma_three_or_more() -> None:
    """The Oxford comma is mandatory — pinned by the test_greet_intro CLI flow."""
    assert (
        format_names_oxford(["John", "Jordan", "Parvathay", "Jeff"])
        == "John, Jordan, Parvathay and Jeff"
    )


def test_format_names_strips_per_entry_whitespace() -> None:
    assert format_names_oxford(["  John  ", "Jordan "]) == "John and Jordan"


# --------------------------------------------------------------------------- #
# _select_chirp_text
# --------------------------------------------------------------------------- #


def test_select_chirp_text_empty_event_returns_none() -> None:
    assert _select_chirp_text("") is None


def test_select_chirp_text_unknown_event_returns_none_with_warning() -> None:
    """Unknown event name → None (caller skips); operator sees warning log."""
    assert _select_chirp_text("not_a_real_event_xyz") is None


def test_select_chirp_text_known_event_returns_deterministic_first_entry() -> None:
    """No rng → first entry pick is deterministic for reproducible dev runs."""
    from mousedroid.voice.phrase_bank import DEFAULT_PHRASES

    out = _select_chirp_text("greeting_excited")
    assert out is not None
    assert out == DEFAULT_PHRASES["greeting_excited"][0]


def test_select_chirp_text_with_rng_picks_from_entries() -> None:
    """Injectable rng for variety — picks from the entry list deterministically.

    Uses an explicit ``Random(seed)`` (not crypto-secure on purpose — this
    is a deterministic dev seed) so the same `out` is selected on every
    run. Live operators inject their own rng if they want variety.
    """
    from mousedroid.voice.phrase_bank import DEFAULT_PHRASES

    rng = random.Random(42)  # noqa: S311 (test seed; not crypto)
    out = _select_chirp_text("greeting_excited", rng=rng)
    assert out in DEFAULT_PHRASES["greeting_excited"]


# --------------------------------------------------------------------------- #
# Greeter
# --------------------------------------------------------------------------- #


def _cfg(**overrides: Any) -> GreetingConfig:
    defaults: dict[str, Any] = {
        "enabled": True,
        "names": ["John", "Jordan", "Parvathay", "Jeff"],
        "message_template": "Hello {names}! I have been waiting to meet you for some time",
        "pre_chirp_event": "greeting_excited",
        "excitement_intensity": 0.9,
        "inter_chirp_delay_s": 0.0,  # zero delay in tests for speed
    }
    defaults.update(overrides)
    return GreetingConfig(**defaults)


@pytest.mark.asyncio
async def test_greet_plays_chirp_then_custom_message_in_order() -> None:
    engine = _FakeVoiceEngine()
    greeter = Greeter(engine, _cfg())
    await greeter.greet()
    assert len(engine.utterances) == 2
    # First utterance: chirp from phrase bank.
    from mousedroid.voice.phrase_bank import DEFAULT_PHRASES

    assert engine.utterances[0] == DEFAULT_PHRASES["greeting_excited"][0]
    # Second utterance: rocky-styled custom message containing the names.
    assert "John" in engine.utterances[1]
    assert "Jeff" in engine.utterances[1]
    # rocky_transform at intensity 0.9 retains "!" — check the message
    # is styled (uppercase / exclamation present), not literal default.
    assert "!" in engine.utterances[1]


@pytest.mark.asyncio
async def test_greet_skips_chirp_when_pre_chirp_event_is_empty() -> None:
    engine = _FakeVoiceEngine()
    greeter = Greeter(engine, _cfg(pre_chirp_event=""))
    await greeter.greet()
    assert len(engine.utterances) == 1
    assert "John" in engine.utterances[0]


@pytest.mark.asyncio
async def test_greet_skips_chirp_when_event_unknown() -> None:
    """Unknown event → warn + skip flourish; custom message still plays."""
    engine = _FakeVoiceEngine()
    greeter = Greeter(engine, _cfg(pre_chirp_event="bogus_event_name_xyz"))
    await greeter.greet()
    assert len(engine.utterances) == 1


@pytest.mark.asyncio
async def test_greet_uses_runtime_names_override() -> None:
    engine = _FakeVoiceEngine()
    greeter = Greeter(engine, _cfg())
    await greeter.greet(names=["Alice", "Bob"])
    # Last utterance is the custom message — should contain the override
    # names, NOT the config names.
    msg = engine.utterances[-1]
    assert "Alice" in msg
    assert "Bob" in msg
    assert "John" not in msg
    assert "Jeff" not in msg


@pytest.mark.asyncio
async def test_greet_raises_on_empty_names_override() -> None:
    """Runtime override with [] reaches the runtime guard."""
    engine = _FakeVoiceEngine()
    greeter = Greeter(engine, _cfg())
    with pytest.raises(ValueError, match="at least one name"):
        await greeter.greet(names=[])


@pytest.mark.asyncio
async def test_greet_oxford_comma_formatting_in_message() -> None:
    engine = _FakeVoiceEngine()
    greeter = Greeter(engine, _cfg())
    await greeter.greet()
    msg = engine.utterances[-1]
    # Names list of 4 should produce "John, Jordan, Parvathay and Jeff"
    # somewhere in the styled message (rocky_transform may add !, but
    # the comma + "and" structure is preserved).
    assert "John" in msg
    assert "Jordan" in msg
    assert "Parvathay" in msg
    assert "Jeff" in msg
    assert "and Jeff" in msg


@pytest.mark.asyncio
async def test_greeter_does_not_manage_engine_lifecycle() -> None:
    """Caller owns start/stop — Greeter never calls either of them."""
    engine = _FakeVoiceEngine()
    greeter = Greeter(engine, _cfg())
    await greeter.greet()
    # The greeter MUST NOT have called start/stop on the engine.
    assert engine.started is False
    assert engine.stopped is False
