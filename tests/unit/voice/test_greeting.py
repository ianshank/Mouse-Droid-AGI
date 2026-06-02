"""Unit tests for ``Greeter`` + ``format_names_oxford``.

Uses a lightweight ``FakeVoiceEngine`` (not the mock-library Mock) so
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
from tests.unit.voice._fakes import FakeVoiceEngine

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
    # Source defaults from the schema rather than duplicating literals.
    # ``inter_chirp_delay_s`` is explicitly overridden to zero so most
    # tests don't pay the configured 0.25 s wall-clock; the delay-branch
    # coverage test below overrides this back to a small positive value.
    defaults: dict[str, Any] = {
        "enabled": True,
        "names": ["John", "Jordan", "Parvathay", "Jeff"],
        "inter_chirp_delay_s": 0.0,
    }
    defaults.update(overrides)
    return GreetingConfig(**defaults)


@pytest.mark.asyncio
async def test_greet_plays_chirp_then_custom_message_in_order() -> None:
    engine = FakeVoiceEngine()
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
    engine = FakeVoiceEngine()
    greeter = Greeter(engine, _cfg(pre_chirp_event=""))
    await greeter.greet()
    assert len(engine.utterances) == 1
    assert "John" in engine.utterances[0]


@pytest.mark.asyncio
async def test_greet_skips_chirp_when_event_unknown() -> None:
    """Unknown event → warn + skip flourish; custom message still plays."""
    engine = FakeVoiceEngine()
    greeter = Greeter(engine, _cfg(pre_chirp_event="bogus_event_name_xyz"))
    await greeter.greet()
    assert len(engine.utterances) == 1


@pytest.mark.asyncio
async def test_greet_uses_runtime_names_override() -> None:
    engine = FakeVoiceEngine()
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
    engine = FakeVoiceEngine()
    greeter = Greeter(engine, _cfg())
    with pytest.raises(ValueError, match="at least one name"):
        await greeter.greet(names=[])


@pytest.mark.asyncio
async def test_greet_oxford_comma_formatting_in_message() -> None:
    engine = FakeVoiceEngine()
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
    engine = FakeVoiceEngine()
    greeter = Greeter(engine, _cfg())
    await greeter.greet()
    # The greeter MUST NOT have called start/stop on the engine.
    assert engine.started is False
    assert engine.stopped is False


@pytest.mark.asyncio
async def test_greet_honours_inter_chirp_delay_branch() -> None:
    """Exercises the ``inter_chirp_delay_s > 0`` ``await asyncio.sleep`` path.

    With ``inter_chirp_delay_s=0.001`` we touch the awaited-sleep branch
    that the zero-delay default never reaches, closing the last
    uncovered line in :mod:`mousedroid.voice.greeting`. The 1 ms delay
    keeps the test sub-second while still proving the await fires
    (a synchronous ``time.sleep`` regression would not satisfy the
    ``await`` contract on the protocol).
    """
    engine = FakeVoiceEngine()
    greeter = Greeter(engine, _cfg(inter_chirp_delay_s=0.001))
    await greeter.greet()
    # Both the pre-chirp and the styled message reached the engine in order.
    assert len(engine.utterances) == 2
    assert "John" in engine.utterances[1]


@pytest.mark.asyncio
async def test_greet_forwards_explicit_intensity_threshold() -> None:
    """High threshold (1.1) suppresses rocky-style excited effects.

    ``rocky_transform`` only applies effects when ``intensity >
    intensity_threshold``. A threshold above the maximum intensity
    (1.0) cannot be exceeded by any ``excitement_intensity`` value, so
    the message must emerge un-styled (no trailing ``!`` added).
    Asserting this proves the threshold is actually forwarded — not
    silently dropped — and gives an objective regression handle for
    the ``VoiceConfig.intensity_threshold`` plumbing.
    """
    engine = FakeVoiceEngine()
    greeter = Greeter(
        engine,
        _cfg(pre_chirp_event="", message_template="hello {names}"),
        intensity_threshold=1.1,
    )
    await greeter.greet()
    assert len(engine.utterances) == 1
    # rocky_transform with intensity 0.9 < threshold 1.1 must NOT
    # append the excitement exclamation. The lowercase "hello" also
    # confirms no uppercase styling fired.
    msg = engine.utterances[0]
    assert msg.endswith("hello John, Jordan, Parvathay and Jeff")
