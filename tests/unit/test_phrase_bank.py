"""Unit tests for the Rocky phrase bank."""

from __future__ import annotations

import pytest

from mousedroid.voice.phrase_bank import DEFAULT_PHRASES

# ---------------------------------------------------------------------------
# Expected event keys — must all be present in DEFAULT_PHRASES.
# Add new event names here as they are added to phrase_bank.py.
# ---------------------------------------------------------------------------

_EXPECTED_EVENTS = [
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
    "turn_left",
    "turn_right",
    "arrived",
    "battery_low_warn",
    "battery_critical",
    "llm_translation_ack",
    "llm_translation_failed",
    "greeting",
    "greeting_formal",
    "greeting_excited",
    "farewell",
    # Conversational vocabulary (LLM answer_query path).
    "query_received",
    "query_answered",
    "query_failed",
    "thinking",
    "acknowledge",
    "affirmative",
    "negative",
]


@pytest.mark.parametrize("event", _EXPECTED_EVENTS)
def test_event_has_phrases(event: str) -> None:
    """Every expected event is present in DEFAULT_PHRASES."""
    assert event in DEFAULT_PHRASES, f"Missing event key: {event!r}"


@pytest.mark.parametrize("event", _EXPECTED_EVENTS)
def test_event_phrases_is_non_empty_list(event: str) -> None:
    """Every event maps to a non-empty list."""
    phrases = DEFAULT_PHRASES[event]
    assert isinstance(phrases, list)
    assert len(phrases) > 0, f"Event {event!r} has no phrases"


@pytest.mark.parametrize("event", _EXPECTED_EVENTS)
def test_event_phrases_all_non_empty_strings(event: str) -> None:
    """Every phrase in each event list is a non-empty string."""
    for i, phrase in enumerate(DEFAULT_PHRASES[event]):
        assert isinstance(phrase, str), f"{event}[{i}]: expected str, got {type(phrase)}"
        assert phrase.strip(), f"{event}[{i}]: phrase is empty or whitespace-only"


def test_all_keys_are_expected() -> None:
    """No unexpected keys were added without updating this test."""
    unexpected = set(DEFAULT_PHRASES.keys()) - set(_EXPECTED_EVENTS)
    assert not unexpected, f"New events added to phrase_bank but not to test: {unexpected}"


def test_phrase_bank_minimum_size() -> None:
    """phrase bank has at least the minimum expected number of events."""
    assert len(DEFAULT_PHRASES) >= len(_EXPECTED_EVENTS)
