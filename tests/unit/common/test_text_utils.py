"""Unit tests for :func:`mousedroid.common.text_utils.format_names_oxford`.

The helper was relocated from ``voice/greeting.py`` to
``common/text_utils.py`` so non-voice subsystems (dialogue, narration,
HRI logs) can reuse it without pulling in the voice stack. These tests
pin the canonical behaviour at the new location; the legacy import in
``voice/greeting.py`` is a re-export and is exercised by
``tests/unit/voice/test_greeting.py``.
"""

from __future__ import annotations

import pytest
import structlog.testing

from mousedroid.common.text_utils import format_names_oxford


def test_empty_list_returns_empty_string() -> None:
    assert format_names_oxford([]) == ""


def test_single_name_returns_verbatim() -> None:
    assert format_names_oxford(["Alpha"]) == "Alpha"


def test_two_names_joined_with_and() -> None:
    assert format_names_oxford(["Alpha", "Bravo"]) == "Alpha and Bravo"


def test_three_names_uses_listing_comma() -> None:
    assert format_names_oxford(["Alpha", "Bravo", "Charlie"]) == "Alpha, Bravo and Charlie"


def test_four_names_uses_listing_comma() -> None:
    assert (
        format_names_oxford(["Alpha", "Bravo", "Charlie", "Delta"])
        == "Alpha, Bravo, Charlie and Delta"
    )


def test_whitespace_only_entries_are_stripped() -> None:
    assert format_names_oxford(["Alpha", "", "  ", "Bravo"]) == "Alpha and Bravo"


def test_per_entry_whitespace_is_trimmed() -> None:
    assert format_names_oxford(["  Alpha  ", "Bravo \t"]) == "Alpha and Bravo"


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        ([], ""),
        (["only"], "only"),
        (["a", "b"], "a and b"),
        (["a", "b", "c"], "a, b and c"),
        (["a", "b", "c", "d", "e"], "a, b, c, d and e"),
    ],
)
def test_parametric_shapes(names: list[str], expected: str) -> None:
    assert format_names_oxford(names) == expected


def test_re_export_via_voice_greeting_matches_canonical() -> None:
    """The ``voice/greeting.py`` re-export must be the same function object.

    Prevents drift: a future refactor that introduces a second copy in
    ``voice/greeting.py`` would silently shadow the canonical helper
    and let the two implementations diverge.
    """
    from mousedroid.voice.greeting import format_names_oxford as via_voice

    assert via_voice is format_names_oxford


def test_drops_none_entries_without_raising() -> None:
    """A ``None`` slipping in from sloppy YAML must NOT crash playback.

    Round-3 review (Gemini #2): the prior implementation called
    ``.strip()`` directly on each entry, raising ``AttributeError`` on
    a non-string. Now non-strings are dropped silently with a
    structured debug log so misconfigured overlays are diagnosable
    without taking down the greeter.
    """
    assert format_names_oxford(["Alpha", None, "Bravo"]) == "Alpha and Bravo"  # type: ignore[list-item]


def test_drops_numeric_entries_without_raising() -> None:
    assert format_names_oxford(["Alpha", 42, "Bravo"]) == "Alpha and Bravo"  # type: ignore[list-item]


def test_all_non_string_entries_yields_empty_string() -> None:
    """A pathological YAML with zero string entries returns empty cleanly."""
    assert format_names_oxford([None, 42, object()]) == ""  # type: ignore[list-item]


def test_dropped_entries_logged_as_structured_event() -> None:
    """The drop must surface as a ``format_names_dropped_non_string`` event.

    Uses ``structlog.testing.capture_logs`` — the project's standard
    pattern for asserting structured-log events without depending on
    stdlib-logging routing (see e.g.
    ``tests/unit/telemetry/test_failure_recorder.py``).
    """
    with structlog.testing.capture_logs() as logs:
        format_names_oxford(["Alpha", None, 7, "Bravo"])  # type: ignore[list-item]

    drop_events = [r for r in logs if r.get("event") == "format_names_dropped_non_string"]
    assert len(drop_events) == 1
    payload = drop_events[0]
    assert payload["count"] == 2
    # Order-independent on the contents; sorting is deterministic in the
    # production code (``sorted(set(dropped))``) so the set form here is
    # not a brittleness concern, just future-proofing against a sort
    # comparator change.
    assert set(payload["types"]) == {"NoneType", "int"}
