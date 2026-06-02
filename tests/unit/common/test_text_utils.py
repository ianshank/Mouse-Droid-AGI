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
