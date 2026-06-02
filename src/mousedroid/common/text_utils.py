"""Platform-agnostic text formatting helpers.

Reusable string utilities that are intentionally NOT coupled to the
voice / dialogue / narration / HRI subsystems. Lives in
:mod:`mousedroid.common` so any layer can import them without pulling
in audio dependencies (numpy / phrase_bank). The module imports
``structlog`` lazily inside the one helper that needs it, keeping the
import cost minimal for callers that only need pure text helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

_log = get_logger(__name__)


def format_names_oxford(names: Sequence[Any]) -> str:
    """Format ``names`` as a comma-separated list with ``" and "`` before the final entry.

    Examples:
        ``[]`` → ``""``
        ``["A"]`` → ``"A"``
        ``["A", "B"]`` → ``"A and B"``
        ``["A", "B", "C"]`` → ``"A, B and C"``
        ``["A", "B", "C", "D"]`` → ``"A, B, C and D"``

    Note:
        This is the *listing comma* form, not the strict Oxford / serial
        comma (which would be ``"A, B, C, and D"``). The function name is
        preserved for backwards compatibility with PR #108; the behaviour
        is locked by ``tests/unit/voice/test_greeting.py`` and
        ``tests/unit/common/test_text_utils.py``. If the strict-Oxford
        form is needed later, add a separate ``format_names_serial``
        rather than changing this contract.

    Defensive typing (round-3 review, Gemini): the caller contract is
    a sequence of strings, but YAML / Pydantic boundaries can leak a
    ``None`` or numeric entry under sloppy operator config. Rather
    than crashing with ``AttributeError`` deep inside the voice loop,
    non-string entries are silently dropped here AND surfaced in a
    structured ``format_names_dropped_non_string`` debug log so the
    operator can spot the bad overlay without taking down playback.

    Args:
        names: Ordered list of names to format. Empty / whitespace-only
            entries are stripped. Non-string entries are dropped with a
            debug-log emit; this keeps the typed signature loose at the
            YAML edge while staying loud enough to diagnose.

    Returns:
        The joined display string.
    """
    cleaned: list[str] = []
    dropped: list[str] = []
    for raw in names:
        if not isinstance(raw, str):
            dropped.append(type(raw).__name__)
            continue
        stripped = raw.strip()
        if stripped:
            cleaned.append(stripped)
    if dropped:
        _log.debug(
            "format_names_dropped_non_string",
            count=len(dropped),
            types=sorted(set(dropped)),
        )
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])} and {cleaned[-1]}"


__all__ = ["format_names_oxford"]
