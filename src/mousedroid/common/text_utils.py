"""Platform-agnostic text formatting helpers.

Reusable string utilities that are intentionally NOT coupled to the
voice / dialogue / narration / HRI subsystems. Lives in
:mod:`mousedroid.common` so any layer can import them without pulling
in audio dependencies (numpy / structlog / phrase_bank).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def format_names_oxford(names: Sequence[str]) -> str:
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

    Args:
        names: Ordered list of names to format. Empty / whitespace-only
            entries are stripped silently.

    Returns:
        The joined display string.
    """
    cleaned = [n.strip() for n in names if n.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])} and {cleaned[-1]}"


__all__ = ["format_names_oxford"]
