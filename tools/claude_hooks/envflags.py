"""Truthiness for environment-variable flags.

A presence check (``bool(os.environ.get(NAME))``) is the wrong test for a flag
that gates a safety decision: it treats ``NAME=0`` and ``NAME=false`` as *on*,
which is the opposite of what an operator writing them intends. Both the freeze
gate's override and the hook debug switch resolve through here so the two agree
and neither re-implements the rule.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Values accepted as "on". Anything else — including ``0``, ``false``, ``no``,
#: ``off``, the empty string, and whitespace — reads as off.
TRUTHY = frozenset({"1", "true", "yes", "on"})


def is_truthy(value: str | None) -> bool:
    """Return whether ``value`` reads as an enabled flag.

    Args:
        value: Raw environment value, possibly ``None``.

    Returns:
        ``True`` only for a recognised affirmative token, case-insensitively.
    """
    if value is None:
        return False
    return value.strip().lower() in TRUTHY


def env_flag(env: Mapping[str, str], name: str) -> bool:
    """Return whether the environment variable ``name`` is set to a truthy value.

    Args:
        env: Environment mapping to read.
        name: Variable name.

    Returns:
        ``True`` only when the variable is present *and* affirmative.
    """
    return is_truthy(env.get(name))
