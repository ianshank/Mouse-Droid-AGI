"""Portability checks for workforce assets (invariant I-3).

Assets under ``.claude/`` are meant to be lifted into another repository or a
shared plugin unchanged, so they must not embed machine-specific absolute paths.
Host/IP literals are covered by the existing rule in
:mod:`tools.validate_skill_commands` — reused rather than reimplemented so the
policy lives in exactly one place.

The absolute-path rule targets *filesystem roots* specifically. It deliberately
does not flag URL paths, POSIX-looking glob patterns, or the ``/`` inside a
repo-relative path, all of which are legitimate in these assets.
"""

from __future__ import annotations

import re

#: Directory names that begin a machine-specific absolute POSIX path.
_ABSOLUTE_ROOTS = ("home", "Users", "root", "var", "opt", "mnt", "media", "srv", "etc")

#: POSIX absolute paths under a machine-specific root, not preceded by a scheme
#: separator (so ``https://example.com/etc/x`` is not flagged).
_POSIX_ABSOLUTE_RE = re.compile(
    r"(?<![\w:/])/(?:" + "|".join(_ABSOLUTE_ROOTS) + r")/[\w.\-/]+",
)

#: Windows drive-letter paths, e.g. ``C:\Users\me`` or ``C:/Users/me``.
_WINDOWS_ABSOLUTE_RE = re.compile(r"(?<![\w])[A-Za-z]:[\\/](?:[\w.\-]+[\\/]?)+")


def find_absolute_paths(text: str) -> list[str]:
    """Return machine-specific absolute filesystem paths found in ``text``.

    Args:
        text: Asset content to inspect.

    Returns:
        Every matched absolute path, in order of appearance. An empty list means
        the asset is portable by this rule.
    """
    matches = [match.group(0) for match in _POSIX_ABSOLUTE_RE.finditer(text)]
    matches.extend(match.group(0) for match in _WINDOWS_ABSOLUTE_RE.finditer(text))
    return matches
