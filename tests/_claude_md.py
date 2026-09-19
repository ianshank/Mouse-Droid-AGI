"""Shared helpers for reasoning about Claude Code instruction files.

Follows the repo's ``tests/_<name>.py`` shared-helper convention (cf.
``tests/_bash.py``, ``tests/_pyproject.py``, ``tests/_script_loader.py``).

**Why the import detection lives here rather than inline in a test.** Claude Code
expands ``@path`` imports in a ``CLAUDE.md``, but *"import parsing skips Markdown
code spans and fenced code blocks"* — so ``@AGENTS.md`` is an import and
``` `@AGENTS.md` ``` is not. A test that greps for the bare substring passes on a
file that only *mentions* the path, which is precisely the failure mode F-053
exists to prevent: a documentation file that nothing loads, guarded by a check
that cannot tell the difference.

That distinction is one regex with two exclusions, it is easy to get wrong, and
it is needed by more than one caller, so it is named once here.

**Why the file sets are discovered, not enumerated.** ``AGENTS.md`` and
``CLAUDE.md`` locations change as packages are added. A hardcoded roster silently
stops covering new directories, which is the same defect one level up.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

__all__ = [
    "AGENTS_MD",
    "AGENT_FACING_WHEEL_PATTERNS",
    "CLAUDE_MD",
    "discover_tracked",
    "imported_paths",
    "imports_target",
    "repo_root",
]

AGENTS_MD = "AGENTS.md"
CLAUDE_MD = "CLAUDE.md"

#: Filename globs that must stay out of the built wheel. Agent instructions are
#: internal; ``pyproject.toml``'s own comment records that without the exclusion
#: the wheel ships them to PyPI. Kept here so the wheel gate and any future
#: agent-facing filename share one list instead of drifting apart.
AGENT_FACING_WHEEL_PATTERNS: tuple[str, ...] = (
    "**/CLAUDE.md",
    "**/agent.md",
    "**/AGENTS.md",
)

# A fenced block, then an inline code span. Stripped in that order so a fence
# containing backticks cannot leave a stray span behind.
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_SPAN_RE = re.compile(r"`[^`\n]*`")

# Claude Code's import form: '@' followed by a path, at a word boundary. Matches
# the documented syntax (relative or absolute, resolved against the importing
# file) without trying to validate the path itself.
_IMPORT_RE = re.compile(r"(?<![\w`])@([A-Za-z0-9_./~@-]+)")


def repo_root() -> Path:
    """Repository root, resolved from this file rather than the cwd."""
    return Path(__file__).resolve().parents[1]


def _strip_code(text: str) -> str:
    """Remove fenced blocks and inline spans, where ``@path`` is not an import."""
    return _SPAN_RE.sub(" ", _FENCE_RE.sub(" ", text))


def imported_paths(markdown: str) -> tuple[str, ...]:
    """Every ``@path`` import in ``markdown``, in order, code excluded.

    Args:
        markdown: Raw file contents.

    Returns:
        The imported path strings. A path mentioned only inside a code span or a
        fenced block is absent, because Claude Code does not import those.
    """
    return tuple(_IMPORT_RE.findall(_strip_code(markdown)))


def imports_target(markdown: str, target: str) -> bool:
    """Whether ``markdown`` imports ``target`` as a real (non-code) import."""
    return target in imported_paths(markdown)


def discover_tracked(filename: str) -> tuple[Path, ...]:
    """Every git-tracked file named ``filename``, as repo-relative paths.

    Uses ``git ls-files`` rather than ``rglob`` so untracked scratch files and
    ignored trees cannot influence a gate. Sorted for deterministic failure
    messages.

    Args:
        filename: Exact basename to match, case-sensitively.

    Returns:
        Repo-relative paths, sorted.
    """
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", f"*{filename}", filename],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root(),
    )
    return tuple(
        sorted(
            Path(entry)
            for entry in completed.stdout.split("\0")
            if entry and Path(entry).name == filename
        )
    )
