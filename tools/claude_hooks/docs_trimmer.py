# tools/claude_hooks/docs_trimmer.py
"""Validate root CLAUDE.md line count against WorkforceConfig.docs.core_max_lines.

Implements the active consumer for DocsConfig.core_max_lines (F-026 / F-024 Phase 6).
Ensures the root developer instructions file stays evergreen, scannable, and under budget
by pushing subsystem-scoped rules to nested per-directory CLAUDE.md files and cross-cutting
surfaces to docs/claude/surfaces/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tools.claude_hooks.config import load_config
from tools.claude_hooks.logging_setup import get_logger
from tools.claude_hooks.paths import resolve_repo_root

_log = get_logger("docs_trimmer")


def check_claude_md_lines(repo_root: Path | None = None) -> tuple[bool, int, int]:
    """Check whether root CLAUDE.md satisfies the core_max_lines budget.

    Args:
        repo_root: Optional repository root path. Discovered if None.

    Returns:
        Tuple of (is_under_budget, current_lines, max_allowed_lines).
    """
    root = repo_root or resolve_repo_root()
    cfg = load_config(repo_root=root)
    claude_md = root / "CLAUDE.md"

    if not claude_md.is_file():
        _log.error("claude_md_missing", path=str(claude_md))
        return False, 0, cfg.docs.core_max_lines

    text = claude_md.read_text(encoding="utf-8")
    lines = len(text.splitlines())
    max_lines = cfg.docs.core_max_lines

    is_valid = lines <= max_lines
    if not is_valid:
        _log.error(
            "claude_md_over_budget",
            current_lines=lines,
            max_lines=max_lines,
            excess=lines - max_lines,
        )
    else:
        _log.info(
            "claude_md_within_budget",
            current_lines=lines,
            max_lines=max_lines,
        )

    return is_valid, lines, max_lines


def main() -> int:
    """CLI entry point for CI and pre-commit checks."""
    root = resolve_repo_root()
    claude_md = root / "CLAUDE.md"
    if not claude_md.is_file():
        print(
            f"ERROR: Root CLAUDE.md not found at {claude_md}. "
            f"The core instructions file is required (DocsConfig.core_max_lines gate).",
            file=sys.stderr,
        )
        return 1

    is_valid, current_lines, max_lines = check_claude_md_lines(repo_root=root)
    if not is_valid:
        print(
            f"ERROR: Root CLAUDE.md has {current_lines} lines, exceeding the "
            f"budget of {max_lines} lines (DocsConfig.core_max_lines). "
            f"Delegate subsystem details to nested CLAUDE.md files.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
