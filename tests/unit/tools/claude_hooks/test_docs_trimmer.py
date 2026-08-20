# tests/unit/tools/claude_hooks/test_docs_trimmer.py
"""Unit tests for tools.claude_hooks.docs_trimmer (DocsConfig consumer)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from tools.claude_hooks.docs_trimmer import check_claude_md_lines, main


def test_check_claude_md_lines_under_budget(tmp_path: Path) -> None:
    """Returns True and correct line count when CLAUDE.md is within budget."""
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("Line 1\nLine 2\nLine 3\n", encoding="utf-8")

    # workforce.yaml with core_max_lines: 10
    workforce_yaml = tmp_path / ".claude" / "workforce.yaml"
    workforce_yaml.parent.mkdir(parents=True, exist_ok=True)
    workforce_yaml.write_text("docs:\n  core_max_lines: 10\n", encoding="utf-8")

    is_valid, count, max_lines = check_claude_md_lines(repo_root=tmp_path)
    assert is_valid is True
    assert count == 3
    assert max_lines == 10


def test_check_claude_md_lines_over_budget(tmp_path: Path) -> None:
    """Returns False and correct excess when CLAUDE.md exceeds budget."""
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("\n".join(f"Line {i}" for i in range(15)) + "\n", encoding="utf-8")

    workforce_yaml = tmp_path / ".claude" / "workforce.yaml"
    workforce_yaml.parent.mkdir(parents=True, exist_ok=True)
    workforce_yaml.write_text("docs:\n  core_max_lines: 10\n", encoding="utf-8")

    is_valid, count, max_lines = check_claude_md_lines(repo_root=tmp_path)
    assert is_valid is False
    assert count == 15
    assert max_lines == 10


def test_check_claude_md_lines_missing_file(tmp_path: Path) -> None:
    """Returns False and 0 count when CLAUDE.md does not exist."""
    is_valid, count, max_lines = check_claude_md_lines(repo_root=tmp_path)
    assert is_valid is False
    assert count == 0
    assert max_lines == 250


def test_main_cli_success(tmp_path: Path) -> None:
    """main() returns 0 when check passes."""
    with patch(
        "tools.claude_hooks.docs_trimmer.check_claude_md_lines",
        return_value=(True, 100, 250),
    ):
        assert main() == 0


def test_main_cli_failure(tmp_path: Path) -> None:
    """main() returns 1 and writes error message to stderr when check fails."""
    with patch(
        "tools.claude_hooks.docs_trimmer.check_claude_md_lines",
        return_value=(False, 300, 250),
    ):
        assert main() == 1
