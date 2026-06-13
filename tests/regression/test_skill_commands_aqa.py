# tests/regression/test_skill_commands_aqa.py
"""AQA: .claude/commands skill-file hygiene.

Locks the contract a careless edit could break: every command skill carries a
non-empty description, references only paths that exist, and bakes in no host/IP.
Reuses the shared validator so the rule lives in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

from tools.validate_skill_commands import validate_all

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMANDS = _REPO_ROOT / ".claude" / "commands"


def test_all_command_skills_are_valid() -> None:
    issues = validate_all(_COMMANDS, repo_root=_REPO_ROOT)
    assert issues == [], "skill-command issues:\n" + "\n".join(
        f"  {i.path.name}: [{i.code}] {i.detail}" for i in issues
    )


def test_commands_dir_is_non_empty() -> None:
    # Guards against the validator silently passing on an empty/renamed dir.
    assert list(_COMMANDS.glob("*.md")), "no .claude/commands/*.md skills found"
