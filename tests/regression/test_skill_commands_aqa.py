# tests/regression/test_skill_commands_aqa.py
"""AQA: .claude/skills skill-file hygiene.

Locks the contract a careless edit could break: every skill carries a
non-empty description, references only paths that exist, and bakes in no
host/IP. Reuses the shared validator so the rule lives in exactly one place.

This repo migrated `.claude/commands/*.md` -> `.claude/skills/<name>/SKILL.md`
(foundry plan WS-F7a); the legacy directory must stay deleted — dead
conventions get removed, not adapted.
"""

from __future__ import annotations

from pathlib import Path

from tools.validate_skill_commands import validate_repo, validate_skills

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKILLS = _REPO_ROOT / ".claude" / "skills"
_LEGACY_COMMANDS = _REPO_ROOT / ".claude" / "commands"


def test_all_skills_are_valid() -> None:
    issues = validate_skills(_SKILLS, repo_root=_REPO_ROOT)
    assert issues == [], "skill issues:\n" + "\n".join(
        f"  {i.path.name}: [{i.code}] {i.detail}" for i in issues
    )


def test_repo_sweep_finds_a_layout_and_is_clean() -> None:
    # The CLI-default sweep must discover the skills layout (never the
    # "no-skill-layout" false-valid guard) and report zero issues.
    issues = validate_repo(_REPO_ROOT)
    assert issues == [], "repo skill-sweep issues:\n" + "\n".join(
        f"  {i.path}: [{i.code}] {i.detail}" for i in issues
    )


def test_skills_dir_is_non_empty() -> None:
    # Guards against the validator silently passing on an empty/renamed dir.
    assert list(_SKILLS.glob("*/SKILL.md")), "no .claude/skills/*/SKILL.md skills found"


def test_legacy_commands_dir_stays_deleted() -> None:
    # WS-F7a deleted the legacy layout; resurrecting it would fork the skill
    # set across two conventions again — exactly the drift this repo left.
    assert not _LEGACY_COMMANDS.exists(), (
        ".claude/commands/ has been resurrected — add skills under "
        ".claude/skills/<name>/SKILL.md instead"
    )
