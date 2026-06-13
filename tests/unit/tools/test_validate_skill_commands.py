# tests/unit/tools/test_validate_skill_commands.py
"""Unit tests for the reusable command-skill validator."""

from __future__ import annotations

from pathlib import Path

from tools.validate_skill_commands import (
    referenced_repo_paths,
    validate_command_skill,
)


def _write(p: Path, body: str) -> Path:
    p.write_text(body, encoding="utf-8")
    return p


def test_valid_skill_has_no_issues(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / "src").mkdir()
    real = _write(repo / "src" / "thing.py", "x = 1\n")
    skill = _write(
        repo / "ok.md",
        "---\ndescription: Does a thing\n---\n\nUses `src/thing.py`.\n",
    )
    assert validate_command_skill(skill, repo_root=repo) == []
    assert real.exists()


def test_missing_description_is_flagged(tmp_path: Path) -> None:
    skill = _write(tmp_path / "bad.md", "---\nname: x\n---\nbody\n")
    issues = validate_command_skill(skill, repo_root=tmp_path)
    assert any(i.code == "missing-description" for i in issues)


def test_missing_referenced_path_is_flagged(tmp_path: Path) -> None:
    skill = _write(
        tmp_path / "bad.md",
        "---\ndescription: d\n---\nUses `config/nope.yaml`.\n",
    )
    issues = validate_command_skill(skill, repo_root=tmp_path)
    assert any(i.code == "missing-path" and "config/nope.yaml" in i.detail for i in issues)


def test_glob_and_format_tokens_are_ignored(tmp_path: Path) -> None:
    # Pattern paths are NOT real files and must not be flagged.
    skill = _write(
        tmp_path / "ok.md",
        "---\ndescription: d\n---\nOutput `weights/arm/{task}_{stage}_final.pt`.\n",
    )
    assert referenced_repo_paths(skill.read_text(encoding="utf-8")) == []
