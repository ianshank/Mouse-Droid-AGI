# tests/unit/tools/test_validate_skill_commands.py
"""Unit tests for the reusable command-skill validator."""

from __future__ import annotations

from pathlib import Path

from tools.validate_skill_commands import (
    referenced_repo_paths,
    validate_all,
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


def test_hardcoded_ipv4_is_flagged(tmp_path: Path) -> None:
    # An IPv4 literal in the body must be reported as ``hardcoded-host``.
    skill = _write(
        tmp_path / "bad.md",
        "---\ndescription: d\n---\nSSH to the rover at 192.168.1.5 first.\n",
    )
    issues = validate_command_skill(skill, repo_root=tmp_path)
    assert any(i.code == "hardcoded-host" and i.detail == "192.168.1.5" for i in issues)


def test_parent_escaping_reference_is_flagged(tmp_path: Path) -> None:
    # A ``..`` traversal that escapes the repo root must be flagged as
    # ``non-relative-path`` rather than silently probed on the host FS.
    repo = tmp_path / "repo"
    repo.mkdir()
    skill = _write(
        repo / "bad.md",
        "---\ndescription: d\n---\nReads `../../etc/passwd.yaml`.\n",
    )
    issues = validate_command_skill(skill, repo_root=repo)
    assert any(
        i.code == "non-relative-path" and i.detail == "../../etc/passwd.yaml" for i in issues
    )
    # And it must NOT be reported as a (host-probed) missing-path.
    assert not any(i.code == "missing-path" for i in issues)


def test_validate_all_missing_dir_is_handled(tmp_path: Path) -> None:
    # A non-existent commands dir yields a deterministic issue, not a crash or
    # a false "all valid" empty result.
    missing = tmp_path / "nope"
    issues = validate_all(missing, repo_root=tmp_path)
    assert [i.code for i in issues] == ["missing-commands-dir"]


def test_non_utf8_file_is_reported_not_raised(tmp_path: Path) -> None:
    # A skill file that is not valid UTF-8 must surface as an ``unreadable``
    # issue, not crash the sweep with UnicodeDecodeError.
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"\xff\xfe not utf-8 \x80\x81")
    issues = validate_command_skill(bad, repo_root=tmp_path)
    assert [i.code for i in issues] == ["unreadable"]


def test_validate_all_skips_unreadable_without_aborting(tmp_path: Path) -> None:
    # One corrupt file must not prevent the rest of the dir from validating.
    _write(
        tmp_path / "good.md",
        "---\ndescription: ok\n---\nNo refs here.\n",
    )
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x80")
    issues = validate_all(tmp_path, repo_root=tmp_path)
    codes = [i.code for i in issues]
    assert codes == ["unreadable"]  # good.md produced zero issues
