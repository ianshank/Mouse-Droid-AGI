# tests/unit/tools/test_validate_skill_commands.py
"""Unit tests for the reusable command-skill validator."""

from __future__ import annotations

from pathlib import Path

from tools.validate_skill_commands import (
    find_hardcoded_hosts,
    referenced_repo_paths,
    validate_all,
    validate_command_skill,
    validate_repo,
    validate_skills,
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


def test_absolute_reference_is_flagged(tmp_path: Path) -> None:
    # An absolute POSIX ref must be flagged ``non-relative-path`` explicitly —
    # even if it happens to resolve inside the repo — and never host-probed.
    repo = tmp_path / "repo"
    repo.mkdir()
    skill = _write(
        repo / "bad.md",
        "---\ndescription: d\n---\nReads `/etc/secret/config.yaml`.\n",
    )
    issues = validate_command_skill(skill, repo_root=repo)
    assert any(
        i.code == "non-relative-path" and i.detail == "/etc/secret/config.yaml" for i in issues
    )
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


def test_find_hardcoded_hosts_detects_ipv4_literals() -> None:
    # Public helper used by doc-hygiene tests beyond the skill sweep
    # (tests/regression/test_foundry_plan_doc.py) — same single rule.
    text = "Connect to 10.0.0.7 then fall back to 192.168.4.1.\n"
    assert find_hardcoded_hosts(text) == ["10.0.0.7", "192.168.4.1"]


def test_find_hardcoded_hosts_ignores_hostnames_and_semver() -> None:
    # Hostnames, URLs, and three-part versions are legitimate doc content;
    # the IPv4-literal-only rule must not flag them.
    text = "See https://docs.claude.com and pin numpy 2.5.0 or v0.1.0.\n"
    assert find_hardcoded_hosts(text) == []


def _write_skill(skills_dir: Path, name: str, body: str) -> Path:
    d = skills_dir / name
    d.mkdir(parents=True)
    return _write(d / "SKILL.md", body)


def test_validate_skills_valid_nested_layout(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / "src").mkdir()
    _write(repo / "src" / "thing.py", "x = 1\n")
    skills = repo / ".claude" / "skills"
    _write_skill(skills, "demo", "---\ndescription: Does a thing\n---\nUses `src/thing.py`.\n")
    assert validate_skills(skills, repo_root=repo) == []


def test_validate_skills_missing_dir_is_handled(tmp_path: Path) -> None:
    issues = validate_skills(tmp_path / "nope", repo_root=tmp_path)
    assert [i.code for i in issues] == ["missing-skills-dir"]


def test_validate_skills_flags_subdir_without_skill_md(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    (skills / "half-migrated").mkdir(parents=True)
    issues = validate_skills(skills, repo_root=tmp_path)
    assert [i.code for i in issues] == ["missing-skill-file"]


def test_validate_skills_flags_name_dir_mismatch(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    _write_skill(skills, "real-name", "---\nname: other-name\ndescription: d\n---\nbody\n")
    issues = validate_skills(skills, repo_root=tmp_path)
    assert [i.code for i in issues] == ["name-dir-mismatch"]


def test_validate_skills_accepts_matching_or_absent_name(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    _write_skill(skills, "named", "---\nname: named\ndescription: d\n---\nbody\n")
    _write_skill(skills, "unnamed", "---\ndescription: d\n---\nbody\n")
    assert validate_skills(skills, repo_root=tmp_path) == []


def test_validate_repo_sweeps_both_layouts(tmp_path: Path) -> None:
    # A consumer mid-migration has BOTH layouts; both must be swept.
    skills = tmp_path / ".claude" / "skills"
    commands = tmp_path / ".claude" / "commands"
    commands.mkdir(parents=True)
    _write_skill(skills, "ok", "---\ndescription: d\n---\nbody\n")
    _write(commands / "bad.md", "---\nname: x\n---\nbody\n")  # missing description
    issues = validate_repo(tmp_path)
    assert [i.code for i in issues] == ["missing-description"]


def test_validate_repo_neither_layout_is_an_error(tmp_path: Path) -> None:
    # No layout at all must be a deterministic failure, not a false "all valid".
    issues = validate_repo(tmp_path)
    assert [i.code for i in issues] == ["no-skill-layout"]


def test_validate_repo_explicit_dir_is_mandatory(tmp_path: Path) -> None:
    # Pinning a layout explicitly surfaces its missing-dir issue instead of
    # silently skipping it.
    issues = validate_repo(tmp_path, skills_dir=tmp_path / "gone")
    assert [i.code for i in issues] == ["missing-skills-dir"]


def test_validate_repo_explicit_dir_scopes_the_sweep(tmp_path: Path) -> None:
    # Pinning one layout must NOT auto-discover the other: --commands-dir X
    # validates only X (the pre-auto-discovery CLI contract). The repo's
    # default skills dir here contains an invalid skill that would surface if
    # the sweep leaked beyond the pinned dir.
    skills = tmp_path / ".claude" / "skills"
    _write_skill(skills, "leaky", "---\nname: x\n---\nno description\n")
    commands = tmp_path / "external-commands"
    commands.mkdir()
    _write(commands / "ok.md", "---\ndescription: d\n---\nbody\n")
    assert validate_repo(tmp_path, commands_dir=commands) == []


def test_validate_skills_unreadable_short_circuits_name_check(tmp_path: Path) -> None:
    # A non-UTF-8 SKILL.md yields exactly `unreadable` — never a
    # name-dir-mismatch verdict derived from replacement-char content.
    skills = tmp_path / "skills"
    d = skills / "corrupt"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_bytes(b"---\nname: other\n---\n\xff\xfe\x80")
    issues = validate_skills(skills, repo_root=tmp_path)
    assert [i.code for i in issues] == ["unreadable"]


def test_bom_prefixed_front_matter_is_parsed(tmp_path: Path) -> None:
    # An editor-prepended UTF-8 BOM must not make the front-matter fence
    # invisible (utf-8-sig read).
    skill = tmp_path / "bom.md"
    skill.write_bytes(b"\xef\xbb\xbf---\ndescription: d\n---\nbody\n")
    assert validate_command_skill(skill, repo_root=tmp_path) == []
