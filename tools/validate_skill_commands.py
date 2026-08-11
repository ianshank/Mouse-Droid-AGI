# tools/validate_skill_commands.py
"""Validate Claude Code skill files in either supported layout.

Reusable library + CLI. Two layouts are validated with the same per-file rules:
  * ``.claude/skills/<name>/SKILL.md`` — the current skills layout.
  * ``.claude/commands/*.md`` — the legacy flat slash-command layout, kept for
    any consumer mid-migration (``validate_all``); this repo migrated off it
    (foundry plan WS-F7a).

Checks, per skill file:
  * YAML front-matter parses and carries a non-empty ``description``.
  * Every backtick-wrapped repo path it references actually exists.
  * It contains no hardcoded IPv4 address (skills must stay environment-
    agnostic). Detection is deliberately IPv4-literal-only: example URLs and
    hostnames are legitimate in skill docs, so broadening to hostnames would
    only add false positives.

Paths are *discovered* from the body, never enumerated here, so the tool keeps
working as skills evolve. Format/glob tokens ({}, *, $, <>) are excluded so
illustrative patterns like ``weights/arm/{task}_final.pt`` are not false flags.

The CLI auto-discovers whichever layout(s) exist under ``<repo-root>/.claude``
and fails when NEITHER exists (a silent false "all valid" would mask a renamed
directory). Pass ``--skills-dir`` / ``--commands-dir`` to pin a layout
explicitly.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

# Capture any backtick-delimited token, then decide IN CODE whether it looks
# like a repo-relative file reference. Filtering in Python (not the regex) keeps
# the format/glob exclusion reachable + testable and catches partially-braced
# tokens a stricter regex would silently miss.
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_PATH_EXT_RE = re.compile(r".+\.(?:py|ya?ml|md|sh|pt|onnx|json|urdf|usd)$")
_FORBIDDEN_IN_PATH = set("{}*$<> ")
# IPv4-literal only by design — see module docstring. Hostnames/URLs in skill
# docs are legitimate, so broadening this would only produce false positives.
# Valid-octet grammar (0-255) with boundary guards, so dotted version/build
# strings never false-flag: 999.1.1.1 (octet >255), 1.20.300.4 / 470.82.01.1
# (4-part versions; leading-zero octets are not valid IPv4 grammar),
# 10.0.0.7.5 (5-part), and v1.2.3.4 build tags all pass clean, while
# sentence-final IPs ("...to 192.168.4.1.") and host prefixes
# (192.168.1.5.example.com) are still flagged. Accepted tradeoff: a
# zero-padded IP (192.168.001.5) is no longer flagged — invalid grammar is
# indistinguishable from a version string at this layer.
_IPV4_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_HARDCODED_HOST_RE = re.compile(rf"(?<![\w.])(?:{_IPV4_OCTET}\.){{3}}{_IPV4_OCTET}(?!\.?\d)")

# Optional front-matter lifecycle field. Absent = fine (external skill layouts
# like .github/skills/ never carry it); present = must be one of these, so a
# typo like ``status: forzen`` can't silently un-freeze a paused skill.
_ALLOWED_SKILL_STATUSES = frozenset({"active", "frozen", "deferred"})


@dataclass(frozen=True)
class SkillCommandIssue:
    """A single validation problem found in a skill file."""

    path: Path
    code: str
    detail: str


def find_hardcoded_hosts(text: str) -> list[str]:
    """Return every hardcoded IPv4 literal found in ``text``.

    Deliberately IPv4-literal-only (see module docstring): hostnames and
    example URLs are legitimate in docs, so broadening the pattern would only
    add false positives. Public so doc-hygiene tests outside the skill-command
    sweep (e.g. plan-document regression tests) reuse the exact same rule
    instead of duplicating the regex.
    """
    return _HARDCODED_HOST_RE.findall(text)


def referenced_repo_paths(text: str) -> list[str]:
    """Return backtick-wrapped repo-relative file references, excluding patterns.

    A token counts when it ends in a known source/config extension, contains a
    ``/`` (repo-relative, not a bare word), and carries no format/glob
    metacharacters — so illustrative patterns like ``weights/arm/{task}.pt`` are
    correctly skipped while real refs like ``config/foo.yaml`` are validated.
    """
    out: list[str] = []
    for m in _BACKTICK_RE.finditer(text):
        token = m.group(1).strip()
        if "/" not in token:
            continue
        if not _PATH_EXT_RE.match(token):
            continue
        if any(c in _FORBIDDEN_IN_PATH for c in token):
            continue
        out.append(token)
    return out


def _split_front_matter(text: str) -> tuple[dict[str, object] | None, str]:
    if not text.startswith("---"):
        return None, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None, text
    return (meta if isinstance(meta, dict) else None), parts[2]


def validate_command_skill(
    path: Path, *, repo_root: Path, text: str | None = None
) -> list[SkillCommandIssue]:
    """Validate one skill file; return a list of issues (empty == valid).

    A file that is not valid UTF-8 yields a single ``unreadable`` issue rather
    than raising ``UnicodeDecodeError`` — one corrupt skill must not abort the
    whole ``validate_all`` sweep (and the caller gets an actionable signal).
    ``utf-8-sig`` tolerates an editor-prepended BOM, which would otherwise make
    ``_split_front_matter`` miss a perfectly valid ``---`` fence.

    ``text`` lets a caller that already read the file (``validate_skills``)
    skip a second read+parse; omitted, the file is read here (legacy contract).
    """
    if text is None:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError) as exc:
            return [SkillCommandIssue(path, "unreadable", str(exc))]
    issues: list[SkillCommandIssue] = []

    meta, body = _split_front_matter(text)
    if meta is None:
        issues.append(
            SkillCommandIssue(path, "bad-front-matter", "missing/invalid YAML front-matter")
        )
        meta, body = {}, text

    description = str(meta.get("description", "")).strip()
    if not description:
        issues.append(
            SkillCommandIssue(path, "missing-description", "front-matter 'description' is empty")
        )

    if "status" in meta:
        status = str(meta.get("status", "")).strip()
        if status not in _ALLOWED_SKILL_STATUSES:
            issues.append(
                SkillCommandIssue(
                    path,
                    "invalid-status",
                    f"front-matter 'status' {status!r} not in {sorted(_ALLOWED_SKILL_STATUSES)}",
                )
            )

    repo_root_resolved = repo_root.resolve()
    for ref in referenced_repo_paths(body):
        # Skill docs promise *repo-relative* references only. Reject absolute
        # refs and parent-escaping traversals (``../../x.py``) BEFORE probing the
        # filesystem, so a ``.exists()`` check can never reach outside the repo.
        # An absolute POSIX ref (``/etc/passwd.yaml``) is flagged explicitly even
        # if it happens to resolve inside the repo. (A Windows-drive ``C:\…``
        # literal never reaches here: ``referenced_repo_paths`` only emits
        # ``/``-containing tokens, so a backslash drive path is filtered out at
        # tokenisation.)
        if Path(ref).is_absolute():
            issues.append(SkillCommandIssue(path, "non-relative-path", ref))
            continue
        try:
            candidate = (repo_root / ref).resolve()
        except (OSError, RuntimeError) as exc:
            # Symlink loop (RuntimeError) / permission or OS error from a
            # malformed or malicious ref must not crash the whole sweep.
            issues.append(SkillCommandIssue(path, "invalid-path", f"{ref}: {exc}"))
            continue
        if not candidate.is_relative_to(repo_root_resolved):
            issues.append(SkillCommandIssue(path, "non-relative-path", ref))
            continue
        if not candidate.exists():
            issues.append(SkillCommandIssue(path, "missing-path", ref))

    for host in find_hardcoded_hosts(body):
        issues.append(SkillCommandIssue(path, "hardcoded-host", host))

    return issues


def validate_all(commands_dir: Path, *, repo_root: Path) -> list[SkillCommandIssue]:
    """Validate every ``*.md`` skill in ``commands_dir``.

    A missing/renamed ``commands_dir`` yields a single ``missing-commands-dir``
    issue instead of relying on ``glob``'s silent empty result — the caller gets
    a deterministic, actionable signal rather than a false "all valid".
    """
    issues: list[SkillCommandIssue] = []
    if not commands_dir.is_dir():
        return [SkillCommandIssue(commands_dir, "missing-commands-dir", str(commands_dir))]
    for md in sorted(commands_dir.glob("*.md")):
        issues.extend(validate_command_skill(md, repo_root=repo_root))
    return issues


def validate_skills(skills_dir: Path, *, repo_root: Path) -> list[SkillCommandIssue]:
    """Validate every ``<name>/SKILL.md`` under ``skills_dir``.

    Same per-file rules as the legacy layout (``validate_command_skill`` is
    layout-agnostic), plus two structural checks:

    * A skill subdirectory without a ``SKILL.md`` yields ``missing-skill-file``
      (a half-migrated or typo'd skill must not silently vanish from the sweep).
    * A front-matter ``name`` that disagrees with its directory name yields
      ``name-dir-mismatch`` — the directory is what Claude Code namespaces by,
      so a mismatch means the skill answers to a name nobody invokes. ``name``
      stays optional; absent means "directory name", which is always consistent.

    A missing/renamed ``skills_dir`` yields a single ``missing-skills-dir``
    issue, mirroring ``validate_all``'s deterministic-signal contract.
    """
    if not skills_dir.is_dir():
        return [SkillCommandIssue(skills_dir, "missing-skills-dir", str(skills_dir))]
    issues: list[SkillCommandIssue] = []
    for sub in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
        skill_md = sub / "SKILL.md"
        if not skill_md.is_file():
            issues.append(SkillCommandIssue(sub, "missing-skill-file", f"{sub.name}/SKILL.md"))
            continue
        # Single read feeds both the per-file rules and the name check — no
        # second read whose decoding could diverge from the first. An
        # unreadable file short-circuits: a name verdict derived from corrupt
        # bytes would be noise on top of the real signal.
        try:
            text = skill_md.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError) as exc:
            issues.append(SkillCommandIssue(skill_md, "unreadable", str(exc)))
            continue
        issues.extend(validate_command_skill(skill_md, repo_root=repo_root, text=text))
        meta, _ = _split_front_matter(text)
        declared = str((meta or {}).get("name", "")).strip()
        if declared and declared != sub.name:
            issues.append(
                SkillCommandIssue(
                    skill_md, "name-dir-mismatch", f"front-matter name {declared!r} != {sub.name!r}"
                )
            )
    return issues


def validate_repo(
    repo_root: Path,
    *,
    skills_dir: Path | None = None,
    commands_dir: Path | None = None,
) -> list[SkillCommandIssue]:
    """Validate every skill layout present under ``repo_root``.

    Default behaviour (both dirs ``None``): sweep whichever of
    ``.claude/skills/`` and ``.claude/commands/`` exist; if NEITHER exists,
    return a single ``no-skill-layout`` issue rather than a false "all valid".

    Passing a dir explicitly SCOPES the sweep to exactly the dir(s) passed —
    a pinned layout is mandatory (its missing-dir issue surfaces) and the
    other layout is not auto-discovered, preserving the pre-auto-discovery
    CLI contract that ``--commands-dir X`` validates only ``X``.
    """
    if skills_dir is not None or commands_dir is not None:
        issues: list[SkillCommandIssue] = []
        if skills_dir is not None:
            issues.extend(validate_skills(skills_dir, repo_root=repo_root))
        if commands_dir is not None:
            issues.extend(validate_all(commands_dir, repo_root=repo_root))
        return issues

    default_skill_dirs = [repo_root / ".claude" / "skills", repo_root / ".agents" / "skills"]
    default_commands = repo_root / ".claude" / "commands"
    discovered: list[SkillCommandIssue] = []
    swept = False

    for default_skills in default_skill_dirs:
        if default_skills.is_dir():
            discovered.extend(validate_skills(default_skills, repo_root=repo_root))
            swept = True

    if default_commands.is_dir():
        discovered.extend(validate_all(default_commands, repo_root=repo_root))
        swept = True
    if not swept:
        discovered.append(
            SkillCommandIssue(
                repo_root / ".claude",
                "no-skill-layout",
                "neither .claude/skills/, .agents/skills/, nor .claude/commands/ exists",
            )
        )
    return discovered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--skills-dir", type=Path, default=None)
    parser.add_argument("--commands-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    repo_root: Path = args.repo_root.resolve()

    issues = validate_repo(
        repo_root,
        skills_dir=args.skills_dir.resolve() if args.skills_dir else None,
        commands_dir=args.commands_dir.resolve() if args.commands_dir else None,
    )
    # print in tools/ is already exempt from T20 via pyproject per-file-ignores
    # (pyproject.toml: "tools/**/*.py" = [..., "T20"]) — no inline noqa needed.
    for i in issues:
        print(f"{i.path}: [{i.code}] {i.detail}")
    if issues:
        print(f"FAIL: {len(issues)} skill-command issue(s)")
        return 1
    print("OK: all skill commands valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
