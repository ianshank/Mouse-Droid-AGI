# tools/validate_skill_commands.py
"""Validate ``.claude/commands/*.md`` skill files.

Reusable library + CLI. Checks, per skill file:
  * YAML front-matter parses and carries a non-empty ``description``.
  * Every backtick-wrapped repo path it references actually exists.
  * It contains no hardcoded IPv4 address (skills must stay environment-
    agnostic). Detection is deliberately IPv4-literal-only: example URLs and
    hostnames are legitimate in skill docs, so broadening to hostnames would
    only add false positives.

Paths are *discovered* from the body, never enumerated here, so the tool keeps
working as skills evolve. Format/glob tokens ({}, *, $, <>) are excluded so
illustrative patterns like ``weights/arm/{task}_final.pt`` are not false flags.
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
_HARDCODED_HOST_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@dataclass(frozen=True)
class SkillCommandIssue:
    """A single validation problem found in a skill file."""

    path: Path
    code: str
    detail: str


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


def validate_command_skill(path: Path, *, repo_root: Path) -> list[SkillCommandIssue]:
    """Validate one skill file; return a list of issues (empty == valid).

    A file that is not valid UTF-8 yields a single ``unreadable`` issue rather
    than raising ``UnicodeDecodeError`` — one corrupt skill must not abort the
    whole ``validate_all`` sweep (and the caller gets an actionable signal).
    """
    try:
        text = path.read_text(encoding="utf-8")
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

    repo_root_resolved = repo_root.resolve()
    for ref in referenced_repo_paths(body):
        # Skill docs promise *repo-relative* references only. Reject absolute
        # paths (``/etc/passwd``, ``C:\…``) and parent-escaping traversals
        # (``../../x.py``) BEFORE probing the filesystem, so a `.exists()` check
        # can never reach outside the repo tree.
        candidate = (repo_root / ref).resolve()
        if not candidate.is_relative_to(repo_root_resolved):
            issues.append(SkillCommandIssue(path, "non-relative-path", ref))
            continue
        if not candidate.exists():
            issues.append(SkillCommandIssue(path, "missing-path", ref))

    for host in _HARDCODED_HOST_RE.findall(body):
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--commands-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    repo_root: Path = args.repo_root.resolve()
    commands_dir: Path = (args.commands_dir or repo_root / ".claude" / "commands").resolve()

    issues = validate_all(commands_dir, repo_root=repo_root)
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
