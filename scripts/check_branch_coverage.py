#!/usr/bin/env python3
"""Enforce coverage threshold for changed source files on the current branch.

This script is intended for CI and local branch validation. It detects changed
Python files under src/mousedroid from git status, runs pytest with coverage,
and fails if any changed source file falls below --min coverage.

Example:
    python scripts/check_branch_coverage.py --min 85 \
        --tests tests/unit tests/property tests/integration
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def _normalize_repo_path(path_part: str) -> str:
    """Normalize a git path to repo-relative POSIX form."""
    normalized = path_part.strip().replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return str(PurePosixPath(normalized))


def _is_target_source(path_part: str) -> bool:
    """Return True for Python files under src/mousedroid."""
    posix_path = PurePosixPath(path_part)
    return (
        len(posix_path.parts) >= 3
        and posix_path.parts[:2] == ("src", "mousedroid")
        and posix_path.suffix == ".py"
    )


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """De-duplicate while preserving input order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _local_dev_base_candidates() -> list[str]:
    """Return git base-ref candidates for local-dev use (no CI env / no CLI flag).

    Each candidate is probed via ``git rev-parse --verify`` downstream; the
    first one that resolves wins. PR-105b added this chain so the script's
    branch-coverage gate works locally without operators having to export
    ``GITHUB_BASE_REF`` manually (the gap that bit PR #104).

    Order of preference:

    1. Upstream-tracking branch (``git rev-parse --abbrev-ref @{u}``) —
       the branch the local one was created from, available whenever
       ``git push -u`` was used.
    2. ``origin/HEAD`` symbolic ref target — the remote's default branch
       as recorded at clone time (typically ``main`` or ``master``, but
       can be any branch the maintainer set as default).
    3. ``origin/main`` as a literal fallback for clones that lack
       ``origin/HEAD``.
    """
    candidates: list[str] = []

    upstream = _run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if upstream.returncode == 0:
        ref = upstream.stdout.strip()
        if ref:
            candidates.append(ref)

    origin_head = _run(["git", "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"])
    if origin_head.returncode == 0:
        ref = origin_head.stdout.strip()
        if ref:
            # ``refs/remotes/origin/main`` → ``origin/main`` for the diff command.
            short = ref.removeprefix("refs/remotes/")
            if short:
                candidates.append(short)

    candidates.append("origin/main")
    return _dedupe_keep_order(candidates)


def _git_base_candidates(base_ref: str | None) -> list[str]:
    """Return candidate git base refs from CLI / env / local-dev fallbacks.

    PR-105b extended the previous (CLI + env only) chain so a local
    invocation without ``--base-ref`` or ``GITHUB_BASE_REF`` still
    resolves to a meaningful diff base (upstream-tracking branch first,
    then ``origin/HEAD``, then ``origin/main``). Without this extension
    the script silently fell back to ``git status --porcelain`` working-
    tree diffs, which miss every committed-but-unpushed change — the
    failure mode that masked PR #104's per-file coverage data.
    """
    raw = (base_ref or os.environ.get("GITHUB_BASE_REF") or "").strip()

    candidates: list[str] = []
    if raw:
        candidates.append(raw)
        # In GitHub Actions ``GITHUB_BASE_REF`` is often a plain branch name.
        if "/" not in raw:
            candidates.insert(0, f"origin/{raw}")

    # Local-dev fallbacks only fire when nothing higher-priority resolved.
    # They are SAFE to enumerate even in CI — CI's CLI/env path resolves
    # first, so the fallbacks never run there.
    candidates.extend(_local_dev_base_candidates())
    return _dedupe_keep_order(candidates)


def _first_valid_base_ref(base_ref: str | None) -> str | None:
    """Resolve the first valid base ref available in the local clone.

    Emits an informational line on stderr identifying which candidate
    fired so the operator can see whether the gate is comparing against
    the expected base. Silent base-ref resolution was the PR-104 footgun.
    """
    candidates = _git_base_candidates(base_ref)
    for candidate in candidates:
        result = _run(["git", "rev-parse", "--verify", f"{candidate}^{{commit}}"])
        if result.returncode == 0:
            print(
                f"[check_branch_coverage] resolved base ref: {candidate}",
                file=sys.stderr,
            )
            return candidate
    if candidates:
        print(
            "[check_branch_coverage] no candidate base ref resolved; "
            f"tried: {', '.join(candidates)}",
            file=sys.stderr,
        )
    return None


def _changed_files_from_base(base_ref: str) -> list[str]:
    """Return changed source files from commit diff base_ref...HEAD."""
    result = _run(["git", "diff", "--name-only", f"{base_ref}...HEAD"])
    if result.returncode != 0:
        err = result.stderr.strip() or f"Failed running: git diff --name-only {base_ref}...HEAD"
        print(err, file=sys.stderr)
        return []

    files: list[str] = []
    for raw_path in result.stdout.splitlines():
        normalized = _normalize_repo_path(raw_path)
        if _is_target_source(normalized):
            files.append(normalized)
    return _dedupe_keep_order(files)


def _status_rows() -> list[tuple[str, str]]:
    """Parse git porcelain rows as (status, normalized_path)."""
    result = _run(["git", "status", "--porcelain"])
    if result.returncode != 0:
        print(result.stderr.strip() or "Failed to read git status", file=sys.stderr)
        sys.exit(2)

    rows: list[tuple[str, str]] = []
    for raw_line in result.stdout.splitlines():
        if not raw_line:
            continue
        status = raw_line[:2]
        path_part = raw_line[3:] if len(raw_line) > 3 else ""
        if " -> " in path_part:
            path_part = path_part.split(" -> ", maxsplit=1)[1]
        path_part = _normalize_repo_path(path_part)
        rows.append((status, path_part))
    return rows


def _changed_source_files(base_ref: str | None) -> list[str]:
    """Return changed Python files under src/mousedroid.

    In CI, prefer commit-based detection from ``git diff <base>...HEAD``.
    Fallback to local working-tree detection via ``git status --porcelain``.
    """
    resolved_base = _first_valid_base_ref(base_ref)
    if resolved_base is not None:
        from_diff = _changed_files_from_base(resolved_base)
        if from_diff:
            return from_diff

    rows = _status_rows()

    files: list[str] = []
    for _, path_part in rows:
        if _is_target_source(path_part):
            files.append(path_part)
    return _dedupe_keep_order(files)


def _path_to_module(file_path: str) -> str:
    """Convert src/mousedroid/foo/bar.py to mousedroid.foo.bar."""
    relative = _normalize_repo_path(file_path)
    rel_path = PurePosixPath(relative)
    if not _is_target_source(relative):
        msg = f"Unsupported source path: {file_path}"
        raise ValueError(msg)
    return ".".join(rel_path.with_suffix("").parts[1:])


def _build_pytest_command(tests: list[str], changed_files: list[str], json_out: Path) -> list[str]:
    # CI safeguard: shallow checkouts may have no base-ref, leading to zero files detected.
    # If running in CI and no files were found, fail explicitly rather than silently passing.
    if not changed_files and os.getenv("CI"):
        msg = (
            "No changed files detected in CI environment (possible shallow checkout). "
            "Cannot validate branch coverage."
        )
        raise RuntimeError(msg)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        # Re-add --import-mode=importlib explicitly: clearing addopts above
        # drops the project's default importmode, and falling back to pytest's
        # "prepend" mode triggers numpy's "cannot load module more than once
        # per process" error under coverage instrumentation. The main test
        # stage in scripts/ci.sh runs with importlib too — keep them aligned.
        "--import-mode=importlib",
        *tests,
        "-q",
        "--disable-warnings",
        f"--cov-report=json:{json_out.as_posix()}",
        "--cov-report=term",
        "--cov-fail-under=0",
    ]

    # Track only the directories containing changed files. We deliberately
    # avoid the dotted-module-name form (e.g. ``--cov=mousedroid.foo.bar``)
    # because coverage imports those modules before tests run, and that
    # import chain often hits ``numpy._core`` which raises
    # ``ImportError: cannot load module more than once per process`` under
    # numpy 2.x once instrumentation is active. Directory targets capture
    # the same data without forcing coverage to walk the import graph at
    # startup. ``_load_coverage`` filters back to per-file results, so the
    # broader sweep does not leak into the per-file gate.
    cov_dirs: list[str] = _dedupe_keep_order(
        [str(PurePosixPath(_normalize_repo_path(f)).parent) for f in changed_files]
    )
    for d in cov_dirs:
        cmd.append(f"--cov={d}")

    return cmd


def _parse_unified_zero(diff_text: str, line_map: dict[str, set[int]]) -> None:
    """Populate changed added/modified lines from unified diff text."""
    current_path: str | None = None
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            token = line[4:].strip()
            if token.startswith("b/"):
                token = token[2:]
            current_path = token.replace("\\", "/")
            continue

        match = hunk_re.match(line)
        if not match or current_path is None or current_path not in line_map:
            continue

        start = int(match.group(1))
        count = int(match.group(2) or "1")
        if count <= 0:
            continue
        line_map[current_path].update(range(start, start + count))


def _changed_line_map(changed_files: list[str], base_ref: str | None) -> dict[str, set[int]]:
    """Return changed line numbers for each changed source file."""
    line_map: dict[str, set[int]] = {p: set() for p in changed_files}

    resolved_base = _first_valid_base_ref(base_ref)
    if resolved_base is not None:
        result = _run(
            [
                "git",
                "diff",
                "--unified=0",
                f"{resolved_base}...HEAD",
                "--",
                "src/mousedroid",
            ]
        )
        if result.returncode != 0:
            err = result.stderr.strip() or (
                f"Failed running: git diff --unified=0 {resolved_base}...HEAD -- src/mousedroid"
            )
            print(err, file=sys.stderr)
            sys.exit(2)
        _parse_unified_zero(result.stdout, line_map)

    # Unstaged and staged modifications.
    for cmd in (
        ["git", "diff", "--unified=0", "--", "src/mousedroid"],
        ["git", "diff", "--cached", "--unified=0", "--", "src/mousedroid"],
    ):
        result = _run(cmd)
        if result.returncode != 0:
            print(result.stderr.strip() or f"Failed running: {' '.join(cmd)}", file=sys.stderr)
            sys.exit(2)
        _parse_unified_zero(result.stdout, line_map)

    # Untracked files: all lines are considered changed.
    for status, path_part in _status_rows():
        if status != "??" or path_part not in line_map:
            continue
        try:
            n_lines = len(Path(path_part).read_text(encoding="utf-8").splitlines())
        except OSError:
            n_lines = 0
        line_map[path_part].update(range(1, n_lines + 1))

    return line_map


def _load_coverage(json_out: Path) -> dict[str, dict[str, object]]:
    if not json_out.exists():
        print(f"Coverage JSON not found: {json_out}", file=sys.stderr)
        sys.exit(2)

    data = json.loads(json_out.read_text(encoding="utf-8"))
    files = data.get("files", {})

    coverage_by_path: dict[str, dict[str, object]] = {}
    for path, details in files.items():
        normalized = path.replace("\\", "/")
        summary = details.get("summary", {})
        coverage_by_path[normalized] = {
            "percent": float(summary.get("percent_covered", 0.0)),
            "executed": set(details.get("executed_lines", [])),
            "missing": set(details.get("missing_lines", [])),
        }
    return coverage_by_path


def main() -> int:
    """Run branch-level changed-file coverage check and enforce minimum percentage."""
    parser = argparse.ArgumentParser(description="Check coverage for changed branch files.")
    parser.add_argument("--min", type=float, default=85.0, dest="min_cov")
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["tests/unit", "tests/property", "tests/integration"],
        help="Pytest targets to run for branch coverage",
    )
    parser.add_argument(
        "--json",
        default="coverage-branch.json",
        help="Coverage JSON output file",
    )
    parser.add_argument(
        "--base-ref",
        default=None,
        help=(
            "Optional git base ref for commit-based change detection "
            "(defaults to GITHUB_BASE_REF in CI when set)"
        ),
    )
    args = parser.parse_args()

    # Resolve an effective base ref in the following order:
    #   1. --base-ref argument
    #   2. GITHUB_BASE_REF (e.g., for PRs on GitHub Actions)
    #   3. In CI, fall back to the previous commit (HEAD^) so push builds work
    base_ref = args.base_ref
    if base_ref is None:
        env_base = os.getenv("GITHUB_BASE_REF")
        if env_base:
            base_ref = env_base
    if base_ref is None and os.getenv("CI"):
        # CI-safe fallback: diff against the previous commit when no base ref is provided.
        rev_parse = _run(["git", "rev-parse", "--verify", "HEAD^"])
        if rev_parse.returncode == 0:
            candidate = rev_parse.stdout.strip()
            if candidate:
                base_ref = candidate

    changed_files = _changed_source_files(base_ref)
    if not changed_files:
        if os.getenv("CI"):
            print(
                "CI is set but no changed src/mousedroid Python files were detected; "
                "failing branch coverage check.",
                file=sys.stderr,
            )
            return 1
        print("No changed src/mousedroid Python files detected; skipping branch coverage check.")
        return 0

    json_out = Path(args.json)
    cmd = _build_pytest_command(args.tests, changed_files, json_out)
    print("Running branch coverage command:")
    print(" ".join(cmd))

    run_result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    # Always display the captured pytest output so developers can see what happened.
    sys.stdout.write(run_result.stdout)
    sys.stdout.flush()
    sys.stderr.write(run_result.stderr)
    sys.stderr.flush()
    if run_result.returncode != 0:
        # Narrow checks for known environment bugs under coverage.py instrumentation.
        # Require ALLOW_PYTEST_COLLECTION_SKIP=1 so CI blocks by default.
        combined = run_result.stdout + run_result.stderr
        allow_skip = os.environ.get("ALLOW_PYTEST_COLLECTION_SKIP") == "1"

        is_torch_coverage_bug = (
            "errors during collection" in combined.lower() and "_has_torch_function" in combined
        )
        # Pydantic Settings + coverage.py class identity mismatch: coverage
        # instrumentation can cause the Settings class to be loaded via two
        # different import paths, breaking isinstance checks inside
        # pydantic_settings __init__.  Manifests as "is_instance_of" errors
        # for Settings objects that work fine outside coverage mode.
        is_pydantic_coverage_bug = (
            "is_instance_of" in combined and "Settings" in combined and "--cov" in " ".join(cmd)
        )

        known_bug = is_torch_coverage_bug or is_pydantic_coverage_bug

        if known_bug and allow_skip:
            bug_names = []
            if is_torch_coverage_bug:
                bug_names.append("torch/coverage.py docstring collision")
            if is_pydantic_coverage_bug:
                bug_names.append("pydantic_settings/coverage.py class identity mismatch")
            print(
                f"\nWARNING: Known coverage-mode bug detected: {', '.join(bug_names)}.",
                file=sys.stderr,
            )
            print("Skipping branch coverage gate. Run tests manually to verify.", file=sys.stderr)
            return 0
        if known_bug:
            if is_torch_coverage_bug:
                print(
                    "\nERROR: torch/coverage.py collection conflict detected. "
                    "Set ALLOW_PYTEST_COLLECTION_SKIP=1 to skip, or fix the environment.",
                    file=sys.stderr,
                )
            if is_pydantic_coverage_bug:
                print(
                    "\nERROR: pydantic_settings/coverage.py class identity mismatch detected. "
                    "Set ALLOW_PYTEST_COLLECTION_SKIP=1 to skip, or fix the environment.",
                    file=sys.stderr,
                )
        return run_result.returncode

    coverage_by_path = _load_coverage(json_out)
    line_map = _changed_line_map(changed_files, base_ref)

    failures: list[tuple[str, float]] = []
    print("\nChanged-line coverage:")
    for file_path in changed_files:
        abs_path = str((Path.cwd() / file_path).resolve()).replace("\\", "/")
        rel_path = file_path.replace("\\", "/")

        # Coverage JSON may store absolute or relative keys depending on environment.
        info = coverage_by_path.get(abs_path, coverage_by_path.get(rel_path, {}))
        changed_lines = line_map.get(rel_path, set())

        executed = info.get("executed", set())
        missing = info.get("missing", set())
        if not isinstance(executed, set) or not isinstance(missing, set):
            executed = set()
            missing = set()

        coverable_changed = (executed | missing) & changed_lines
        if coverable_changed:
            pct = (len(executed & coverable_changed) / len(coverable_changed)) * 100.0
            scope = f"{len(coverable_changed)} changed executable lines"
        else:
            # No changed executable lines in this file: treat as not applicable
            # for the changed-line coverage gate by considering it fully covered.
            pct = 100.0
            scope = "no changed executable lines"

        print(f"  {rel_path}: {pct:.2f}% ({scope})")
        if pct < args.min_cov:
            failures.append((rel_path, pct))

    if failures:
        print(f"\nBranch coverage gate failed (min {args.min_cov:.2f}%):", file=sys.stderr)
        for rel_path, pct in failures:
            print(f"  - {rel_path}: {pct:.2f}%", file=sys.stderr)
        return 1

    print(f"\nBranch coverage gate passed (min {args.min_cov:.2f}%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
