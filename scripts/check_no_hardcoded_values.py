#!/usr/bin/env python3
"""Detect newly introduced hardcoded numeric literals in changed source lines.

This guard only inspects added/modified lines in changed Python files under
``src/mousedroid``. Existing debt is tolerated while preventing new hardcoded
runtime values from entering the codebase.

Unlike regex-only scans, this script parses Python AST and flags true numeric
literals only, which avoids false positives from comments, strings, and symbol
names.

Suppressions:
- Add ``# hardcoded-ok`` to a line when a literal is intentional.
- Add ``# noqa: PLR2004`` when the literal is semantically appropriate.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

TARGET_PREFIX = ("src", "mousedroid")
ALLOWED_FILES = {
    "src/mousedroid/constants.py",
}
# Directory-prefix exemptions for packages produced by a same-PR module
# split: config/schema.py -> config/schema/, telemetry/metrics.py ->
# telemetry/metrics/, telemetry/server.py -> telemetry/server/, and
# validation/runtime.py -> validation/runtime/. A 1-file-to-many split has
# no git rename correspondence, so every relocated line reads as newly
# "added" against the pre-split base — pre-existing debt that predates this
# gate ever running in CI, not a violation introduced by the split itself.
# config/schema/ additionally stays exempt on its own merits (every module
# there is, like the file it replaced, literally where runtime defaults are
# declared as Pydantic Field() literals). Growing this list beyond these
# four requires updating test_hardcoded_value_dir_exemptions_are_pinned —
# don't add an entry here to silence an unrelated finding.
ALLOWED_DIR_PREFIXES = (
    "src/mousedroid/config/schema/",
    "src/mousedroid/telemetry/metrics/",
    "src/mousedroid/telemetry/server/",
    "src/mousedroid/validation/runtime/",
)
ALLOWED_NUMERIC_VALUES = {0.0, 1.0, -1.0}
HUNK_PREFIX = "@@ "
SAFE_NUMERIC_CALLS = {
    "range",
    "enumerate",
}
SUPPRESSION_MARKERS = {
    "# hardcoded-ok",
    "# noqa: PLR2004",
}


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=True)


def _normalize(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return str(PurePosixPath(cleaned))


def _is_target_source(path: str) -> bool:
    posix_path = PurePosixPath(path)
    return (
        len(posix_path.parts) >= 3
        and posix_path.parts[:2] == TARGET_PREFIX
        and posix_path.suffix == ".py"
    )


def _changed_source_files() -> list[str]:
    commands = [
        ["git", "diff", "--name-only"],
        ["git", "diff", "--cached", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ]

    files: list[str] = []
    for command in commands:
        result = _run(command)
        if result.returncode != 0:
            err = result.stderr.strip() or f"Failed running: {' '.join(command)}"
            print(err, file=sys.stderr)
            sys.exit(2)

        for raw_path in result.stdout.splitlines():
            normalized = _normalize(raw_path)
            if _is_target_source(normalized) and normalized not in files:
                files.append(normalized)

    return files


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _git_base_candidates(base_ref: str | None) -> list[str]:
    raw = (base_ref or os.environ.get("GITHUB_BASE_REF") or "").strip()
    if not raw:
        return []
    candidates = [raw]
    if not raw.startswith("origin/"):
        candidates.insert(0, f"origin/{raw}")
    return _dedupe_keep_order(candidates)


def _first_valid_base_ref(base_ref: str | None) -> str | None:
    for candidate in _git_base_candidates(base_ref):
        result = _run(["git", "rev-parse", "--verify", f"{candidate}^{{commit}}"])
        if result.returncode == 0:
            return candidate
    return None


def _changed_files_from_base(base_ref: str) -> list[str]:
    result = _run(["git", "diff", "--name-only", f"{base_ref}...HEAD"])
    if result.returncode != 0:
        err = result.stderr.strip() or (f"Failed running: git diff --name-only {base_ref}...HEAD")
        print(err, file=sys.stderr)
        sys.exit(2)

    files: list[str] = []
    for raw_path in result.stdout.splitlines():
        normalized = _normalize(raw_path)
        if _is_target_source(normalized):
            files.append(normalized)
    return _dedupe_keep_order(files)


def _changed_source_files_from_range(base_ref: str | None) -> tuple[list[str], str | None]:
    """Return (changed_files, resolved_base) using commit-range detection.

    Resolved_base is ``None`` when no base ref could be resolved; callers may
    then fall back to working-tree detection.
    """
    resolved = _first_valid_base_ref(base_ref)
    if resolved is None:
        return ([], None)
    return (_changed_files_from_base(resolved), resolved)


def _parse_hunk_header(raw_line: str) -> int | None:
    if not raw_line.startswith(HUNK_PREFIX):
        return None

    parts = raw_line.split(" ")
    if len(parts) < 3:
        return None

    new_range = parts[2]
    if not new_range.startswith("+"):
        return None

    start_part = new_range[1:].split(",", maxsplit=1)[0]
    if not start_part.isdigit():
        return None
    return int(start_part)


def _parse_added_lines(diff_text: str) -> list[tuple[int, str]]:
    added_lines: list[tuple[int, str]] = []
    current_line = 0
    in_hunk = False

    for raw_line in diff_text.splitlines():
        hunk_start = _parse_hunk_header(raw_line)
        if hunk_start is not None:
            current_line = hunk_start
            in_hunk = True
            continue

        if not in_hunk:
            continue

        if raw_line.startswith("+++") or raw_line.startswith("---"):
            continue

        if raw_line.startswith("+"):
            added_lines.append((current_line, raw_line[1:]))
            current_line += 1
            continue

        if raw_line.startswith("-"):
            continue

        current_line += 1

    return added_lines


def _is_tracked(path: str) -> bool:
    result = _run(["git", "ls-files", "--error-unmatch", "--", path])
    return result.returncode == 0


def _added_lines_for_file(path: str, base_ref: str | None = None) -> list[tuple[int, str]]:
    added_lines: list[tuple[int, str]] = []

    if base_ref is not None:
        range_cmd = ["git", "diff", "--unified=0", f"{base_ref}...HEAD", "--", path]
        result = _run(range_cmd)
        if result.returncode != 0:
            err = result.stderr.strip() or f"Failed running: {' '.join(range_cmd)}"
            print(err, file=sys.stderr)
            sys.exit(2)
        added_lines.extend(_parse_added_lines(result.stdout))
    else:
        for command in (
            ["git", "diff", "--unified=0", "--", path],
            ["git", "diff", "--cached", "--unified=0", "--", path],
        ):
            result = _run(command)
            if result.returncode != 0:
                err = result.stderr.strip() or f"Failed running: {' '.join(command)}"
                print(err, file=sys.stderr)
                sys.exit(2)
            added_lines.extend(_parse_added_lines(result.stdout))

        if not _is_tracked(path):
            source_path = Path(path)
            if source_path.exists():
                for index, line in enumerate(
                    source_path.read_text(encoding="utf-8").splitlines(), 1
                ):
                    added_lines.append((index, line))

    deduped: dict[int, str] = {}
    for line_number, line_text in added_lines:
        deduped[line_number] = line_text
    return sorted(deduped.items(), key=lambda item: item[0])


def _line_is_ignored(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    if stripped.startswith("from ") or stripped.startswith("import "):
        return True
    if "Field(" in stripped:
        return True
    if any(marker in stripped for marker in SUPPRESSION_MARKERS):
        return True
    return "DEFAULT_" in stripped


def _is_numeric_constant(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _literal_from_node(node: ast.AST, parent_map: dict[int, ast.AST]) -> float | int | None:
    parent = parent_map.get(id(node))

    if isinstance(node, ast.Constant) and _is_numeric_constant(node.value):
        if isinstance(parent, ast.UnaryOp) and parent.operand is node:
            return None
        return node.value

    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.UAdd | ast.USub)
        and isinstance(node.operand, ast.Constant)
        and _is_numeric_constant(node.operand.value)
    ):
        base_value = float(node.operand.value)
        return -base_value if isinstance(node.op, ast.USub) else base_value

    return None


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_ignored_context(node: ast.AST, parent_map: dict[int, ast.AST]) -> bool:
    current = node
    while True:
        parent = parent_map.get(id(current))
        if parent is None:
            return False

        if isinstance(parent, ast.Slice):
            return True

        if isinstance(parent, ast.Subscript) and parent.slice is current:
            return True

        if isinstance(parent, ast.Call):
            call_name = _call_name(parent)
            if call_name in SAFE_NUMERIC_CALLS:
                return True

        current = parent


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parent_map: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[id(child)] = parent
    return parent_map


def _format_numeric_literal(value: float | int) -> str:
    int_value = int(value)
    if value == int_value:
        return str(int_value)
    return repr(value)


def _find_suspicious_literals(
    source: str,
    changed_lines: set[int],
) -> list[tuple[int, str]]:
    if not changed_lines:
        return []

    tree = ast.parse(source)
    parent_map = _build_parent_map(tree)
    source_lines = source.splitlines()

    findings: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        numeric_value = _literal_from_node(node, parent_map)
        if numeric_value is None:
            continue

        line_number = getattr(node, "lineno", None)
        if line_number is None or line_number not in changed_lines:
            continue

        line_text = source_lines[line_number - 1] if line_number - 1 < len(source_lines) else ""
        if _line_is_ignored(line_text):
            continue

        if float(numeric_value) in ALLOWED_NUMERIC_VALUES:
            continue

        if _is_ignored_context(node, parent_map):
            continue

        literal = _format_numeric_literal(numeric_value)
        key = (line_number, literal)
        if key in seen:
            continue

        seen.add(key)
        findings.append(key)

    findings.sort(key=lambda item: (item[0], item[1]))
    return findings


def _find_suspicious_literals_for_file(
    path: str,
    base_ref: str | None = None,
) -> list[tuple[int, str, str]]:
    changed_lines = {line_number for line_number, _ in _added_lines_for_file(path, base_ref)}
    if not changed_lines:
        return []

    source_path = Path(path)
    if not source_path.exists():
        return []

    source_text = source_path.read_text(encoding="utf-8")
    literals = _find_suspicious_literals(source_text, changed_lines)
    source_lines = source_text.splitlines()

    findings: list[tuple[int, str, str]] = []
    for line_number, literal in literals:
        line_text = (
            source_lines[line_number - 1].rstrip() if line_number - 1 < len(source_lines) else ""
        )
        findings.append((line_number, line_text, literal))
    return findings


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default=None,
        help=(
            "Base ref for commit-range detection (defaults to $GITHUB_BASE_REF). "
            "Falls back to working-tree diff when unresolvable outside CI."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    in_ci = os.getenv("CI", "").strip().lower() in {"1", "true", "yes"}

    range_files, resolved_base = _changed_source_files_from_range(args.base_ref)
    if resolved_base is not None:
        changed_files = range_files
        print(
            f"Hardcoded-value gate using commit-range base: {resolved_base}",
            file=sys.stderr,
        )
    else:
        if in_ci:
            print(
                "Hardcoded-value gate: base ref unresolved in CI environment "
                "(possible shallow checkout or missing GITHUB_BASE_REF).",
                file=sys.stderr,
            )
            return 2
        changed_files = _changed_source_files()

    if not changed_files:
        print("No changed src/mousedroid Python files detected; hardcoded-value gate skipped.")
        return 0

    findings: list[tuple[str, int, str, list[str]]] = []
    for file_path in changed_files:
        if file_path in ALLOWED_FILES or file_path.startswith(ALLOWED_DIR_PREFIXES):
            continue

        try:
            file_findings = _find_suspicious_literals_for_file(file_path, resolved_base)
        except SyntaxError as exc:
            lineno = exc.lineno or 1
            detail = exc.msg or "invalid syntax"
            print(f"Failed to parse {file_path}:{lineno}: {detail}", file=sys.stderr)
            return 2

        for line_number, line_text, literal in file_findings:
            findings.append((file_path, line_number, line_text, [literal]))

    if not findings:
        print("Hardcoded-value gate passed for changed source lines.")
        return 0

    print("Potential hardcoded runtime values detected in changed lines:")
    for file_path, line_number, line_text, literals in findings:
        literal_list = ", ".join(literals)
        print(f"- {file_path}:{line_number} -> [{literal_list}] {line_text}")

    print(
        "\nMove runtime-tunable values into config/constants or annotate intentional literals "
        "with '# hardcoded-ok' or '# noqa: PLR2004'."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
