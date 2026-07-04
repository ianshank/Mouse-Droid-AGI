"""Advisory size/drift guard for forward-looking planning docs (F-016).

NEXT_STEPS.md has a documented failure mode: landed work accretes as ✅ marks
until the file is a changelog (the 2026-07-03 reconciliation moved 37 KB /
72 ✅ into CHANGELOG.md). This guard keeps the drift visible without turning
prose edits into red PRs:

* default mode prints ``WARN:`` lines and **always exits 0** (the repo's
  report-only-script convention — there is deliberately no warn-in-pytest
  pattern here);
* ``--strict`` flips warnings into a non-zero exit for callers that want a
  hard gate (e.g. the regression test pins the post-reconciliation budget).

Thresholds live in the module constants below (single definition point) and
are operator-tunable per invocation via CLI flags — no other call site may
restate them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Single source for the default budgets. The post-reconciliation NEXT_STEPS.md
# is ~12 KB / 4 ✅; the caps leave headroom for normal growth while firing well
# before the observed 37 KB / 72 ✅ drift state.
_DEFAULT_MAX_BYTES = 20_000
_DEFAULT_MAX_DONE_MARKS = 10

_DONE_MARK = "✅"  # ✅ — landed-work marker that belongs in CHANGELOG.md


def check_doc(path: Path, *, max_bytes: int, max_done_marks: int) -> list[str]:
    """Return human-readable warnings for ``path`` (empty list == healthy).

    Pure helper — no printing, no exit codes — so tests and the regression
    suite consume the same logic the CLI does.
    """
    warnings: list[str] = []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"{path}: unreadable ({exc})"]

    size = len(raw)
    if size > max_bytes:
        warnings.append(
            f"{path}: {size} bytes exceeds the {max_bytes}-byte budget - "
            "move landed items to CHANGELOG.md (see F-016)"
        )

    done_marks = raw.decode("utf-8", errors="replace").count(_DONE_MARK)
    if done_marks > max_done_marks:
        warnings.append(
            f"{path}: {done_marks} '{_DONE_MARK}' marks exceed the "
            f"{max_done_marks}-mark budget - landed work belongs in CHANGELOG.md"
        )
    return warnings


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python tools/doc_hygiene.py",
        description="Advisory size/drift guard for planning docs (WARN-only unless --strict).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["NEXT_STEPS.md"],
        help="Doc paths to check (default: %(default)s).",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=_DEFAULT_MAX_BYTES,
        help="Byte budget per doc (default: %(default)s).",
    )
    parser.add_argument(
        "--max-done-marks",
        type=int,
        default=_DEFAULT_MAX_DONE_MARKS,
        help="Budget of landed-work marks per doc (default: %(default)s).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when any warning fires (default: advisory, always exit 0).",
    )
    args = parser.parse_args(argv)

    all_warnings: list[str] = []
    for raw_path in args.paths:
        all_warnings.extend(
            check_doc(
                Path(raw_path),
                max_bytes=args.max_bytes,
                max_done_marks=args.max_done_marks,
            )
        )

    for warning in all_warnings:
        print(f"WARN: {warning}")
    if not all_warnings:
        print("doc hygiene: all checked docs within budget")
    if args.strict and all_warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
