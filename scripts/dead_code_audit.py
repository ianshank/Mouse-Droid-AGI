#!/usr/bin/env python3
"""Advisory dead-code audit over the production tree (F-020, WS-8.1).

Runs vulture through its Python API (in-process, so the planted-defect tests
can exercise the pipeline without PATH games), writes a dated JSON report
under ``reports/dead_code/``, and prints a human summary. **Findings never
fail the build** — a Protocol/DI-heavy codebase produces false positives by
construction (protocol members, pydantic validators, factory hooks), which is
why the audit is findings-only and deletion decisions stay with a human
(rev. B WS-8 autonomy contract). ``--strict`` exists for operators who want a
hard gate on a curated allowlist.

The allowlist is a vulture whitelist file (``scripts/vulture_allowlist.py``
by default): plain Python attribute references that mark known-alive symbols.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Defaults (single definition point; every one is CLI-overridable).
_DEFAULT_PATHS = ("src/mousedroid",)
_DEFAULT_ALLOWLIST = "scripts/vulture_allowlist.py"
_DEFAULT_REPORT_DIR = "reports/dead_code"
_DEFAULT_MIN_CONFIDENCE = 60
_DEFAULT_MAX_PRINT = 20


def run_audit(
    paths: list[Path],
    *,
    allowlist: Path | None,
    min_confidence: int,
) -> list[dict[str, object]]:
    """Run vulture and return findings as JSON-ready dicts.

    Raises ImportError when vulture is unavailable — the CLI turns that into
    an advisory skip; tests use ``pytest.importorskip``.
    """
    import vulture

    scanner = vulture.Vulture()
    scavenge_paths = [str(p) for p in paths]
    if allowlist is not None and allowlist.is_file():
        scavenge_paths.append(str(allowlist))
    scanner.scavenge(scavenge_paths)
    return [
        {
            "filename": str(item.filename),
            "lineno": item.first_lineno,
            "name": str(item.name),
            "type": str(item.typ),
            "confidence": item.confidence,
        }
        for item in scanner.get_unused_code(min_confidence=min_confidence)
    ]


def write_report(findings: list[dict[str, object]], report_dir: Path) -> Path:
    """Persist findings as ``<report_dir>/<UTC date>.json`` and return the path."""
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = report_dir / f"{stamp}.json"
    out.write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python scripts/dead_code_audit.py",
        description="Advisory vulture dead-code audit (findings never block unless --strict).",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=list(_DEFAULT_PATHS),
        help="Paths to scan (default: %(default)s).",
    )
    parser.add_argument(
        "--allowlist",
        default=_DEFAULT_ALLOWLIST,
        help="Vulture whitelist file of known-alive symbols (default: %(default)s).",
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=_DEFAULT_MIN_CONFIDENCE,
        help="Vulture confidence floor 0-100 (default: %(default)s).",
    )
    parser.add_argument(
        "--report-dir",
        default=_DEFAULT_REPORT_DIR,
        help="Report output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=_DEFAULT_MAX_PRINT,
        help="Console lines before truncating (full list always in the JSON report; default: %(default)s).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when findings remain (default: advisory, always exit 0).",
    )
    args = parser.parse_args(argv)

    try:
        findings = run_audit(
            [Path(p) for p in args.paths],
            allowlist=Path(args.allowlist) if args.allowlist else None,
            min_confidence=args.min_confidence,
        )
    except ImportError:
        print("dead-code audit: vulture not installed - skipping (advisory)")
        return 0

    report_path = write_report(findings, Path(args.report_dir))
    if findings:
        print(f"dead-code audit: {len(findings)} finding(s) -> {report_path}")
        for f in findings[: args.max_print]:
            print(f"  {f['filename']}:{f['lineno']} {f['type']} '{f['name']}' ({f['confidence']}%)")
        if len(findings) > args.max_print:
            print(f"  ... {len(findings) - args.max_print} more in {report_path}")
        print(
            "Findings are ADVISORY: verify against Protocol/DI/pydantic usage "
            "before deleting; add known-alive symbols to the allowlist."
        )
    else:
        print(f"dead-code audit: clean -> {report_path}")
    if args.strict and findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
