#!/usr/bin/env python3
"""Render the full-validation SUMMARY.md from harness results (F-018).

Thin argparse shim over :mod:`mousedroid.validation.summary` — the bash
harness (``scripts/jetson_full_validation.sh``) dumps its ``RESULTS[]`` rows
to a pipe-delimited file and invokes this script; the pure rendering logic
lives under the coverage gate in ``src/``. On any failure the bash side falls
back to its inline table, so a python-less host still gets a summary.

Usage:
    python scripts/render_validation_summary.py \
        --results-file RUN_DIR/results.psv \
        --preflight-log RUN_DIR/phase2_preflight.log \
        --stamp 20260703T000000Z --repo /opt/mousedroid \
        --config config/jetson_production.yaml \
        --telemetry-url http://127.0.0.1:8080 \
        --out RUN_DIR/SUMMARY.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a repo checkout without installation (select_next.py shim).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousedroid.validation.summary import (
    extract_trend_block,
    parse_result_rows,
    render_summary,
)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python scripts/render_validation_summary.py",
        description="Render SUMMARY.md from pipe-delimited harness results.",
    )
    parser.add_argument(
        "--results-file", required=True, help="Pipe-delimited STATUS|name|note rows."
    )
    parser.add_argument(
        "--preflight-log",
        default=None,
        help="Phase-2 preflight log to mine for the --trend block (optional).",
    )
    parser.add_argument("--stamp", required=True, help="UTC run stamp.")
    parser.add_argument("--repo", required=True, help="Repo checkout path.")
    parser.add_argument("--config", required=True, help="Production config path.")
    parser.add_argument("--telemetry-url", required=True, help="Telemetry base URL.")
    parser.add_argument("--out", required=True, help="SUMMARY.md destination path.")
    args = parser.parse_args(argv)

    rows = parse_result_rows(
        Path(args.results_file).read_text(encoding="utf-8").splitlines(),
    )

    trend_block = None
    if args.preflight_log:
        try:
            trend_block = extract_trend_block(
                Path(args.preflight_log).read_text(encoding="utf-8"),
            )
        except OSError:
            trend_block = None  # absent log == no trend this run

    Path(args.out).write_text(
        render_summary(
            rows,
            stamp=args.stamp,
            repo=args.repo,
            config=args.config,
            telemetry_url=args.telemetry_url,
            trend_block=trend_block,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
