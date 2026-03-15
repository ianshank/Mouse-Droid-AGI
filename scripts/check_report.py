#!/usr/bin/env python
"""CI gate: check BDI accuracy in training_report.json.

Usage::

    python scripts/check_report.py \\
        --report training/results/training_report.json \\
        --phase bdi_accuracy \\
        --must-pass

Environment variable overrides:
    MOUSEDROID_BDI_TRAINING__ACCURACY_THRESHOLD  (default: 0.60)

Exit codes:
    0  — phase passed or was skipped
    1  — phase failed (accuracy below threshold)
    2  — report file not found or malformed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    """Entry point for CI accuracy gate."""
    parser = argparse.ArgumentParser(description="Check training report phase")
    parser.add_argument("--report", required=True, help="Path to training_report.json")
    parser.add_argument("--phase", required=True, help="Phase key to check (e.g. bdi_accuracy)")
    parser.add_argument(
        "--must-pass",
        action="store_true",
        help="Exit 1 if phase status is not 'pass'",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"ERROR: Report file not found: {report_path}", file=sys.stderr)
        return 2

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"ERROR: Cannot parse report: {exc}", file=sys.stderr)
        return 2

    phases = report.get("phases", {})
    phase_data = phases.get(args.phase)

    if phase_data is None:
        print(f"WARNING: Phase '{args.phase}' not found in report — skipping.")
        return 0

    status = phase_data.get("status", "unknown")
    accuracy = phase_data.get("accuracy")

    # Allow env-var threshold override
    threshold = float(
        os.environ.get("MOUSEDROID_BDI_TRAINING__ACCURACY_THRESHOLD", "0.60")
    )

    print(f"Phase: {args.phase}")
    print(f"  Status:    {status}")
    print(f"  Accuracy:  {accuracy}")
    print(f"  Threshold: {threshold}")

    if args.must_pass:
        if status == "fail":
            print(f"FAIL: Phase '{args.phase}' did not pass.", file=sys.stderr)
            return 1
        if accuracy is not None and float(accuracy) < threshold:
            print(
                f"FAIL: Accuracy {accuracy} < threshold {threshold}",
                file=sys.stderr,
            )
            return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
