#!/usr/bin/env bash
# Run pytest on the given paths and fail if zero tests passed.
# Skip-all collections exit 0 — that is a false-green `done` and must not pass.
set -euo pipefail

FEATURE_ID="${1:?feature id}"
shift

cd "$(dirname "$0")/../.."

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

"$PY_BIN" - "$FEATURE_ID" "$@" <<'PY'
from __future__ import annotations

import re
import subprocess
import sys

feature_id = sys.argv[1]
pytest_args = sys.argv[2:]
cmd = [
    sys.executable,
    "-m",
    "pytest",
    *pytest_args,
    "--import-mode=importlib",
    "--no-cov",
    "-q",
]
proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
sys.stdout.write(proc.stdout)
sys.stderr.write(proc.stderr)
text = proc.stdout + proc.stderr
if proc.returncode != 0:
    sys.exit(proc.returncode)
match = re.search(r"(\d+) passed", text)
passed = int(match.group(1)) if match else 0
if passed < 1:
    print(
        f"{feature_id} FAIL: passed={passed} (skip-all is not done)",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"{feature_id} OK: {passed} passed")
PY
