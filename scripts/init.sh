#!/usr/bin/env bash
# Idempotent, fast baseline bootstrap for the spec-driven harness (HARNESS_SPEC.md §9/§14).
# Reaches a known-good state an agent session can build on: editable install with the
# dev toolchain, then a quick harness health check.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# Prefer the project venv (matches scripts/ci.sh) so init.sh never pollutes the
# global interpreter; honour an explicit MOUSEDROID_PYTHON override first.
if [[ -n "${MOUSEDROID_PYTHON:-}" ]]; then
    PY="$MOUSEDROID_PYTHON"
elif [[ -x "./.venv/Scripts/python.exe" ]]; then
    PY="./.venv/Scripts/python.exe"
elif [[ -x "./.venv/bin/python" ]]; then
    PY="./.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
else
    echo "No Python interpreter found. Set MOUSEDROID_PYTHON or install Python." >&2
    exit 2
fi

# Editable install with the dev toolchain (pytest, ruff, mypy, jsonschema, ...).
# `-q` keeps the log readable; re-running is a near no-op once satisfied.
"$PY" -m pip install -q -e ".[dev]"

# Quick health check: the harness must parse + pass its own fast tier.
MOUSEDROID_MOCK_HARDWARE="${MOUSEDROID_MOCK_HARDWARE:-true}" \
  "$PY" scripts/validate.py --tier fast

echo "baseline ready"
