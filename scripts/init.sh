#!/usr/bin/env bash
# Idempotent, fast baseline bootstrap for the spec-driven harness (HARNESS_SPEC.md §9/§14).
# Reaches a known-good state an agent session can build on: editable install with the
# dev toolchain, then a quick harness health check.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

PY="${MOUSEDROID_PYTHON:-python}"

# Editable install with the dev toolchain (pytest, ruff, mypy, jsonschema, ...).
# `-q` keeps the log readable; re-running is a near no-op once satisfied.
"$PY" -m pip install -q -e ".[dev]"

# Quick health check: the harness must parse + pass its own fast tier.
MOUSEDROID_MOCK_HARDWARE="${MOUSEDROID_MOCK_HARDWARE:-true}" \
  "$PY" scripts/validate.py --tier fast

echo "baseline ready"
