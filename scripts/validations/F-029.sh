#!/usr/bin/env bash
# F-029 — Every GCP off-device egress channel is opt-in.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

# Explicit `if ! ...` rather than `assert` — the Jetson Docker entrypoint sets
# PYTHONOPTIMIZE=1, which strips Python asserts (CLAUDE.md).
if ! "$PY_BIN" -m pytest \
      tests/regression/test_gcp_egress_defaults_aqa.py \
      tests/regression/test_gcp_egress_defaults_backwards_compat.py \
      --import-mode=importlib --no-cov -q; then
  echo "F-029 FAIL: GCP egress default-OFF contract is broken" >&2
  exit 1
fi

echo "F-029 OK: all GCP egress channels default OFF and opt-in still works"
