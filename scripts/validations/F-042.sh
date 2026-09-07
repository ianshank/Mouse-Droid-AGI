#!/usr/bin/env bash
# F-042 — enumerate factory/orchestrator branch-coverage exemptions.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

if ! "$PY_BIN" -m pytest \
      tests/regression/test_f042_aqa.py \
      tests/regression/test_f042_backwards_compat.py \
      tests/unit/scripts/test_check_branch_coverage_base_ref.py \
      --import-mode=importlib --no-cov -q; then
  echo "F-042 FAIL: branch-coverage allowlist drifted" >&2
  exit 1
fi

echo "F-042 OK: factory/orchestrator prefixes enumerated; algorithmic factory gated"
