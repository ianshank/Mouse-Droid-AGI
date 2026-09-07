#!/usr/bin/env bash
# F-038 — CI/docs honesty: parked-autonomous label + Current Next Steps pin + strict hygiene.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

if ! "$PY_BIN" -m pytest \
      tests/regression/test_f038_aqa.py \
      tests/regression/test_f038_backwards_compat.py \
      tests/regression/test_next_steps_reconciled.py \
      --import-mode=importlib --no-cov -q; then
  echo "F-038 FAIL: parked-autonomous label / Current Next Steps pin / hygiene gate drifted" >&2
  exit 1
fi

if ! "$PY_BIN" tools/doc_hygiene.py NEXT_STEPS.md --strict; then
  echo "F-038 FAIL: NEXT_STEPS.md is outside the strict hygiene budget" >&2
  exit 1
fi

echo "F-038 OK: parked-autonomous CI label; Current Next Steps has no LANDED; hygiene --strict"
