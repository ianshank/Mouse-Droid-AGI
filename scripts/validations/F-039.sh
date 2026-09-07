#!/usr/bin/env bash
# F-039 — Cloud Logging queue + allowlist; tick_complete only on both-DEBUG.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

if ! "$PY_BIN" -m pytest \
      tests/unit/cloud/test_logging_sink.py \
      tests/unit/cloud/test_logging_sink_tick_level.py \
      tests/regression/test_f039_aqa.py \
      tests/regression/test_f039_backwards_compat.py \
      --import-mode=importlib --no-cov -q; then
  echo "F-039 FAIL: Cloud Logging queue / allowlist / tick-level pin drifted" >&2
  exit 1
fi

echo "F-039 OK: sink queues; allowlist redacts NL; INFO drops tick_complete"
