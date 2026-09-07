#!/usr/bin/env bash
# F-041 — BDI Adam trainer + causal intention features; no HF publish.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

if ! "$PY_BIN" -m pytest \
      tests/regression/test_f041_aqa.py \
      tests/regression/test_f041_backwards_compat.py \
      tests/unit/training/test_bdi_training.py \
      tests/unit/training/test_collect_annotations.py \
      tests/unit/training/test_training_utils.py \
      --import-mode=importlib --no-cov -q; then
  echo "F-041 FAIL: BDI trainer honesty drifted" >&2
  exit 1
fi

echo "F-041 OK: Adam+He; causal intention features; publish bars; OTA is policy-v2"
