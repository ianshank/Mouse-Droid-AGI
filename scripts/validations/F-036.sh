#!/usr/bin/env bash
# F-036 — stock FEEDBACK_BASE_INFO IMU parse without RSSM slot widen.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

# Explicit `if ! ...` rather than `assert` — Jetson PYTHONOPTIMIZE=1 strips
# Python asserts (CLAUDE.md).
if ! "$PY_BIN" -m pytest \
      tests/regression/test_f036_aqa.py \
      tests/regression/test_f036_backwards_compat.py \
      tests/unit/comms/test_comms_utils.py \
      tests/unit/sensing/test_sensor_manager.py \
      tests/unit/common/test_motor_tools.py \
      --import-mode=importlib --no-cov -q; then
  echo "F-036 FAIL: stock IMU parse / sensing consumer / legacy zeros drifted" >&2
  exit 1
fi

echo "F-036 OK: stock T=1001 r/p/y parsed; motor_state_dim=4; legacy IMU-inert"
