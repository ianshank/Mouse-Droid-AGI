#!/usr/bin/env bash
# F-040 — optional IMU fusion slot, packer width 6, default imu_dim=0.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

if ! "$PY_BIN" -m pytest \
      tests/regression/test_f040_aqa.py \
      tests/regression/test_f040_backwards_compat.py \
      tests/unit/world_model/test_observation_packer.py \
      tests/unit/world_model/test_encoder.py \
      tests/unit/world_model/test_onnx_io.py \
      tests/unit/test_constants.py \
      --import-mode=importlib --no-cov -q; then
  echo "F-040 FAIL: IMU fusion slot / packer width drifted" >&2
  exit 1
fi

echo "F-040 OK: imu_dim=0 default; slot 5; packer width 6"
