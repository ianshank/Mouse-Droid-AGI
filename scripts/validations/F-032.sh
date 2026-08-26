#!/usr/bin/env bash
# F-032 — GCP observability wiring (CloudLoggingSink/CloudMetricsExporter/CloudFirestoreSync).
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

# Two dedicated F-032 files run in full. Two more F-032 assertions live
# inside the pre-existing F-029 egress files (extending them was the more
# surgical form of "extend" — see openspec bundle tasks.md) — targeted by
# node ID rather than running those files whole, per the F-031 review round's
# own finding: running a shared file whole couples this gate's diagnosis to
# unrelated pins (there, arm/HC-SR04; here, F-029's schema-default and
# env-lever pins), producing a misleading "F-032 FAIL" for an F-029 regression.
# Explicit `if ! ...` rather than `assert` — the Jetson Docker entrypoint sets
# PYTHONOPTIMIZE=1, which strips Python asserts (CLAUDE.md).
if ! "$PY_BIN" -m pytest \
      "tests/regression/test_gcp_egress_defaults_aqa.py::test_partial_gcp_block_builds_no_egress_component" \
      "tests/regression/test_gcp_egress_defaults_backwards_compat.py::test_twin_overlay_now_also_builds_its_logging_and_monitoring_sinks" \
      tests/regression/test_f032_cloud_wiring_aqa.py \
      tests/regression/test_f032_cloud_wiring_backwards_compat.py \
      --import-mode=importlib --no-cov -q; then
  echo "F-032 FAIL: GCP observability wiring (logging/monitoring/Firestore) is broken" >&2
  exit 1
fi

echo "F-032 OK: CloudLoggingSink/CloudMetricsExporter/CloudFirestoreSync wired, gated, and inert by default"
