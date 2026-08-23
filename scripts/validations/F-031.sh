#!/usr/bin/env bash
# F-031 — AutonomousOrchestrator stays off the production path (ADR-016).
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

# Target only the F-031-owned test nodes, not the whole shared file — it
# also carries the pre-existing arm/ and HC-SR04 parked-subsystem cases
# (F-020, WS-8.2). Running the whole file would make an unrelated regression
# in either of those print a misleading "F-031 FAIL" diagnosis.
# Explicit `if ! ...` rather than `assert` — the Jetson Docker entrypoint sets
# PYTHONOPTIMIZE=1, which strips Python asserts (CLAUDE.md).
if ! "$PY_BIN" -m pytest \
      "tests/regression/test_import_graph_freeze.py::test_no_active_module_imports_autonomous_orchestrator_at_module_scope" \
      "tests/regression/test_import_graph_freeze.py::test_no_production_entrypoint_calls_the_autonomous_orchestrator_builder" \
      --import-mode=importlib --no-cov -q; then
  echo "F-031 FAIL: AutonomousOrchestrator's disposition (ADR-016) is not enforced" >&2
  exit 1
fi

ADR_FILE="docs/architecture/ADR-016-autonomous-orchestrator-disposition.md"

# features.yaml's F-031 verification list claims two things beyond mere file
# existence: the ADR is Accepted, and orchestrator/CLAUDE.md's forward
# reference resolves to it. A prior version of this script checked only
# `-f`, which would still pass with a Proposed ADR or a stale/missing
# reference — check both claims explicitly rather than let the catalog
# overstate what this gate covers.
if [ ! -f "$ADR_FILE" ]; then
  echo "F-031 FAIL: disposition ADR-016 is missing" >&2
  exit 1
fi

if ! grep -q '\*\*Status:\*\* Accepted' "$ADR_FILE"; then
  echo "F-031 FAIL: ADR-016 exists but is not marked Accepted" >&2
  exit 1
fi

if ! grep -q 'ADR-016-autonomous-orchestrator-disposition.md' \
      src/mousedroid/orchestrator/CLAUDE.md; then
  echo "F-031 FAIL: orchestrator/CLAUDE.md's forward reference does not resolve to ADR-016" >&2
  exit 1
fi

echo "F-031 OK: AutonomousOrchestrator stays off the production path, ADR-016 recorded"
