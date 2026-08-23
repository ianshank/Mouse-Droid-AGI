#!/usr/bin/env bash
# F-031 — AutonomousOrchestrator stays off the production path (ADR-016).
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

# Explicit `if ! ...` rather than `assert` — the Jetson Docker entrypoint sets
# PYTHONOPTIMIZE=1, which strips Python asserts (CLAUDE.md).
if ! "$PY_BIN" -m pytest \
      tests/regression/test_import_graph_freeze.py \
      --import-mode=importlib --no-cov -q; then
  echo "F-031 FAIL: AutonomousOrchestrator's disposition (ADR-016) is not enforced" >&2
  exit 1
fi

if [ ! -f docs/architecture/ADR-016-autonomous-orchestrator-disposition.md ]; then
  echo "F-031 FAIL: disposition ADR-016 is missing" >&2
  exit 1
fi

echo "F-031 OK: AutonomousOrchestrator stays off the production path, ADR-016 recorded"
