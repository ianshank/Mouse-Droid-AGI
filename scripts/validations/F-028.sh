#!/usr/bin/env bash
# F-028 — The functional / user-journey / security tiers reach a CI path.
#
# Two halves, because "the tests pass" and "the tests run in CI" are different
# claims and F-028 is about the second: (1) the three tiers actually pass, and
# (2) the wiring that makes them run is pinned. Half (2) is delegated to the
# TestOrphanTierWiring AQA class rather than re-implemented here — un-pinned
# wiring is exactly how the smoke tier silently ran in zero CI paths for months
# (see tests/regression/test_ci_gate_wiring_aqa.py).
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

# Honour MOUSEDROID_PYTHON as the other scripts do, then fall back to python3
# on systems that ship no bare `python`.
PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

# Explicit `if ! ...` rather than `assert` — the Jetson Docker entrypoint sets
# PYTHONOPTIMIZE=1, which strips Python asserts (CLAUDE.md).
if ! "$PY_BIN" -m pytest tests/functional tests/user_journey tests/security \
      -m "not hardware" --import-mode=importlib --no-cov -q; then
  echo "F-028 FAIL: the functional/user-journey/security tiers do not pass" >&2
  exit 1
fi

if ! "$PY_BIN" -m pytest \
      tests/regression/test_ci_gate_wiring_aqa.py::TestOrphanTierWiring \
      --import-mode=importlib --no-cov -q; then
  echo "F-028 FAIL: the CI wiring for those tiers is not pinned" >&2
  exit 1
fi

echo "F-028 OK: three tiers pass and their CI wiring is pinned"
