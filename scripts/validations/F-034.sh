#!/usr/bin/env bash
# F-034 — mlflow sqlite tracking default (ExperimentLoggerConfig.tracking_uri).
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

# The two dedicated F-034 files run in full; they need no [mlflow] extra --
# tracking_uri is a plain schema default pin and _resolve_tracking_uri's
# passthrough behavior needs no real mlflow client (see
# tests/unit/training/observability/test_mlflow_logger.py /
# tests/unit/factory/test_factory_observability.py for the [mlflow]-extra
# behavioural coverage, run separately by the mlflow-extras CI job since
# this repo's default test stage never installs that extra).
# Explicit `if ! ...` rather than `assert` — the Jetson Docker entrypoint sets
# PYTHONOPTIMIZE=1, which strips Python asserts (CLAUDE.md).
if ! "$PY_BIN" -m pytest \
      tests/regression/test_f034_mlflow_sqlite_aqa.py \
      tests/regression/test_f034_mlflow_sqlite_backwards_compat.py \
      --import-mode=importlib --no-cov -q; then
  echo "F-034 FAIL: mlflow sqlite tracking default is broken or a shipped config is affected" >&2
  exit 1
fi

echo "F-034 OK: mlflow tracking_uri defaults to sqlite, [mlflow] extras carry sqlalchemy+alembic, no shipped config affected"
