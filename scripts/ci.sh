#!/bin/bash
set -euo pipefail

# Resolve Python deterministically to avoid user-site package drift on Windows.
if [[ -n "${MOUSEDROID_PYTHON:-}" ]]; then
    PYTHON_BIN="$MOUSEDROID_PYTHON"
elif [[ -x "./.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="./.venv/Scripts/python.exe"
elif [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "No Python interpreter found. Set MOUSEDROID_PYTHON or install Python." >&2
    exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Resolved Python is not executable: $PYTHON_BIN" >&2
    exit 2
fi

export PYTHONNOUSERSITE=1
export MOUSEDROID_MOCK_HARDWARE=true

echo "=== Python Environment ==="
"$PYTHON_BIN" -c "import sys,pydantic,pydantic_settings; print(f'python={sys.executable}'); print(f'pydantic={pydantic.__version__}'); print(f'pydantic_settings={pydantic_settings.__version__}')"

echo "=== Lint ==="
"$PYTHON_BIN" -m ruff check src/ tests/

echo "=== Format Check ==="
# Mirrors the `Format check` step in .github/workflows/ci.yml. The ruff
# version is pinned in pyproject.toml's [dev] extra to match CI exactly —
# bump both in the same change to avoid local/CI lint drift.
"$PYTHON_BIN" -m ruff format --check src/ tests/

echo "=== Type Check ==="
"$PYTHON_BIN" -m mypy src/ --strict --ignore-missing-imports

echo "=== Hardcoded Value Gate (changed lines) ==="
"$PYTHON_BIN" scripts/check_no_hardcoded_values.py

echo "=== Settings Identity Smoke Check ==="
"$PYTHON_BIN" scripts/check_settings_identity.py

echo "=== Unit + Property + Integration Tests (with coverage) ==="
"$PYTHON_BIN" -m pytest tests/unit tests/property tests/integration \
    --import-mode=importlib \
    -v --cov=src/mousedroid --cov-report=term-missing --cov-fail-under=85

echo "=== Performance Tests ==="
"$PYTHON_BIN" -m pytest tests/performance/ --import-mode=importlib -v

echo "=== Regression Tests ==="
"$PYTHON_BIN" -m pytest tests/regression/ --import-mode=importlib -v

echo "=== E2E Tests ==="
"$PYTHON_BIN" -m pytest tests/e2e/ --import-mode=importlib -v

echo "=== Branch Coverage Gate (changed files >= 85%) ==="
"$PYTHON_BIN" scripts/check_branch_coverage.py --min 85 \
    --tests tests/unit tests/property tests/integration

echo "=== Prometheus Rules Validation (promtool) ==="
if command -v promtool >/dev/null 2>&1; then
    promtool check rules config/prometheus/alerts.yml
else
    echo "promtool not on PATH - skipping Prometheus rule validation"
fi

echo "=== Health Check ==="
MOUSEDROID_MOCK_HARDWARE=true "$PYTHON_BIN" -m mousedroid.main --health-check

echo "=== All checks passed ==="
