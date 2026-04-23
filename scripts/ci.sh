#!/bin/bash
set -euo pipefail
export MOUSEDROID_MOCK_HARDWARE=true

echo "=== Lint ==="
ruff check src/ tests/

echo "=== Type Check ==="
mypy src/ --strict --ignore-missing-imports

echo "=== Unit + Property + Integration Tests (with coverage) ==="
python -m pytest tests/unit tests/property tests/integration \
    -v --cov=src/mousedroid --cov-report=term-missing --cov-fail-under=85

echo "=== Performance Tests ==="
python -m pytest tests/performance/ -v

echo "=== Regression Tests ==="
python -m pytest tests/regression/ -v

echo "=== E2E Tests ==="
python -m pytest tests/e2e/ -v

echo "=== Branch Coverage Gate (changed files >= 85%) ==="
python scripts/check_branch_coverage.py --min 85 \
    --tests tests/unit tests/property tests/integration

echo "=== Prometheus Rules Validation (promtool) ==="
if command -v promtool >/dev/null 2>&1; then
    promtool check rules config/prometheus/alerts.yml
else
    echo "promtool not on PATH - skipping Prometheus rule validation"
fi

echo "=== Health Check ==="
MOUSEDROID_MOCK_HARDWARE=true python -m mousedroid.main --health-check

echo "=== All checks passed ==="
