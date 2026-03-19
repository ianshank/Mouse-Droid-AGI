#!/bin/bash
set -euo pipefail
export MOUSEDROID_MOCK_HARDWARE=true

echo "=== Lint ==="
ruff check src/ tests/

echo "=== Type Check ==="
mypy src/ --strict --ignore-missing-imports

echo "=== Unit Tests ==="
python -m pytest tests/unit tests/property -v --cov=src/mousedroid --cov-report=term-missing

echo "=== Integration Tests ==="
python -m pytest tests/integration/ -v

echo "=== Performance Tests ==="
python -m pytest tests/performance/ -v

echo "=== Regression Tests ==="
python -m pytest tests/regression/ -v

echo "=== E2E Tests ==="
python -m pytest tests/e2e/ -v

echo "=== Coverage Gate ==="
python -m pytest tests/unit tests/property tests/integration \
    --cov=src/mousedroid --cov-report=term-missing --cov-fail-under=85

echo "=== Branch Coverage Gate (changed files >= 85%) ==="
python scripts/check_branch_coverage.py --min 85 \
    --tests tests/unit tests/property tests/integration

echo "=== Health Check ==="
MOUSEDROID_MOCK_HARDWARE=true python -m mousedroid.main --health-check

echo "=== All checks passed ==="
