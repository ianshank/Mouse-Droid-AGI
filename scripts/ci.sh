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
"$PYTHON_BIN" -m ruff check src/ tests/ tools/

echo "=== Format Check ==="
# Mirrors the `Format check` step in .github/workflows/ci.yml. The ruff
# version is pinned in pyproject.toml's [dev] extra to match CI exactly —
# bump both in the same change to avoid local/CI lint drift.
"$PYTHON_BIN" -m ruff format --check src/ tests/ tools/

echo "=== Skill Validator (.claude/skills) ==="
# Fast standalone signal that every .claude/skills/<name>/SKILL.md skill
# carries a non-empty description, references only paths that exist, and
# bakes in no host/IP. The CLI auto-discovers the layout (legacy
# .claude/commands would also be swept if present). The PR gate is
# tests/regression/test_skill_commands_aqa.py; this is the quick local mirror.
"$PYTHON_BIN" tools/validate_skill_commands.py

echo "=== Claude Workforce Config (.claude/workforce.yaml) ==="
# Fast standalone signal that the workforce config parses and validates against
# WorkforceConfig (extra="forbid"), so a typo like `frozen_path` can't silently
# disable the freeze gate. The PR gate is
# tests/regression/test_claude_workforce_aqa.py; this is the quick local mirror.
"$PYTHON_BIN" -c "from tools.claude_hooks.config import load_config; \
c = load_config(); print(f'workforce config OK: freeze={c.freeze.feature_key} \
frozen_paths={len(c.freeze.frozen_paths)}')"

echo "=== Doc Hygiene (advisory) ==="
# WARN-only drift guard for the forward-looking planning doc (F-016). Exits 0
# unless --strict; the hard post-reconciliation budget is pinned by
# tests/regression/test_next_steps_reconciled.py in the regression stage.
"$PYTHON_BIN" tools/doc_hygiene.py NEXT_STEPS.md

echo "=== Type Check ==="
"$PYTHON_BIN" -m mypy src/ --strict --ignore-missing-imports

echo "=== Type Check (workforce hooks) ==="
# tools/ as a whole is not yet --strict clean, so this is scoped to the hook
# package: new governance code is held to the same bar as src/ from day one.
# --explicit-package-bases + MYPYPATH resolve tools.claude_hooks unambiguously
# (tools/ is a namespace package with no __init__.py).
MYPYPATH=. "$PYTHON_BIN" -m mypy tools/claude_hooks/ --strict --ignore-missing-imports \
    --explicit-package-bases

echo "=== Pillar Validation Dispatch (--dry-run) ==="
# Proves the validate_all_pillars CLI + dispatch table are importable
# and invokable on every CI run, even though full per-pillar checks
# live inside the pytest stages below. --dry-run keeps cost ~50ms.
"$PYTHON_BIN" -m mousedroid.cli.validate_pillars --dry-run

echo "=== Hardcoded Value Gate (changed lines) ==="
"$PYTHON_BIN" scripts/check_no_hardcoded_values.py

echo "=== Settings Identity Smoke Check ==="
"$PYTHON_BIN" scripts/check_settings_identity.py

echo "=== Unit + Property + Integration Tests (with coverage) ==="
"$PYTHON_BIN" -m pytest tests/unit tests/property tests/integration \
    -m "not hardware" \
    --import-mode=importlib \
    -v --cov=src/mousedroid --cov-report=term-missing --cov-fail-under=85

# Slim mode: on memory-constrained hosts (e.g., Jetson container Phase-1 after
# an OOM retry), skip the memory-heaviest pytest stages. Unit+property+
# integration+coverage (above) is preserved — that's the core signal. The
# skipped stages still run in jetson_full_validation.sh Phase 2 hardware
# pytest tier where the rover owns the peripherals.
if [[ "${MOUSEDROID_CI_SLIM:-0}" == "1" ]]; then
    echo "=== Performance/Regression/E2E stages SKIPPED (MOUSEDROID_CI_SLIM=1) ==="
else
    echo "=== Performance Tests ==="
    "$PYTHON_BIN" -m pytest tests/performance/ -m "not hardware" --import-mode=importlib -v

    echo "=== Regression Tests ==="
    "$PYTHON_BIN" -m pytest tests/regression/ -m "not hardware" --import-mode=importlib -v
fi

echo "=== Harness Spec Alignment (fast tier) ==="
# Spec-driven harness gate (HARNESS_SPEC.md §10 / ADR-012): validates
# features.yaml against the schema, checks DAG integrity, and runs every
# `done` feature's fast-tier validation_command. The standalone
# .github/workflows/harness.yml mirrors this; running it here keeps the
# harness green in the local full-CI loop too. Warn-only on git provenance
# (matches the push job); slow/hardware tiers are deferred.
"$PYTHON_BIN" scripts/validate.py --tier fast

if [[ "${MOUSEDROID_CI_SLIM:-0}" == "1" ]]; then
    echo "=== E2E Tests SKIPPED (MOUSEDROID_CI_SLIM=1) ==="
else
    echo "=== E2E Tests ==="
    "$PYTHON_BIN" -m pytest tests/e2e/ -m "not hardware" --import-mode=importlib -v
fi

echo "=== Workforce Hook Coverage (line gate + advisory branch) ==="
# The repository-wide gate measures src/mousedroid only and cannot see tools/,
# so governance code needs its own invocation or it would ship untested. The
# line threshold mirrors coverage.tools_line_min in .claude/workforce.yaml;
# --cov-branch reports branch coverage, which is advisory until a baseline
# exists (never claim a metric that isn't enforced).
WORKFORCE_COV_MIN="$("$PYTHON_BIN" -c "from tools.claude_hooks.config import load_config; \
print(load_config().coverage.tools_line_min)")"
"$PYTHON_BIN" -m pytest tests/unit/tools/claude_hooks \
    -m "not hardware" \
    --import-mode=importlib \
    -q -o addopts="" \
    --cov=tools/claude_hooks --cov-branch --cov-report=term-missing \
    --cov-fail-under="$WORKFORCE_COV_MIN"

echo "=== Branch Coverage Gate (changed files >= 85%) ==="
"$PYTHON_BIN" scripts/check_branch_coverage.py --min 85 \
    --tests tests/unit tests/property tests/integration

echo "=== Prometheus Rules Validation (promtool) ==="
if command -v promtool >/dev/null 2>&1; then
    promtool check rules config/prometheus/alerts.yml
else
    echo "promtool not on PATH - skipping Prometheus rule validation"
fi

echo "=== Dead-Code Audit (vulture, advisory) ==="
# Findings-only (F-020): reports to reports/dead_code/, never blocks. Skips
# cleanly when the optional vulture dep is absent.
if "$PYTHON_BIN" -c "import vulture" >/dev/null 2>&1; then
    "$PYTHON_BIN" scripts/dead_code_audit.py
else
    echo "vulture not installed - skipping dead-code audit (CI runs it in vulture-audit)"
fi

echo "=== Advisory Promotion-Lag Check ==="
"$PYTHON_BIN" scripts/check_advisory_promotions.py

echo "=== Secret Scan (gitleaks, advisory) ==="
if command -v gitleaks >/dev/null 2>&1; then
    gitleaks detect --source . --config .gitleaks.toml --redact --no-banner \
        || echo "WARN: gitleaks reported findings (advisory - see .gitleaks.toml header)"
else
    echo "gitleaks not on PATH - skipping secret scan (CI runs it in the gitleaks job)"
fi

echo "=== Health Check ==="
MOUSEDROID_MOCK_HARDWARE=true "$PYTHON_BIN" -m mousedroid.main --health-check

echo "=== All checks passed ==="
