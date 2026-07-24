# Testing Guide

## Tiers (`tests/`)

unit · integration · property (Hypothesis) · regression (AQA / backwards-compat / golden) · e2e ·
hardware (`@pytest.mark.hardware`, rover-only) · performance · smoke.

## Run

```bash
pytest tests/                                          # all (auto-loads tests/conftest.py hardware mocks)
pytest tests/ --cov=src/mousedroid --cov-report=term-missing
pytest tests/unit/ -m "not hardware"                   # skip rover-only tests
bash scripts/ci.sh                                     # the full local gate
```

## Gates

- **85% branch coverage** — `scripts/check_branch_coverage.py` (changed-line gate).
- **config-compat** — `.github/workflows/config-compat.yml`: existing YAML overlays must load unchanged.
- **AQA regression** — schema-field hygiene + doc-contract pins (e.g. `tests/regression/test_portfolio_reframe_aqa.py`).
- **Complexity** — `ruff C901`, `max-complexity = 15` (ADR-014).

## Conventions

A new config field needs a Pydantic default **and** a regression/AQA test. Pick the next feature to work on
with the spec harness: `python scripts/select_next.py` (honours the dependency DAG in `features.yaml`; ADR-012).
