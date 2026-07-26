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

- **85% line coverage** — `--cov-fail-under=85` over `src/mousedroid` repo-wide, plus
  `scripts/check_branch_coverage.py` for changed files. Both measure **lines**: coverage.py
  runs with `branch` unset, and the script's name refers to the *git* branch, not branch
  coverage. Do not describe this gate as branch coverage.
- **Workforce tooling coverage** — `tools/claude_hooks/` gets its own
  `--cov=tools/claude_hooks --cov-branch` stage in `scripts/ci.sh` **and the `local-gates`
  job in `ci.yml`**, because the repo-wide gate is scoped to `src/mousedroid` and cannot
  see `tools/`. Its line threshold comes from `coverage.tools_line_min` in
  `.claude/workforce.yaml`; the branch figure is reported but advisory until a baseline exists.
- **Smoke tier** — `tests/smoke/` (sub-10-second import/parse sanity) runs in the `test`
  CI job and as a `scripts/ci.sh` stage outside the `MOUSEDROID_CI_SLIM` skip. It
  previously ran in no CI path at all; the wiring is pinned by
  `tests/regression/test_ci_gate_wiring_aqa.py`.
- **Performance tier** — `tests/performance/` runs as an **advisory** CI job
  (`continue-on-error`, tracked in `.github/advisory_stages.yaml`): the latency budgets
  are Jetson-calibrated and shared runners are noisy. The instrumentation-overhead budget
  is relaxed on CI runners via `MOUSEDROID_INSTRUMENTATION_OVERHEAD_BUDGET`.
- **config-compat** — `.github/workflows/config-compat.yml`: existing YAML overlays must load unchanged.
- **AQA regression** — schema-field hygiene + doc-contract pins (e.g. `tests/regression/test_portfolio_reframe_aqa.py`,
  `tests/regression/test_claude_workforce_aqa.py`).
- **Complexity** — `ruff C901`, `max-complexity = 15` (ADR-014).
- **Type checking** — `mypy --strict` over `src/`, and separately over `tools/claude_hooks/`
  (the rest of `tools/` is not strict-clean yet).

## Conventions

A new config field needs a Pydantic default **and** a regression/AQA test. Pick the next feature to work on
with the spec harness: `python scripts/select_next.py` (honours the dependency DAG in `features.yaml`; ADR-012).
