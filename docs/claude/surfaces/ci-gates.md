# CI/CD Gates & Quality Ladders Surface

> Machine-enforced continuous integration pipeline defined in `.github/workflows/ci.yml`.

## Pipeline Structure (12 Jobs)

The CI workflow executes across distinct stages to fail fast and enforce invariants:

1. **Stage 0: Fast Fail (Lint & Security)**
   - `actionlint` — Workflow YAML syntax and expression checks.
   - `lint` — `ruff check` and `ruff format --check` over `src/`, `tests/`, `tools/`, `scripts/`.
   - `secret-scan` — `gitleaks` repository secret scan.
   - `skills` — `tools/validate_skill_commands.py` validating `.claude/skills/`.
2. **Stage 1: Strict Typing & Fast Tests**
   - `typecheck` — `mypy --strict` on `src/` and `tools/claude_hooks/`.
   - `test-fast` — Unit, property, and integration tests with `--no-cov`.
   - `validate` — Spec-harness fast tier + no-hardcoded-values check.
3. **Stage 2: Comprehensive Coverage & Platform Matrix**
   - `test` — Multi-version matrix (Python 3.10, 3.11, 3.12) with full `--cov` (>= 90%).
   - `test-windows` — Advisory Windows test tier (Python 3.11, windows-latest).
   - `local-gates` — Dedicated `--cov=tools/claude_hooks` tools coverage gate.
4. **Stage 3: Regression & AQA**
   - `regression` — Regression test suite (`tests/regression/`), backwards compatibility, and AQA pins.
5. **Stage 4: Build & Packaging**
   - `package` — Wheel building and metadata validation.

## Advisory Stages & Promotion Ladder

Advisory stages are tracked in `.github/advisory_stages.yaml`. They run non-blocking checks to build a stable baseline before promotion to blocking gates:

- `vulture` — Dead code analysis.
- `test-windows` — Windows platform test matrix.
- `tools-branch-cov` — Workforce tools branch coverage.

## Local Replication

Run the authoritative superset locally before pushing:

```bash
bash scripts/ci.sh
```

Or use the Makefile ladder: `make gates`, `make test`, `make ci`.
