# CI/CD Gates & Quality Ladders Surface

> Machine-enforced continuous integration pipeline defined in `.github/workflows/ci.yml`.
> Stage numbers below are the file's own comments (`grep "# Stage" .github/workflows/ci.yml`);
> this doc mirrors them rather than defining a separate taxonomy that can drift from the source.

## Pipeline Structure (16 Jobs)

1. **Stage 0** — `actionlint`: workflow YAML syntax and expression checks.
2. **Stage 1** — `lint`: `ruff check` + `ruff format --check` over `src/`, `tests/`, `tools/`,
   `scripts/` (matrixed Python 3.10/3.11/3.12).
3. **Stage 1b** — `config-validate`: every YAML overlay under `config/` against its Pydantic
   schema.
4. **Stage 1c** — `usbc-config-gate`: USB-C discovery wiring assertion (`jetson_production`
   overlay).
5. **Stage 2** — `typecheck`: `mypy --strict` on `src/` and `tools/claude_hooks/` (matrixed).
6. **Stage 3** — `test`: full suite (unit/property/integration + regression/e2e + smoke tiers)
   with `--cov` (>= 90%), matrixed Python 3.10/3.11/3.12. This is the blocking job that also
   runs `tests/functional`, `tests/user_journey`, `tests/security` (F-028).
7. **Stage 3b** *(advisory)* — `performance`: Jetson-calibrated latency budgets, noisy on
   shared runners.
8. **Stage 3b-win** *(advisory)* — `test-windows`: cross-platform lint/typecheck/fast-test
   subset.
9. **Stage 3c** — `local-gates`: dedicated `--cov=tools/claude_hooks` coverage gate plus the
   deterministic `scripts/ci.sh`-only checks (settings identity, skill validator, doc hygiene,
   ratchet budgets, hardcoded-value gate, subsystem-boundary gate).
10. **Stage 4** — `prometheus-check`: metrics format + alert-rule validation via `promtool`
    (graceful skip if unavailable).
11. **Stage 4b** — `vla-extras`: VLA-only test suite (blocking, Tier C3.1).
12. **Stage 4b** *(advisory)* — `onnx-world-model-extras`: world-model ONNX test suite (Tier B2).
13. **Stage 4c** — `gitleaks`: repository secret scan — blocking since 2026-08-07 (34 days
    advisory beforehand, tracked promotion).
14. **Stage 4d** *(advisory)* — `vulture-audit`: dead-code audit (findings-only by design —
    Protocol/DI false positives).
15. **Stage 5** *(advisory)* — `security`: `pip-audit`.
16. **Stage 6** — `docker`: Dockerfile + docker-compose validation (needs `test` + `typecheck`).

There is no separate `skills`, `secret-scan`, `test-fast`, `validate`, `regression`, or
`package` job name — those checks run as steps inside the jobs above (e.g. the skill
validator is a `local-gates` step, not its own job).

## Advisory Stages & Promotion Ladder

5 jobs carry `continue-on-error: true` and are tracked in `.github/advisory_stages.yaml`
with a `promote_after_days` window — `scripts/check_advisory_promotions.py` warns when a
job is untracked or its window has lapsed:

- `performance` — since 2026-07-25, 90-day window.
- `test-windows` — since 2026-08-20, 30-day window.
- `onnx-world-model-extras` — since 2026-05-16, 180-day window (extended once).
- `vulture-audit` — since 2026-07-03, 90-day window.
- `security` — since 2026-07-25, 60-day window.

## Local Replication

Run the authoritative superset locally before pushing:

```bash
bash scripts/ci.sh
```

Or use the Makefile ladder: `make gates`, `make test`, `make ci`.
