# Progress Log — MouseDroidAGI

Reverse chronological (newest on top). Set the date with `date +%F`; never copy a
literal date. Rotation: keep ~10 sessions; move older entries to
`progress-archive/YYYY-QN.md`. See HARNESS_SPEC.md §11.

## 2026-07-03 — Session 003
**Features worked:** F-015 (secret-scan gate, WS-0.4), F-016 (truth reconciliation, WS-1), F-017 (host-env durability, WS-3.1), F-018 (validation trend instrumentation, WS-4) — rev. B validation-first plan, PR #151.
**Status changes:** F-015..F-018 → done (branch provenance; swap to squash SHA post-merge).
**Structural changes:** F-015 — `.gitleaks.toml` (regex-only allowlist), advisory `gitleaks` CI job (docker-pinned, full-history; first run surfaced + allowlisted the synthetic telemetry-auth test token), guarded `scripts/ci.sh` stage, `docs/runbooks/secret-scanning.md`. F-016 — NEXT_STEPS.md 37 KB/72 ✅ → ~12 KB/4 ✅ (landed work → CHANGELOG "Historical record"; Claude-Code runbook → `docs/runbooks/claude-code-on-jetson.md`), arm arc PAUSED at T2 with unfreeze condition, Phase-5 vocabulary disambiguated across both NEXT_STEPS files, 3 skills frozen via validated `status:` front-matter (`validate_skill_commands.py` gained the `invalid-status` check), `tools/doc_hygiene.py` advisory guard wired into ci.sh.
**Validation evidence:** `python -m pytest tests/regression/test_secret_scan_gate.py tests/unit/tools/test_doc_hygiene.py tests/regression/test_next_steps_reconciled.py -q` green; `bash scripts/ci.sh` green pre-commit per commit.
**Next:** Commits 3-7 of the rev. B implementation (F-017..F-020 + bookkeeping).

## 2026-06-15 — Session 002
**Features worked:** Harness hardening (no catalog status changes).
**Status changes:** none.
**Structural changes:** Gap analysis + tech-debt resolution on `claude/harness-spec-template-g3inh7`. Extracted the harness enforcement logic (schema/DAG/provenance/tier-gating/selection) out of `scripts/` into a new importable, mypy-strict, dependency-injectable module `src/mousedroid/harness/spec.py`; reduced `scripts/validate.py` + `scripts/select_next.py` to thin CLI shims (identical CLI contract, now CWD-robust). Added `tests/unit/harness/test_spec.py` (28 tests, 100% on `spec.py`) so the harness guarantees fall under the 85% coverage gate. Updated `test_harness_spec_aqa.py` to import the single canonical `check_dag` (removed the duplicated cycle-detection DFS).
**ADRs:** Updated ADR-012 (importable-module consequence + scope).
**Validation evidence:** `python scripts/validate.py --tier fast` exits 0 (incl. from `/tmp`); `tests/unit/harness/ + regression` → 142 passed; `ruff check` + `ruff format --check` clean; `python -m mypy --strict src/mousedroid/harness/spec.py` clean; `coverage` on `spec.py` = 100%.
**Next:** Same post-merge provenance maintenance (F-001/F-003 branch refs → squash SHA). Confirm the full `scripts/ci.sh` coverage gate ≥85% in CI.

## 2026-06-14 — Session 001
**Features worked:** F-001, F-003 (introduced); F-002, F-004, F-005, F-006, F-007 (catalogued from existing subsystems); F-008 (seeded as todo).
**Status changes:** F-001/F-003 → done; F-002/F-004/F-005/F-006/F-007 catalogued as done with real landing-commit provenance; F-008 todo (hardware tier).
**Structural changes:** Bootstrapped the spec-driven harness — created `HARNESS_SPEC.md`, `features.yaml`, `features.schema.json`, `scripts/validate.py`, `scripts/select_next.py`, `scripts/init.sh`, `scripts/validations/F-001.sh`, `.github/workflows/harness.yml`, `tests/regression/test_harness_spec_aqa.py`. Added the harness fast-tier stage to `scripts/ci.sh`. Added `jsonschema` + `types-jsonschema` to the `[dev]` extra.
**ADRs:** Added ADR-012 (adopt HARNESS_SPEC v2.1; ADRs stay in `docs/architecture/`; `F-001` validated by a script not a recursive `--check`; split git-strictness by CI job).
**Validation evidence:** `python scripts/validate.py --tier fast` exits 0; `tests/regression/test_harness_spec_aqa.py` passes; `python scripts/select_next.py` resolves the DAG.
**Next:** Post-merge, replace the branch-name `implemented_in` on F-001/F-003 with the squash SHA so the nightly `--strict-git` job stays green on `main`. Grow the catalog as new features are specced.
