# Progress Log — MouseDroidAGI

Reverse chronological (newest on top). Set the date with `date +%F`; never copy a
literal date. Rotation: keep ~10 sessions; move older entries to
`progress-archive/YYYY-QN.md`. See HARNESS_SPEC.md §11.

## 2026-06-14 — Session 001
**Features worked:** F-001, F-003 (introduced); F-002, F-004, F-005, F-006, F-007 (catalogued from existing subsystems); F-008 (seeded as todo).
**Status changes:** F-001/F-003 → done; F-002/F-004/F-005/F-006/F-007 catalogued as done with real landing-commit provenance; F-008 todo (hardware tier).
**Structural changes:** Bootstrapped the spec-driven harness — created `HARNESS_SPEC.md`, `features.yaml`, `features.schema.json`, `scripts/validate.py`, `scripts/select_next.py`, `scripts/init.sh`, `scripts/validations/F-001.sh`, `.github/workflows/harness.yml`, `tests/regression/test_harness_spec_aqa.py`. Added the harness fast-tier stage to `scripts/ci.sh`. Added `jsonschema` + `types-jsonschema` to the `[dev]` extra.
**ADRs:** Added ADR-012 (adopt HARNESS_SPEC v2.1; ADRs stay in `docs/architecture/`; `F-001` validated by a script not a recursive `--check`; split git-strictness by CI job).
**Validation evidence:** `python scripts/validate.py --tier fast` exits 0; `tests/regression/test_harness_spec_aqa.py` passes; `python scripts/select_next.py` resolves the DAG.
**Next:** Post-merge, replace the branch-name `implemented_in` on F-001/F-003 with the squash SHA so the nightly `--strict-git` job stays green on `main`. Grow the catalog as new features are specced.
