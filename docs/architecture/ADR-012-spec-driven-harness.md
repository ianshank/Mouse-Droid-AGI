# ADR-012 — Adopt the Spec-Driven Harness (HARNESS_SPEC v2.1)

> **Addendum (2026-07-03):** the catalog has grown past the seeded F-001–F-008;
> new entries continue from F-015 (F-009–F-014 are burned by the independent
> SMOKE_REPORT findings namespace) — see ADR-013.

* **Status:** Accepted
* **Date:** 2026-06-14
* **Owners:** Ian Cruickshank
* **Scope:** `HARNESS_SPEC.md`, `features.yaml`, `features.schema.json`,
  `src/mousedroid/harness/spec.py`, `scripts/validate.py`, `scripts/select_next.py`,
  `scripts/init.sh`, `scripts/validations/F-001.sh`, `.github/workflows/harness.yml`,
  `tests/unit/harness/test_spec.py`, `tests/regression/test_harness_spec_aqa.py`,
  `scripts/ci.sh` (harness stage), `progress.md`.

## Context

This codebase is developed largely by autonomous agents. The recurring failure
mode in that mode of work is **completeness-by-presence**: an agent infers a
feature is "done" because code exists, marks it complete, and the claim drifts
from reality with nothing to catch it. The project already has strong runtime
invariants (Protocol DI, factory wiring, no hardcoded values, `mypy --strict`,
85% coverage) enforced by `scripts/ci.sh` and the multi-job `ci.yml`, but it had
**no machine-checked source of truth for *what is supposed to exist and whether
it actually works*** — only prose plans under `docs/planning/` and `docs/superpowers/`.

The `PROJECT_HARNESS_SPEC_TEMPLATE.md` (Harness Engineering v2.1) provides a
reusable pattern that closes this gap: a schema-validated feature catalog
(`features.yaml`) plus a runner (`scripts/validate.py`) where a feature is
`done` only when its `validation_command` exits 0 — there is no hand-set
`passes` flag to game. Selection honours a dependency DAG; CI gates the whole
thing.

## Decision

Adopt the v2.1 harness as an **additive** spec-alignment layer on top of the
existing CI, not a replacement. Concretely:

* `features.yaml` is the source of truth, validated against
  `features.schema.json` and run by `scripts/validate.py` (schema + DAG
  integrity + git provenance + tier-gated command execution).
* `scripts/select_next.py` picks the next feature honouring `depends_on`.
* A standalone `.github/workflows/harness.yml` gates it: fast tier on every
  push/PR, fast+slow nightly. `scripts/ci.sh` also runs the fast tier so the
  harness stays green in the local full-CI loop.
* The catalog is seeded with 8 features ("bootstrap + key subsystems") mapping
  to **real, runnable** checks (config validation, skill AQA, LLM-gateway
  dispatch, validation import-decoupling, ten-pillar validation, USB-C rover
  smoke), so the harness validates genuine green checks from day one rather
  than placeholders.

### Deliberate deviations from the upstream template

1. **ADRs live in `docs/architecture/`, not `docs/decisions/`.** This project
   already maintains ADR-004…ADR-011 there; a second ADR root would fragment
   the rationale. `HARNESS_SPEC.md` §11 and the startup sequence point at
   `docs/architecture/`.
2. **`F-001` is validated by `scripts/validations/F-001.sh`, not
   `validate.py --check F-001`.** The template's own example is buggy: `--check
   F-001` runs F-001's `validation_command`, which would be `--check F-001`
   again — infinite recursion. The script is the non-recursive ground truth for
   "the harness exists and parses".
3. **CI installs the full `.[dev]` toolchain**, not just `pyyaml jsonschema`,
   because the seeded validation commands run real pytest suites.
4. **Git-provenance strictness is split by job.** The push/PR job runs
   *without* `--strict-git` (warn-only) because feature-branch `implemented_in`
   refs are brittle pre-merge; the nightly job (on `main`) runs `--strict-git`.

## Consequences

* **Enforcement logic is an importable, covered package module.** The schema /
  DAG / provenance / tier-gating / selection logic lives in
  `src/mousedroid/harness/spec.py` (pure, side-effect-free, dependency-injectable
  `runner`/`rev_checker`); `scripts/validate.py` and `scripts/select_next.py` are
  thin CLI shims. This mirrors the `cli/* → validation/*` split already used for
  preflight/pillars, brings the harness guarantees under the 85% coverage gate
  (`tests/unit/harness/test_spec.py`, 100% on `spec.py`) instead of leaving them
  untested in `scripts/`, and removes the duplicated cycle-detection DFS the AQA
  test previously carried. The shims also resolve commands/git against the repo
  root, so the harness runs correctly from any CWD.
* A feature claiming `done` whose command does not pass fails CI — false
  "green" is now structurally prevented for catalogued features.
* `jsonschema` (+ `types-jsonschema`) is added to the `[dev]` extra. The
  harness is a CI/agent tool surface, so it is dev-only — not a runtime
  dependency of the rover brain.
* **Post-merge maintenance:** harness-introduced features (`F-001`, `F-003`)
  currently set `implemented_in` to the working branch
  `claude/harness-spec-template-g3inh7`. After this PR merges, replace those
  with the squash/merge SHA so the nightly `--strict-git` job stays green on
  `main`. This is the same "update the provenance record" discipline ADR-010
  already established for `deployments/jetson-image.json`.
* The `hardware` tier (`F-008`, USB-C rover smoke) is never run in hosted CI;
  it is proven on the self-hosted Jetson runner via
  `python scripts/validate.py --tier hardware`.

## Related features

`F-001`–`F-008` (`features.yaml`).
