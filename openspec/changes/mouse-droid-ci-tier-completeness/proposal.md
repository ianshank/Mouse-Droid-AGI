# Proposal — CI tier completeness

- change_id: mouse-droid-ci-tier-completeness
- project: mouse-droid
- status: implemented
- feature_id: F-028
- epic: Quality Gates
- owner: ianshank
- created: 2026-08-22
- basis_commit: 175606b
- rev: A

## Why

Three test tiers ran in **zero** CI paths. `tests/functional/`,
`tests/user_journey/` and `tests/security/` appear in none of
`scripts/ci.sh`, `.github/workflows/ci.yml`, or the `Makefile`, so they executed
only under a bare `pytest` invocation nobody runs. A whole tier could rot
invisibly — precisely the failure mode the smoke tier suffered before PR #178,
recorded in `tests/regression/test_ci_gate_wiring_aqa.py`.

This is governance-relevant, not merely hygiene — but the risk is a **wiring**
gap, not a coverage hole, and the distinction matters. `tests/security/` holds
the only coverage of the pre-egress `RegexInjectionFilter`
(`src/mousedroid/security/injection_filter.py`) **through the gateway seam**;
the filter's own unit coverage lives in
`tests/unit/security/test_injection_filter.py` (11 tests) and already ran in the
coverage-gated `test` job. `docs/CHARTER.md` §3 names that filter as the control
making the cloud-LLM egress carve-out acceptable — "the only place rover NL
leaves the device" — so leaving the composite-path tests unenforced meant the
governance evidence for a ratified carve-out ran nowhere.

## What Changes

- `scripts/ci.sh` gains a stage running all three tiers, placed **outside** the
  `MOUSEDROID_CI_SLIM` skip (the whole set runs in ~2.5 s).
- `.github/workflows/ci.yml` gains a step in the **blocking** `test` job.
- `tests/regression/test_ci_gate_wiring_aqa.py` gains `TestOrphanTierWiring`,
  pinning both wirings and pinning that `tests/security` never lands in the
  advisory `security` job.
- `scripts/validations/F-028.sh` becomes the feature's validation command.

## Impact

No production code changes. No config fields. No new dependencies. The three
tiers already pass (17 tests, ~2.5 s), so the change is wiring plus its pin.

## Spec Deltas

`openspec/changes/mouse-droid-ci-tier-completeness/specs/ci-quality-gates/spec.md` — one ADDED requirement with three scenarios.

## Tasks

See `tasks.md`.

## Validation

`bash scripts/validations/F-028.sh` — runs the three tiers, then runs the
`TestOrphanTierWiring` pin.
