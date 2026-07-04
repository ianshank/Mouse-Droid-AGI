# ADR-013 — F-Number Namespaces + Findings-Only Audit Posture

* **Status:** Accepted
* **Date:** 2026-07-03
* **Owners:** Ian Cruickshank
* **Scope:** `features.yaml` (catalog IDs F-015+), `HARNESS_SPEC.md`
  ("F-number namespaces" section), `SMOKE_REPORT.md` (findings namespace),
  `scripts/dead_code_audit.py`, `scripts/check_advisory_promotions.py`,
  `.github/advisory_stages.yaml`, `scripts/vulture_allowlist.py`,
  `tests/regression/test_dead_code_audit.py`,
  `tests/unit/scripts/test_check_advisory_promotions.py`.

## Context

Two permanent decisions landed with the rev. B validation-first work streams
(PR #151) that outlive any single PR and therefore need durable rationale
(HARNESS_SPEC §5: structural changes → `progress.md`; *permanent* decisions →
ADR).

**1. Two F-number sequences already coexisted.** The spec-harness catalog
(`features.yaml`, ADR-012) counts F-001–F-008, while the live-Jetson smoke
campaigns independently minted *operational finding* IDs in `SMOKE_REPORT.md`
that reached F-014 (e.g. finding F-013 "stale prod-YAML deploy drift", closed
by commit `3015283` — "F-013/F-014 closeout"). The rev. A implementation plan
walked straight into this trap by proposing catalog features F-009…F-014;
peer review caught the collision before anything landed. Reusing those IDs
would have poisoned `implemented_in` provenance and made every historical
cross-reference ambiguous.

**2. Automated dead-code detection on a Protocol/DI codebase over-reports by
construction.** The first vulture sweep of `src/mousedroid` produced 437
findings at 60% confidence — dominated by `@runtime_checkable` Protocol
members (called through the protocol, never by name), pydantic
`@field_validator`/`@model_validator` methods (framework-invoked), and
factory `build_*` hooks reached only from tests/CLI entry points. A blocking
gate on that signal would either be red forever or force a giant
threshold/allowlist that hides real rot.

## Decision

1. **The two F-number namespaces are permanently independent.** The catalog
   (`features.yaml`) owns the dependency DAG, schema validation, and
   `validate.py` execution; smoke findings are report-local triage IDs. New
   catalog entries continue from **F-015**, deliberately skipping 9–14
   rather than shadowing the burned finding IDs. Cross-references must say
   "smoke finding F-0xx" vs "feature F-0xx".
2. **The redundancy/gap audit is findings-only (never blocking).**
   `scripts/dead_code_audit.py` writes dated JSON reports and exits 0
   (unless `--strict`); deletion decisions stay with a human, recorded by
   adding verified-alive symbols to `scripts/vulture_allowlist.py` (each
   entry carries a one-line WHY). The CI job runs `continue-on-error: true`.
3. **Advisory stages carry a tracked promotion window.** Every
   `continue-on-error` CI job must have an entry in
   `.github/advisory_stages.yaml` (`since` + `promote_after_days` + reason);
   `scripts/check_advisory_promotions.py` WARNs on untracked stages and
   overdue promotions, so "advisory forever by accident" is impossible —
   staying advisory past the window requires a recorded reason (or, for the
   dead-code audit, the decision at window end is explicitly
   "advisory-forever + this ADR" vs "strict mode on a curated allowlist").

## Consequences

* The catalog's ID sequence has a visible hole (F-009–F-014). This is
  intentional and documented in `features.yaml` (comment above F-015) and
  `HARNESS_SPEC.md`; the alternative (an `OPS-F-…` prefix scheme, permitted
  by the schema's `^([A-Z]+-)?F-[0-9]{3,}$` pattern) was rejected because
  the findings namespace predates it and renaming history is worse than a
  gap.
* Dead-code hygiene relies on humans reading the audit report (surfaced as a
  CI artifact + `reports/dead_code/` locally). The promotion-lag checker is
  the backstop that keeps the posture a *decision* rather than drift.
* Future agents adding catalog features MUST take the next free ID at F-015
  or above; `tests/regression/test_harness_spec_aqa.py` enforces uniqueness,
  and the HARNESS_SPEC section is the discoverable rule.

## References

* `HARNESS_SPEC.md` — "F-number namespaces" section (normative rule).
* ADR-012 — the spec-driven harness this extends.
* `docs/superpowers/plans/2026-07-03-validation-first-rev-b.md` — the
  peer-reviewed plan whose review surfaced the collision.
* `progress.md` Session 003 — the landing log.
