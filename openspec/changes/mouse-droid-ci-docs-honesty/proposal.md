# Proposal — CI/docs honesty (parked journeys + Current Next Steps pin)

- change_id: mouse-droid-ci-docs-honesty
- project: mouse-droid
- status: active
- feature_id: F-038
- epic: Hygiene
- owner: ianshank
- created: 2026-09-07
- rev: A

## Why

The CI step "Run functional + user-journey + security tiers" reads as
operator-path coverage. Those two directories uniquely prove parked
`AutonomousOrchestrator` APIs (`execute_mission_step`, `MockLiDAR.set_scan`)
that production `MouseDroidOrchestrator` does not have. Twins under
`mock_hardware` are vacuous. The bug is the label.

`NEXT_STEPS.md` Current section accreted LANDED rows (F-016 already taught
this). `tools/doc_hygiene.py` exited 0 unless `--strict`, so stripping
prose without a pin would re-drift.

## What Changes

- Relabel CI / Makefile / skill / module docs as parked-autonomous.
- Pin Current Next Steps: no `LANDED`; done catalog ids only on leftover lines.
- Invoke `doc_hygiene.py --strict` in `ci.sh` and local-gates.
- Point CHARTER §5 at root `NEXT_STEPS.md`; disambiguate M6 vs LoRA Phase 6.
- Name catalog vs smoke-report F-008 once (ADR-013).

## Impact

No runtime behaviour change. Parked tests keep calling
`build_autonomous_orchestrator`.

## Charter

No CHARTER §3 carve-out: docs + CI labels only.

## Validation

`bash scripts/validations/F-038.sh`
