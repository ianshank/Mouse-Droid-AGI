# Proposal — AutonomousOrchestrator disposition

- change_id: mouse-droid-autonomous-orchestrator-disposition
- project: mouse-droid
- status: in_progress
- feature_id: F-031
- epic: Quality Gates
- owner: ianshank
- created: 2026-08-23
- basis_commit: 15905a5
- rev: A

## Why

`src/mousedroid/orchestrator/autonomous.py::AutonomousOrchestrator` is a second
sense-plan-act implementation with zero production callers — nothing in
`main.py` routes to it — but nothing in the tree previously recorded *why*.
`orchestrator/CLAUDE.md` even carried a dangling forward reference: "see
`docs/architecture/adr/` for the disposition ADR once F-031 lands."

Two properties put it in direct conflict with `docs/CHARTER.md` as implemented
today: `execute_mission_step` calls `self._llm.translate_mission(...)`
unconditionally inside `run_loop`, which defaults to 30 Hz
(`DEFAULT_CONTROL_LOOP_INTERVAL_S = 0.033`) — CHARTER §3 puts any LLM work
inside the 30 Hz hot loop explicitly out of scope, and invariant 10 requires
the loop stay "deterministic, LLM-free, and training-free." Second, it has
zero safety-monitor machinery — a repo-wide grep for
`SafetyMonitor|ThreeLaws|safety_projector|sensor_stale|max_loop_time` returns
0 matches in `autonomous.py` against 8 in the production `orchestrator.py`.

So the absence of a production wiring path is the *correct* posture. An
undocumented disposition is a hazard, not a neutral state — the next engineer
reading `factory.py::build_autonomous_orchestrator` has no signal not to wire
it up.

## What Changes

`docs/architecture/ADR-016-autonomous-orchestrator-disposition.md` records the
decision. `tests/regression/test_import_graph_freeze.py` gains a third,
file-level (not directory-level) parked-subsystem case pinning that no
production module grows a module-scope import of
`mousedroid.orchestrator.autonomous` — `autonomous.py` shares a directory with
the production `orchestrator.py`, so a directory-level exemption (the pattern
the two existing cases use) would also license `orchestrator.py` to import it.
`orchestrator/CLAUDE.md`'s forward reference is updated to point at the real
ADR filename.

## Impact

No runtime behavior change. `factory.py::build_autonomous_orchestrator`
already only imports `AutonomousOrchestrator` function-scoped (lazy), so the
new pin is expected to pass immediately on the current tree — its value is in
freezing that state against future drift, not in fixing a present violation.

## Spec Deltas

`openspec/changes/mouse-droid-autonomous-orchestrator-disposition/specs/orchestrator-disposition/spec.md`

## Tasks

See `openspec/changes/mouse-droid-autonomous-orchestrator-disposition/tasks.md`.

## Validation

`bash scripts/validations/F-031.sh`
