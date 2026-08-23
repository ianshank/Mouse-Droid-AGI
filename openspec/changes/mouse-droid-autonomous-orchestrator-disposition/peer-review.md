# Peer review — AutonomousOrchestrator disposition

This change was designed and independently reviewed as part of a broader
multi-agent adversarial review covering six deliverables (F-030 closeout,
F-031, F-032, F-034, F-035, F-036) before any implementation began. F-031
specifically was reviewed by four independent passes — architect/product,
SQE/test-pyramid, config/schema, and security — each verifying claims
directly against the tree rather than trusting the design draft.

## Verdict table

| Claim | Verdict |
|---|---|
| `AutonomousOrchestrator` has zero production call sites | **CONFIRMED** — `factory.py::build_autonomous_orchestrator` is the sole reference, function-scoped; no `main.py` call site |
| It calls the LLM gateway unconditionally inside a 30 Hz loop | **CONFIRMED** — `execute_mission_step` → `self._llm.translate_mission(...)`, `run_loop` defaults `interval_s=DEFAULT_CONTROL_LOOP_INTERVAL_S` = `0.033` |
| It has zero safety-monitor machinery | **CONFIRMED** — 0 matches for `SafetyMonitor\|ThreeLaws\|safety_projector\|sensor_stale\|max_loop_time` in `autonomous.py`, 8 in `orchestrator.py` |
| ADR-016 is the next free ADR number | **CONFIRMED** — log runs ADR-004 through ADR-015 plus one unnumbered `ADR-l4t-container` |
| A directory-level exemption would be wrong for this case | **CONFIRMED** — `autonomous.py` shares a directory with production `orchestrator.py`, unlike `arm/` or `hardware/sensors/` |
| The pin is provable | **CONFIRMED** — injected a throwaway module-scope import into `orchestrator.py`, confirmed red naming the offending file, restored, confirmed green and byte-identical |
| `AutonomousOrchestrator` is otherwise dead code | **REFUTED** — exercised across `tests/unit/`, `tests/property/`, `tests/integration/`, `tests/functional/`, `tests/user_journey/`, `tests/e2e/`, `tests/smoke/`, and the `autonomous-mission-probe` skill's documented workflows; a wider surface than the original design draft enumerated |

## What survives review unchanged

- Keeping the disposition as a documentation + enforcement change, not a
  redesign — CHARTER §3 already draws the scope line deliberately, and
  retrofitting the class for production eligibility is a substantial,
  separate architectural decision.
- Not deleting the class — its test and tooling surface is real and would be
  destroyed for no operational benefit, since a parked class with zero
  production callers already costs nothing at runtime.

## Load-bearing pins any implementation must satisfy

1. `docs/architecture/ADR-016-autonomous-orchestrator-disposition.md` exists
   and is `Accepted`.
2. `tests/regression/test_import_graph_freeze.py::test_no_active_module_imports_autonomous_orchestrator_at_module_scope`
   fails when any file outside `orchestrator/autonomous.py` itself carries a
   module-scope import of `mousedroid.orchestrator.autonomous`.
3. `orchestrator/CLAUDE.md`'s forward reference resolves to the real ADR
   filename, not a "once F-031 lands" placeholder.

## Appendix — related, not folded in

The review that produced this change's design also caught that F-030 itself
had shipped without its own catalog closeout (`features.yaml` still showed
`status: "in_progress"` after its PR had already merged). That gap was fixed
separately, first, as its own PR — not folded into F-031, since it was
unrelated bookkeeping debt on a different feature. This change's own
`implemented_in` pin follows the same convention going forward: `null` while
in flight, pinned to the real merge SHA in a small follow-up commit once the
PR is known.
