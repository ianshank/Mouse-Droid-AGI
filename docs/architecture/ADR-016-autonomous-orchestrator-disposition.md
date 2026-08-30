# ADR-016 — AutonomousOrchestrator Disposition

* **Status:** Accepted
* **Date:** 2026-08-23
* **Deciders:** @ianshank
* **Area:** Orchestrator / architecture

## Context

`src/mousedroid/orchestrator/autonomous.py::AutonomousOrchestrator` is a second,
independent sense-plan-act implementation alongside the production
`orchestrator.py::MouseDroidOrchestrator`. It has zero production call sites —
`factory/autonomous.py::build_autonomous_orchestrator` is never invoked from `main.py`'s `_run`
or `_health_check`, only from test code and a documentation skill — but nothing in
the tree previously recorded *why* it is parked, which is itself a hazard: the next
engineer reading `factory/autonomous.py` and finding a fully-formed, apparently-usable builder
has no signal not to wire it up.

Two properties of `AutonomousOrchestrator`, as implemented today, put it in direct
conflict with this project's own constitution:

1. **It calls the LLM gateway inside its control loop.**
   `execute_mission_step` does `goal: GoalVector = await
   self._llm.translate_mission(mission_command)` unconditionally on every step, and
   `run_loop` defaults `interval_s=DEFAULT_CONTROL_LOOP_INTERVAL_S` —
   `constants.py:281`'s `0.033` seconds, i.e. 30 Hz. `docs/CHARTER.md` §3 lists, under
   what is explicitly out of scope: *"Any LLM or training work placed inside the 30 Hz
   hot loop. Deliberation and learning are off-loop by construction (§4, invariant
   10)."* Invariant 10 itself: *"Hot-loop purity. The 30 Hz reactive loop (RSSM → MCTS
   → ESP32) stays deterministic, LLM-free, and training-free. All deliberation (NL
   translation, operator Q&A) ... run at slow-cadence seams outside the hot loop."*
   `AutonomousOrchestrator`'s hot loop is neither.
2. **It has no safety-monitor machinery.** A repo-wide search for
   `SafetyMonitor|ThreeLaws|safety_projector|sensor_stale|max_loop_time` returns zero
   matches in `autonomous.py` and eight across the production
   `MouseDroidOrchestrator` class (`orchestrator.py` plus its `_*_mixin.py`
   files post-ADR-017 decomposition; `SafetyMonitorProtocol` at construction,
   `safety_projector` gating actuation in `_action_mixin.py`).
   `AutonomousOrchestrator`'s only interlock is a bare proximity check
   (`min(scan) < min_range_m`) plus a `self._safety_latched` bool — no monitor
   evaluates every action before dispatch the way invariant 4 of
   `orchestrator/CLAUDE.md` requires of the production loop.

So the absence of a production wiring path is the *correct* posture, not a gap to
close. This ADR exists to record that decision explicitly, and a companion regression
test (`tests/regression/test_import_graph_freeze.py`) pins that no production module
grows a module-scope dependency on it going forward.

## Decision

We will keep `AutonomousOrchestrator` off the production path. `main.py`'s `_run` and
`_health_check` continue to route exclusively through
`factory/orchestrator.py::build_orchestrator` → `MouseDroidOrchestrator`.
`factory/autonomous.py::build_autonomous_orchestrator` remains an explicit, opt-in builder whose
only reference to `mousedroid.orchestrator.autonomous` is a function-scoped (lazy)
import — never a module-top-level one. A regression test enforces this: no file other
than `autonomous.py` itself may carry a module-scope import of
`mousedroid.orchestrator.autonomous`.

## Consequences

**Positive.** A charter-violating component is decisively and legibly kept off the
production path, with the reasoning recorded once rather than re-derived by every
future reader of `factory/`. The disposition is now enforced by a regression pin,
not resting on convention alone — a future refactor that accidentally promotes it to a
module-scope import fails CI immediately. `orchestrator/CLAUDE.md`'s forward reference
("see `docs/architecture/adr/` for the disposition ADR once F-031 lands") now resolves
to a real document.

**Negative / neutral.** The class remains in the tree and continues to need
maintenance for its own test surface — it is exercised (directly or by name) across
`tests/unit/`, `tests/property/`, `tests/integration/`, `tests/functional/`,
`tests/user_journey/`, `tests/e2e/`, and `tests/smoke/`, plus the
`autonomous-mission-probe` skill's documented workflows. None of that is dead weight
this ADR removes; it stays as-is. If `AutonomousOrchestrator` is ever made
production-eligible (LLM call moved off-loop, a real safety monitor wired in), that
requires a new ADR superseding this one — this decision does not expire on its own.

## Alternatives considered

1. **Delete the class.** Rejected — it is exercised by test files across nearly every
   tier and by a shipped skill's documented workflows; deleting it destroys that
   coverage and tooling for no operational benefit, since a parked, zero-production-caller
   class already costs nothing at runtime.
2. **Retrofit it to be production-eligible** — move the LLM call off-loop and wire in
   `SafetyMonitorProtocol` so it could legitimately replace or supplement the
   production orchestrator. Rejected as out of scope here: CHARTER §3 draws this
   boundary deliberately, and making the retrofit would be a substantial, separate
   architectural change needing its own design and review, not a documentation bundle.
3. **Leave the disposition undocumented (status quo).** Rejected — this is the exact
   shape of drift F-030 found elsewhere in this same file (`orchestrator/CLAUDE.md`
   previously named phantom symbols entirely) and, after that fix, left a dangling
   forward reference to an ADR that didn't exist yet. An undocumented disposition is a
   hazard, not a neutral state: it invites the next engineer to wire the component up
   without knowing why it was left alone.
