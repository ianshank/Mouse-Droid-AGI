# ADR-011 — Mission Closed-Loop + Safety Projection (Tier C2)

* **Status:** Accepted
* **Date:** 2026-05-16
* **Owners:** Ian Cruickshank
* **Scope:** `src/mousedroid/safety/projector.py`,
  `src/mousedroid/orchestrator/mission_lifecycle.py`,
  `src/mousedroid/orchestrator/orchestrator.py` (tick seam),
  `src/mousedroid/telemetry/metrics.py` (Tier C2 families).

## Context

Pre-C2 the orchestrator executed one-shot commands: a natural-language
mission was parsed into a `GoalVector`, the policy ran open-loop, and
the mission ended when the operator changed it or the tick budget
expired. Two structural gaps remained:

1. **No closed-loop progress feedback.** Nothing measured whether the
   rover was actually advancing toward the goal — a policy could spin
   in place for minutes and the system would not know.
2. **No soft-constraint enforcement on the action stream.** The hard
   `emergency_stop` short-circuit at the top of `tick()` zeroes the
   motors and aborts the tick, but between full-stop emergencies the
   policy was free to ignore soft constraints (forward clearance, human
   keep-out, tight-quarters rotation).

The Track C2 plan §"Peer-Review Findings" frames both as missing
*architectural seams*, not missing features — every dependency
(`SafetyContext`, `VLMProgressHead`, `LLMGatewayProtocol`,
`InMemoryTaskTracker`) already existed.

## Decision

Add two stateless, default-disabled seams:

* **`SafetyActionProjector`** — a pure function of `SafetyContext` + the
  policy's proposed action. Applied as a soft constraint AFTER
  `_select_action` returns and BEFORE `_execute_action` runs.
* **`MissionLifecycle`** — a small state machine
  (PENDING → RUNNING → SUCCEEDED | FAILED | REPLANNING → RUNNING) wrapping
  the existing `TaskTrackerProtocol`. Polls `VLMProgressHead` once per
  tick; transitions to REPLANNING on stall and asks the LLM gateway for
  a fresh `GoalVector`.

Both are off by default (`cfg.safety.projector.enabled = false` AND
`cfg.mission.replan_enabled = false`) so existing deployments are
byte-identical post-PR.

## Why geometric projection (not action-masking, not Lagrangian)

The plan agent considered three mechanisms for the soft-constraint seam:

| Mechanism | Verdict | Reason |
| --- | --- | --- |
| **Action masking** | Wrong fit | The rover action space is continuous (`[vx, vy, omega]`). Masking is a discrete-space technique — there is no obvious set of "valid actions" to enumerate. |
| **Lagrangian relaxation** | Wrong fit | Lagrangian methods carry a dual variable across ticks; the rest of the safety system (the frozen `SafetyContext` per-tick dataclass) is explicitly stateless. Introducing per-tick state for one component contradicts the design. |
| **Geometric constraint projection** | Chosen | A pure function of `SafetyContext` + the action. <1 ms on CPU. Deterministic. Three independent clamp rules (forward velocity / human proximity / tight quarters) compose cleanly. |

The CfC safety-trace consumer is a separate follow-on (deferred per
plan §"Out-of-Scope for C2"); geometry-only ships first.

## Soft vs. hard constraints

| Layer | Triggers | Effect | Recoverable |
| --- | --- | --- | --- |
| **Hard E-stop** (`orchestrator.py:399-418`) | `safety_ctx.is_emergency` | Motors → 0, voice event, face update, `POST_TICK` runs, `tick()` returns early. Policy never runs. | Yes (sensor recovery → re-evaluate). |
| **Soft projection** (`_maybe_project_action`) | Three independent clamp rules read from `SafetyContext`. | Action component-wise clamped; tick continues. | Always — the projector simply tightens limits. |

Inverting the order would be a regression: the policy must never run
during a hard emergency, and the projector must never run instead of
a hard emergency. The tick body enforces the order
(`emergency_stop_check → _update_world_model → _select_action →
safety_projection → execute → POST_TICK`).

## Mission lifecycle vs. `TaskTrackerProtocol`

`InMemoryTaskTracker` is the deterministic part of the agent harness —
acceptance predicates, timeouts, history bookkeeping. The
`MissionLifecycle` introduced here is a thin state machine that:

1. Polls a `VLMProgressHead` once per tick for progress feedback.
2. Triggers an async LLM replan when progress stalls.
3. Does NOT replace the tracker — it forwards terminal states to it
   (when a tracker is supplied) so operators keep their active-task list.

The lifecycle uses a `MissionLifecycleState` enum (an extension of
`TaskStatus` with `REPLANNING`) rather than extending `TaskStatus`
itself, because the tracker's existing terminal-state predicates would
not know what to do with a non-terminal `REPLANNING` value.

## Seam location — peer-review GAP 1

`Orchestrator._select_action` has FOUR return sites:

1. `cognitive` core action (`~676`)
2. VLA-policy action (`~686`)
3. VLA-strict-timeout safe stop (`~691`)
4. `nav_agent` fallback (`~693` / now `~701`)

Inserting the projector at one return site silently misses the others —
a human standing in front of the rover while the VLA policy is active
would see an unclamped action. The implementation wraps the call site
in `tick()` itself (around line 275) as:

```python
action = self._select_action(safety_ctx, observation, loop_time_ms)
action = self._maybe_project_action(action, safety_ctx)
```

so all four branches get clamped uniformly. The
`test_projector_applied_when_*` branch-coverage regression suite
parametrises this contract.

## Telemetry families

Four new Prometheus families ship in `MetricsRegistry`:

* `mousedroid_safety_action_clamps_total{reason}` —
  `forward_velocity` / `human_proximity` / `tight_quarters`
* `mousedroid_mission_state_transitions_total{from_state,to_state}`
* `mousedroid_mission_replans_total{outcome}` — `succeeded` / `failed`
* `mousedroid_mission_active_duration_seconds` — histogram, buckets
  from `MetricsConfig.mission_duration_seconds_buckets`

All four follow the PR-A2 pure-add pattern: rendered only after a
writer first touches them, so default-disabled deployments produce
byte-identical `/metrics` output.

## Consequences

**Positive**

* Closed-loop missions become possible with one config flip
  (`mission.replan_enabled = true`).
* Soft-constraint enforcement is unbypassable by future
  `_select_action` refactors (the seam is in `tick()`, not in the
  policy router).
* All Tier C2 metrics surface in Grafana / Loki without further code
  changes — the registry plumbing handles label cardinality.

**Negative**

* Two new feature flags (`safety.projector.enabled`,
  `mission.replan_enabled`) widen the config surface. Defaults
  preserve pre-PR behaviour so the production overlay decides when to
  flip them.
* `MissionLifecycle` introduces a second source of truth for mission
  state alongside `TaskTrackerProtocol`. The state machine documents
  the relationship; operators wiring both must keep them consistent.

## Alternatives considered

* **CfC safety-trace projection** — deferred to a C2.1 follow-on per
  plan §"Out-of-Scope". Geometry-only ships first.
* **Extending `TaskStatus` with `REPLANNING`** — rejected because the
  tracker's terminal-state predicates would need updating across the
  codebase. The lifecycle uses its own enum and is the only consumer
  of `REPLANNING`.

## Migration / rollback

* Rollback: set both feature flags to `false` (the default).
  `build_safety_projector` and `build_mission_lifecycle` both return
  `None`; the orchestrator behaves byte-identically to pre-PR.
* Migration: flip `safety.projector.enabled = true` first (low-risk;
  only clamps; no replans). Watch the
  `mousedroid_safety_action_clamps_total` counter. Then flip
  `mission.replan_enabled = true` with `max_replans_per_mission`
  tuned to the deployment's LLM budget.
