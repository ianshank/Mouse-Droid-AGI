# Orchestrator Subsystem — Surface Contract

> Sense-plan-act loop at 30 Hz (33.3 ms per tick). Coordinates sensor ingestion,
> latent world model planning, and actuator command generation.

## Invariants & Timing Budgets

1. **Strict 33.3 ms Cadence**: The main loop ticks at 30 Hz. I/O-bound tasks run concurrently
   via `asyncio.gather` / `asyncio.create_task`. Blocking calls go through `asyncio.to_thread`.
2. **Emergency Stop (E-Stop)**: `self._esp32.emergency_stop()` halts motor execution
   immediately. Cooperative task cancellation (`asyncio.CancelledError`, a `BaseException`
   subclass — never caught by a bare `except Exception`) must always propagate.
3. **Telemetry Server & Shared Registry**: Handed a single `MetricsRegistry` from `factory/`
   via keyword-only argument `metrics: MetricsRegistry | None = None`.
4. **Safety Filter Projection**: Every action passes through `self._safety_monitor.evaluate(...)`
   (`SafetyMonitorProtocol`) before `_maybe_project_action` and serial dispatch to the ESP32
   driver.

## Key Files & Entry Points

- `orchestrator.py::MouseDroidOrchestrator` — main sense-plan-act execution loop (the
  production entrypoint; `factory/orchestrator.py::build_orchestrator` wires it). Contains
  `__init__` and `tick()` only; implementation is composition via seven mixins (see below).
- `_lifecycle_mixin.py` — startup, shutdown, background task management, health checks.
- `_mission_mixin.py` — natural language mission acceptance and lifecycle coordination.
- `_world_model_state_mixin.py` — latent state validation, NaN recovery, OTA weight updates.
- `_action_mixin.py` — action selection, VLA/cognitive dispatch, safety projection.
- `_telemetry_experience_mixin.py` — frame publishing, experience logging, curiosity scoring.
- `_voice_face_mixin.py` — voice output and facial expression control.
- `_background_cadence_mixin.py` — sensor recovery, memory consolidation, on-device learning,
  growth distillation loops.
- `autonomous.py::AutonomousOrchestrator` — an alternate loop with **zero production
  callers**, deliberately parked off the production path per
  `docs/architecture/ADR-016-autonomous-orchestrator-disposition.md`; do not confuse it
  with the production path above.
- `mission_dispatcher.py`, `mission_lifecycle.py`, `llm_replanner.py`, `face_controller.py` —
  supporting collaborators, not the loop itself.
- `factory/orchestrator.py::build_orchestrator` — factory builder wiring `MouseDroidOrchestrator`.
- `../safety/monitor.py::MouseDroidSafetyMonitor` — the concrete safety monitor
  (`SafetyMonitorProtocol` is the interface application code is typed against).
- `tests/unit/orchestrator/` — subsystem unit tests.

There is no `state.py` in this directory and no `_build_orchestrator` (private) helper — the
public `build_orchestrator` above is the sole factory entry point, per invariant 1
(factory-first DI).
