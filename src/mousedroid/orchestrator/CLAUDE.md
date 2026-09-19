# Orchestrator Subsystem — Surface Contract

> Sense-plan-act loop at 30 Hz (33.3 ms per tick). Coordinates sensor ingestion,
> latent world model planning, and actuator command generation.

## Invariants & Timing Budgets

1. **Strict 33.3 ms Cadence**: The main loop ticks at 30 Hz. I/O-bound tasks run concurrently
   via `asyncio.gather` / `asyncio.create_task`. Blocking calls go through `asyncio.to_thread`.
2. **Emergency Stop (E-Stop)**: `self._esp32.emergency_stop()` writes a zero-velocity stop
   frame to the ESP32 and returns once those bytes are handed to the serial port. It does
   **not** wait for, nor observe, motion ceasing — stock `General_Driver` firmware sends no
   per-command ACK, so nothing downstream confirms the wheels stopped. Corrected 2026-09-19
   (peer review D-12); this line previously claimed it "halts motor execution immediately".
   `ESP32Config.emergency_stop_budget_ms` bounds that write, not a stopping time or distance.
   Cooperative task cancellation (`asyncio.CancelledError`, a `BaseException` subclass — never
   caught by a bare `except Exception`) must always propagate.
3. **Telemetry Server & Shared Registry**: Handed a single `MetricsRegistry` from `factory/`
   via keyword-only argument `metrics: MetricsRegistry | None = None`.
4. **Safety Filter Projection**: Every action passes through `self._safety_monitor.evaluate(...)`
   (`SafetyMonitorProtocol`) before `_maybe_project_action` and serial dispatch to the ESP32
   driver.
5. **Graceful Shutdown (S-1)**: Entry points call `serve()`, which owns the whole lifecycle —
   signal handlers, `start()`, the loop, and `stop()` in a `finally`. Never drive
   `start()`/`run()`/`stop()` directly from an entry point: handlers must be installed *before*
   `start()`, because `start()` connects the ESP32 and brings up the sensors while the firmware
   may still hold a velocity latched from a previous unclean stop — the rover can be moving
   throughout bring-up. Without this, SIGTERM (`docker stop`, `systemctl stop`) terminates the
   process before any `finally` unwinds and `_halt_actuators`' `emergency_stop()` never fires.
   A signal calls `request_shutdown(reason)` (sync, idempotent, non-blocking — it is a signal
   handler), which latches `_shutdown_requested` and clears `_running` so the in-flight tick
   finishes. `start()` ends with `self._running = not self._shutdown_requested`, never an
   unconditional `True`, so a stop that lands mid-bring-up is honoured rather than overwritten.
   If the loop has not exited within `cfg.loop.shutdown_grace_s` the run task is cancelled
   outright. Signal registration is POSIX-only and degrades to a logged warning elsewhere;
   `run()` remains available for embedders driving their own lifecycle. Note the firmware-side chassis
   heartbeat (`comms/command_set.py`, `ESP32Config.heartbeat_enabled`) is the *complementary*
   failsafe covering what this cannot: a wedged Jetson or dropped USB link delivers no signal
   at all. It only arms under `command_set: waveshare_stock`; under the `legacy` default the
   driver now logs `esp32_heartbeat_unavailable` rather than staying silent about its absence.

## Key Files & Entry Points

- `orchestrator.py::MouseDroidOrchestrator` — main sense-plan-act execution loop (the
  production entrypoint; `factory/orchestrator.py::build_orchestrator` wires it). Contains
  `__init__` and `tick()` only; implementation is composition via seven mixins (see below).
- `_state.py::_OrchestratorState` — bare type-only attribute/method declarations every
  mixin inherits from (in addition to `object`), so mypy --strict can resolve cross-mixin
  `self._foo` access without per-call `# type: ignore[attr-defined]`. Never instantiated;
  never given a production implementation — its method stubs `raise NotImplementedError`
  as a fail-loud backstop, never a silent no-op. Keep it in sync with `__init__` in the
  same PR that changes either.
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
- `mission_lifecycle.py::MissionLifecycle` — the mission state machine. `_require_mission` is
  the single narrowing guard behind its four lifecycle methods (`_transition`,
  `_handle_stall`, `_transition_to_failed`, `_record_terminal_duration`); it raises
  `MissionLifecycleStateError` (a `RuntimeError` subclass, so existing handlers still catch)
  and is deliberately **not** an `assert` — `PYTHONOPTIMIZE=1` strips asserts on the rover,
  so an assert guard on the mission path does not exist in the shipped image.
- `mission_dispatcher.py`, `llm_replanner.py`, `face_controller.py` — supporting
  collaborators, not the loop itself.
- `factory/orchestrator.py::build_orchestrator` — factory builder wiring `MouseDroidOrchestrator`.
- `../safety/monitor.py::MouseDroidSafetyMonitor` — the concrete safety monitor
  (`SafetyMonitorProtocol` is the interface application code is typed against).
- `tests/unit/orchestrator/` — subsystem unit tests.

There is no `state.py` in this directory and no `_build_orchestrator` (private) helper — the
public `build_orchestrator` above is the sole factory entry point, per invariant 1
(factory-first DI).
