# Orchestrator Subsystem — Surface Contract

> Sense-plan-act loop at 30 Hz (33.3 ms per tick). Coordinates sensor ingestion,
> latent world model planning, and actuator command generation.

## Invariants & Timing Budgets

1. **Strict 33.3 ms Cadence**: The main loop ticks at 30 Hz. I/O-bound tasks run concurrently
   via `asyncio.gather` / `asyncio.create_task`. Blocking calls go through `asyncio.to_thread`.
2. **Emergency Stop (E-Stop)**: An e-stop signal halts motor execution immediately and tears down
   the active mission task. Cooperative task cancellation (`asyncio.CancelledError`) must always
   propagate.
3. **Telemetry Server & Shared Registry**: Handed a single `MetricsRegistry` from `factory.py`
   via keyword-only argument `metrics: MetricsRegistry | None = None`.
4. **Safety Filter Projection**: Motor commands pass through the runtime safety filter
   (`ConstitutionalSafetyMonitor`) before serial dispatch to the ESP32 driver.

## Key Files & Entry Points

- `orchestrator.py` — Main `RobotOrchestrator` execution loop.
- `state.py` — Sense-plan-act state container.
- `factory.py:_build_orchestrator` — Factory builder wiring all subsystems.
- `tests/unit/orchestrator/` — Subsystem unit tests.
