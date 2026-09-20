# Safety Subsystem — Surface Contract

> Per-tick hazard evaluation producing a frozen ``SafetyContext``
> (``SafetyMonitorProtocol``; ``MouseDroidSafetyMonitor``, ``RoboticsLawChecker``).

## Invariants & Safety Rules

1. **Protocol Surface**: Monitors implement ``SafetyMonitorProtocol.evaluate(observation,
   loop_time_ms, *, tick_index=...)`` and return a ``SafetyContext``.
2. **Fail-Closed Clearance**: When LiDAR is unavailable the monitor reports
   ``LIDAR_UNAVAILABLE_DIST_M = 0.0`` — not a tunable — so clearance checks fail closed.
3. **Projection & Latch**: Action projection (``SafetyActionProjectorProtocol``) and
   emergency latch (``EmergencyLatchProtocol``) are separate seams injected beside the
   monitor; Three Laws checks live in ``RoboticsLawChecker``.

## Key Files

- `protocol.py::SafetyMonitorProtocol` — monitor interface.
- `monitor.py::MouseDroidSafetyMonitor` — concrete per-tick evaluator.
- `context.py::SafetyContext` — frozen safety state agents consume.
- `three_laws.py::RoboticsLawChecker` — Three Laws checks.
- `projector.py` / `projector_protocol.py` — action projection seam.
- `latch.py` — emergency latch protocol and implementation.
- `tests/unit/safety/` — subsystem unit tests.
