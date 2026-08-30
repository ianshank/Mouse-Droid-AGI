---
name: autonomous-mission-probe
description: Probe and validate the AutonomousOrchestrator 30 Hz mission control loop, safety interlocks, and Prometheus telemetry.
---

# Autonomous Mission Probe Skill

Workflow for probing the autonomous mission orchestration loop on simulated or edge hardware.

## Target Paths

- Orchestrator: `src/mousedroid/orchestrator/autonomous.py`
- Factory Builder: `src/mousedroid/factory/autonomous.py`
- Interface Contracts: `src/mousedroid/interfaces/protocols.py`

## Execution Steps

1. Build orchestrator using `mousedroid.factory.build_autonomous_orchestrator(cfg)`.
2. Run pre-flight sensor checks with `orch.validate_sensors()`.
3. Execute sample mission steps ("drive forward", "turn left").
4. Validate that proximity violations trigger instant e-stop.
5. Scrape Prometheus telemetry metrics from `metrics.render_prometheus()`.
