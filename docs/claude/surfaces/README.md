# Claude Workforce Operational Surfaces

> Cross-cutting technical and governance surfaces partitioned from root instructions.

## Operational Surfaces

- [CI Gates & Quality Ladders](file:///docs/claude/surfaces/ci-gates.md) — 17-job CI pipeline, advisory stages, promotion ladder.
- [On-Device Full Validation](file:///docs/claude/surfaces/full-validation.md) — Cold-then-warm validation methodology and commands.
- [USB-C Discovery Protocol](file:///docs/claude/surfaces/usbc-smoke.md) — Dynamic endpoint resolution and hardware enumeration.
- [MCP Evaluation Notes](file:///docs/claude/surfaces/mcp-evaluation.md) — Model Context Protocol evaluate-first decisions.
- [Package import map](file:///docs/architecture/package-map.md) — Generated import graph (not dataflow); epic splits under `docs/architecture/package-map/`.

## Subsystem Contracts

- [Orchestrator](file:///src/mousedroid/orchestrator/CLAUDE.md) — 30 Hz mission loop, timing budgets, e-stop.
- [LLM Gateway](file:///src/mousedroid/llm_gateway/CLAUDE.md) — Composite dispatch, prompt injection sanitization, failover.
- [Hardware Drivers](file:///src/mousedroid/hardware/CLAUDE.md) — Sensor drivers, sysfs encoding, ring buffers, mock discipline.
- [Telemetry Server](file:///src/mousedroid/telemetry/CLAUDE.md) — Prometheus metrics, label validation, REST API.
- [Continual Learning](file:///src/mousedroid/learning/CLAUDE.md) — EWC, progressive nets, memory bounds.
- [Growth & Distillation](file:///src/mousedroid/growth/CLAUDE.md) — Off-loop VLA teacher-student distillation.
- [World Model](file:///src/mousedroid/world_model/CLAUDE.md) — RSSM latent dynamics, MCTS planner, ONNX engine.
- [Robot Arm](file:///src/mousedroid/arm/CLAUDE.md) — Parked platform, F-008 freeze notice, MuJoCo envs.
- [Agents](file:///src/mousedroid/agents/CLAUDE.md) — MCTS navigation agents, AgentProtocol, safety override.
- [Cognitive](file:///src/mousedroid/cognitive/CLAUDE.md) — Dual-cadence BDI / metacognition / constitutional RL.
- [Comms](file:///src/mousedroid/comms/CLAUDE.md) — ESP32 serial/WiFi drivers (ESP32CommProtocol).
- [Config](file:///src/mousedroid/config/CLAUDE.md) — Settings schema and YAML loader.
- [Experience](file:///src/mousedroid/experience/CLAUDE.md) — LMDB experience logging and versioned records.
- [Logging](file:///src/mousedroid/logging/CLAUDE.md) — structlog setup, safe extras, URI redaction.
- [Memory](file:///src/mousedroid/memory/CLAUDE.md) — Working / episodic / semantic tiers and consolidation.
- [Safety](file:///src/mousedroid/safety/CLAUDE.md) — SafetyMonitorProtocol, SafetyContext, Three Laws.
- [Sensing](file:///src/mousedroid/sensing/CLAUDE.md) — SensorManager fusion and ObservationProtocol bundles.
