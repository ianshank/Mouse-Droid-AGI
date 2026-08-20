# Claude Workforce Operational Surfaces

> Cross-cutting technical and governance surfaces partitioned from root instructions.

## Operational Surfaces

- [CI Gates & Quality Ladders](file:///docs/claude/surfaces/ci-gates.md) — 12-job CI pipeline, advisory stages, promotion ladder.
- [On-Device Full Validation](file:///docs/claude/surfaces/full-validation.md) — Cold-then-warm validation methodology and commands.
- [USB-C Discovery Protocol](file:///docs/claude/surfaces/usbc-smoke.md) — Dynamic endpoint resolution and hardware enumeration.
- [MCP Evaluation Notes](file:///docs/claude/surfaces/mcp-evaluation.md) — Model Context Protocol evaluate-first decisions.

## Subsystem Contracts

- [Orchestrator](file:///src/mousedroid/orchestrator/CLAUDE.md) — 30 Hz mission loop, e-stop, telemetry registry.
- [LLM Gateway](file:///src/mousedroid/llm_gateway/CLAUDE.md) — Composite dispatch, prompt injection sanitization, failover.
- [Hardware Drivers](file:///src/mousedroid/hardware/CLAUDE.md) — Camera, LiDAR, ESP32, ring buffers, mock discipline.
- [Telemetry Server](file:///src/mousedroid/telemetry/CLAUDE.md) — Prometheus metrics, label validation, REST API.
- [Continual Learning](file:///src/mousedroid/learning/CLAUDE.md) — EWC, progressive nets, memory bounds.
- [Growth & Distillation](file:///src/mousedroid/growth/CLAUDE.md) — Off-loop VLA teacher-student distillation.
- [World Model](file:///src/mousedroid/world_model/CLAUDE.md) — RSSM latent dynamics, MCTS planner, ONNX engine.
- [Robot Arm](file:///src/mousedroid/arm/CLAUDE.md) — Parked platform, F-008 freeze notice, MuJoCo envs.
