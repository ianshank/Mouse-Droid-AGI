# MouseDroid — Claude Code Project Instructions

> Autonomous navigation system for a Star Wars MSE-6 droid running on NVIDIA Jetson Orin Nano.
> `docs/CHARTER.md` is the project constitution (vision, scope, invariants, roadmap) and sits
> above this document. When a change touches scope or an invariant, defer to the charter.

## Architectural Invariants (Never Weaken These)

1. **Factory-First DI**: Concrete types are imported *inside* `src/mousedroid/factory.py` only.
   Application code is typed against `@runtime_checkable Protocol` interfaces.
2. **Schema-Driven Configuration**: Every threshold, dimension, pin, path, and tunable parameter
   comes from Pydantic schemas in `src/mousedroid/config/schema/` loaded from YAML in `config/`.
3. **Structured Logging**: Use `from mousedroid.logging.setup import get_logger`, then
   `_log.info("event_name", key=value)`. No `print()`. No f-string log messages.
4. **Asyncio Everywhere**: All I/O-bound operations are `async`. Blocking syscalls use `asyncio.to_thread`.
5. **Strict Typing**: `mypy --strict` must pass on every PR. All public APIs carry type annotations.
6. **Backwards Compatibility**: New config fields MUST carry `Field(default=..., description=...)`.
   Existing YAML must load unchanged after a `git pull`.
7. **`torch.no_grad()`**: Required on every inference path.
8. **`deque(maxlen=N)`**: Required for every sensor ring buffer (`N` from schema config).
9. **Test-Pyramid Discipline**: Every change lands across matching test tiers (Unit, Property,
   Integration, E2E, Regression, Smoke). Bounded complexity: `ruff C901` (`max-complexity = 15`).

## Collaboration Directive (I-6)

- **Autonomous Multi-Agent Orchestration**: Act proactively as lead developer, architect, and SQE.
- **Subagent Delegation**: Delegate deep tasks (security scans, code quality, research) to specialized
  subagents. Subagent briefs must be self-contained and concise.
- **Minimal Tool Latency**: Bundle parallel checks and run background tasks efficiently.

## Capability Freeze Rule (F-024 Governance)

- **F-008 Hardware Priority**: Modifications to `src/mousedroid/arm/**` are blocked by
  PreToolUse hook `freeze_gate.py` while F-008 is not `done`. Hardware readiness preempts
  in-flight software streams.
- **Override**: Set `MOUSEDROID_WORKFORCE_ALLOW_FROZEN=1` for reviewed exceptional edits.

## Hardware-Mode Discipline

- **Global Mock Toggle (`Settings.mock_hardware`)**: Set via `MOUSEDROID_MOCK_HARDWARE`.
  Root `tests/conftest.py` forces mock true; `tests/hardware/conftest.py` reverses it for live tests.
- **Subsystem Toggles (`enabled: bool`)**: Fine-grained dev escape hatches (e.g. `esp32.enabled`,
  `lidar.enabled`). If code needs to behave differently when hardware is missing, use a schema toggle.

## CI/CD Pipeline (12 Jobs)

Authoritative pipeline in `.github/workflows/ci.yml`:

- **Stage 0 (Fast Fail)**: `actionlint`, `lint` (ruff), `secret-scan` (gitleaks), `skills`.
- **Stage 1 (Strict Types & Fast Tests)**: `typecheck` (mypy --strict), `test-fast`, `validate`.
- **Stage 2 (Coverage & Matrix)**: `test` (Python 3.10/3.11/3.12, cov >= 90%), `test-windows` (advisory), `local-gates` (tools cov).
- **Stage 3 (Regression & AQA)**: `regression` (`tests/regression/`).
- **Stage 4 (Release Packaging)**: `package`.

Run locally: `make gates` (fast lint/typecheck/validate), `make test` (coverage), or `bash scripts/ci.sh` (superset).

## Surface Map

Detailed operational guidelines and subsystem contracts live in partitioned surface docs:

### Subsystem Contracts

- [Orchestrator](file:///src/mousedroid/orchestrator/CLAUDE.md) — 30 Hz mission loop, timing budgets, e-stop.
- [LLM Gateway](file:///src/mousedroid/llm_gateway/CLAUDE.md) — Composite dispatch, prompt injection sanitization, failover.
- [Hardware Drivers](file:///src/mousedroid/hardware/CLAUDE.md) — Sensor drivers, sysfs encoding, ring buffers, mock discipline.
- [Telemetry Server](file:///src/mousedroid/telemetry/CLAUDE.md) — Prometheus metrics, label validation, REST API.
- [Continual Learning](file:///src/mousedroid/learning/CLAUDE.md) — EWC, progressive nets, memory bounds.
- [Growth & Distillation](file:///src/mousedroid/growth/CLAUDE.md) — Off-loop VLA teacher-student distillation.
- [World Model](file:///src/mousedroid/world_model/CLAUDE.md) — RSSM latent dynamics, MCTS planner, ONNX engine.
- [Robot Arm](file:///src/mousedroid/arm/CLAUDE.md) — Parked platform, F-008 freeze notice, MuJoCo envs.

### Cross-Cutting Operational Surfaces

- [CI Gates & Quality Ladders](file:///docs/claude/surfaces/ci-gates.md) — 12-job CI matrix, advisory promotion ladder.
- [On-Device Full Validation](file:///docs/claude/surfaces/full-validation.md) — Cold-then-warm validation methodology and commands.
- [USB-C Discovery Protocol](file:///docs/claude/surfaces/usbc-smoke.md) — Dynamic endpoint resolution and hardware enumeration.
- [MCP Evaluation Notes](file:///docs/claude/surfaces/mcp-evaluation.md) — Model Context Protocol evaluate-first decisions.
- [Parallel Worktrees](file:///docs/runbooks/worktrees.md) — Multi-agent Git worktree isolation guide.

## Key Developer Commands

```bash
make help               # List all developer targets
make gates              # Fast lint, format, typecheck, skills, and validation gates
make test               # Full unit + property + integration test suite with coverage gate
make hooks              # Dedicated workforce tooling coverage test
bash scripts/ci.sh      # Authoritative local CI superset
```

## Red Flags (Pause and Check)

- **Hardcoded Values**: Never hardcode ports, pins, paths, or thresholds — use Pydantic schema config.
- **Leaked Secrets**: Never put credentials in code or YAML defaults; use `SecretStr` and environment vars.
- **Blocking Syscalls**: Never call blocking I/O in async routines — use `asyncio.to_thread`.
- **`assert` under Optimization**: Never use `assert` in runtime code paths running under `PYTHONOPTIMIZE=1`.
- **Untracked `.claude/` Assets**: New shared workforce files must have `.gitignore` negation (`!.claude/<path>`).
- **Sysfs File Encoding**: Always open sysfs files with `encoding="utf-8", errors="replace"`.
