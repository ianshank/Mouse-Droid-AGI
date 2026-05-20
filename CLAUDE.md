# MouseDroidAGI — Claude Code Project Instructions

## Overview

MouseDroidAGI is a Star Wars MSE-6 autonomous navigation system and hierarchical robot arm training platform running on NVIDIA Jetson Orin Nano. It implements the 10 Pillars of the Ideal Neural Network as a cohesive agentic system.

## Architecture Invariants

These invariants are **non-negotiable** across all modules:

1. **Protocol-based DI**: All interfaces are `@runtime_checkable Protocol`. Concrete types are only imported inside factory functions.
2. **Factory pattern**: `src/mousedroid/factory.py` is the single wiring point. All `build_*()` functions return protocol types.
3. **No hardcoded values**: Every threshold, dimension, pin number, path, and tunable parameter comes from Pydantic config (`src/mousedroid/config/schema.py`) loaded from YAML (`config/*.yaml`).
4. **Structured logging**: Use `structlog` everywhere via `from mousedroid.logging.setup import get_logger`. Never use `print()`.
5. **Asyncio everywhere**: No threading. All I/O-bound operations are `async`.
6. **Type safety**: `mypy --strict` must pass. All public functions have type annotations.
7. **`torch.no_grad()`**: Required for all inference paths.
8. **`deque(maxlen=N)`**: Required for all sensor ring buffers. `N` comes from config.
9. **Backwards compatibility**: New config fields MUST have defaults. Existing YAML files must load unchanged.

## Project Structure

```
src/mousedroid/
  config/schema.py    # Pydantic v2 root Settings — single source of truth
  factory.py          # DI wiring — only place that imports concrete types
  orchestrator/       # 30 Hz sense-plan-act loop
  world_model/        # RSSM latent dynamics + MCTS planning
  cognitive/          # Dual-cadence BDI + metacognitive loop
  memory/             # Working, episodic, semantic, consolidation
  learning/           # EWC + progressive neural networks
  meta/               # MAML + in-context adaptation
  curiosity/          # ICM intrinsic curiosity
  growth/             # Knowledge distillation
  reward/             # Constitutional multi-objective reward
  scaling/            # Mixture-of-Experts + adaptive compute
  safety/             # Constitutional RL + runtime safety monitor
  arm/                # Robot arm training platform (Tower of Hanoi -> laundry sorting)
    perception/       # Depth camera, YOLO detection, 6-DoF pose, symbolic state
    planning/         # PDDL symbolic planner, LLM replanner, laundry rules
    control/          # SAC+HER policy, grasp/place primitives, trajectory
    environments/     # MuJoCo Gymnasium envs, domain randomization, curriculum
    hardware/         # SO-ARM100 driver, mock driver
  hardware/           # Sensor drivers (camera, ultrasonic, audio)
  harness/            # Agent harness — task tracker, hooks, journal, skills, HITL, replanner (PR #61)
  training/replay/    # Phase 2 real-episode replay loop — async LMDB reader + sim/real mixer (PR #60)
  comms/              # ESP32 serial/WiFi communication
  sensing/            # Sensor fusion
  telemetry/          # REST + WebSocket monitoring server
  resilience/         # Circuit breaker + retry wrappers
  logging/            # structlog setup
  common/tools/       # Tool registry for runtime operations
```

## Configuration System

- **Schema**: `src/mousedroid/config/schema.py` — Pydantic v2 `BaseSettings` with nested models
- **YAML configs**: `config/` directory — `default.yaml` (mouse droid), `robot_arm_default.yaml` (arm platform)
- **Environment variables**: Prefix `MOUSEDROID_`, nested delimiter `__` (e.g., `MOUSEDROID_ARM__DOF=6`)
- **Platform selection**: `platform: mouse_droid` or `platform: robot_arm` in YAML
- **Adding new config**: Create a Pydantic `BaseModel` subclass, add as `Optional` field with `None` default to `Settings`

## Testing

- **Framework**: pytest with `pytest-asyncio`, `pytest-cov`, `hypothesis`
- **Coverage gate**: 85% minimum (`--cov-fail-under=85`)
- **Markers**: `@pytest.mark.slow`, `@pytest.mark.hardware`, `@pytest.mark.smoke`
- **Test structure**: `tests/unit/`, `tests/integration/`, `tests/property/`, `tests/regression/`, `tests/e2e/`, `tests/hardware/`
- **Fixtures**: Global `conftest.py` auto-mocks hardware env. Unit `conftest.py` resets structlog.
- **Optional deps**: Use `pytest.importorskip("mujoco")` for modules requiring arm dependencies
- **Run**: `pytest tests/` or `pytest tests/unit/arm/ -v` for arm-specific tests

## Code Style

- **Linter**: `ruff==0.8.0` — version pinned in `pyproject.toml [dev]` to match `.github/workflows/ci.yml`. Line length 100, Google docstrings, comprehensive rule set.
- **Type checker**: `mypy --strict` with `ignore_missing_imports`
- **Format**: `ruff format` (CI runs `ruff format --check src/ tests/`; same in `bash scripts/ci.sh` post-PR #62)
- **Docstrings**: Google convention, required on all public functions/classes
- **Imports**: `from __future__ import annotations` in every module
- **Pytest invocation**: always pass `--import-mode=importlib` (matches `scripts/ci.sh`); `pytest tests/` works at root because of the auto-loaded `tests/conftest.py`

## CI Pipeline (5 stages)

1. **Lint**: `ruff check` + `ruff format --check` (Python 3.10 + 3.11)
2. **Type check**: `mypy --strict` (depends on lint)
3. **Test + Coverage**: `pytest --cov --cov-fail-under=85` (depends on lint)
4. **Security**: `pip-audit --strict` (advisory)
5. **Docker**: Validate `Dockerfile.jetson` + `docker-compose.jetson.yml` (depends on test + typecheck)

## Robot Arm Platform (Hierarchical Reasoning Architecture)

### Four-Layer Architecture

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Layer 0: Perception | `arm/perception/` | RealSense D435i -> YOLO -> 6-DoF pose -> symbolic PDDL state |
| Layer 1: Symbolic Planning | `arm/planning/` | PDDL planner (Pyperplan) + MCTS + LLM replanner |
| Layer 2: World Modeling | `world_model/` (reused) | RSSM latent dynamics, Dreamer-V3 imagination |
| Layer 3: Motor Control | `arm/control/` | SAC+HER goal-conditioned policy, grasp/place primitives |

### Key Design Patterns for Arm Modules

- **Sim-first**: Train in MuJoCo, transfer to real hardware via domain randomization
- **Curriculum learning**: Progressive difficulty (1 disk -> 3 -> 5 -> 7 disks)
- **HER (Hindsight Experience Replay)**: Relabel failed episodes as successful toward achieved goals
- **Reward shaping**: Configurable weights for grasp, place, collision, completion rewards
- **PDDL optimality**: Tower of Hanoi must produce 2^n - 1 optimal moves

### Reusable Modules (Platform-Agnostic)

These existing modules work for both mouse_droid and robot_arm platforms:
- `world_model/` — RSSM + MCTS (latent dynamics, planning)
- `safety/` — Runtime safety monitor (joint limits, E-stop for arm)
- `reward/` — Multi-objective reward framework
- `memory/` — Episodic replay (HER buffer), semantic memory
- `curiosity/` — ICM intrinsic reward for exploration
- `learning/` — EWC continual learning (Hanoi -> laundry transfer)
- `resilience/` — Circuit breaker + retry for hardware drivers
- `common/tools/` — Tool registry for arm calibration, diagnostics

## Common Commands

```bash
# Development setup
pip install -e ".[dev]"

# With robot arm dependencies
pip install -e ".[dev,arm]"

# Run all tests
pytest tests/ -v

# Run arm-specific tests
pytest tests/unit/arm/ -v

# Lint + format check
ruff check src/ tests/ && ruff format --check src/ tests/

# Type check
mypy --strict src/mousedroid/

# Full local CI
bash scripts/ci.sh
```

## Validation surface (smoke-stability sprint)

When adding hardware probes or pillar checks, follow the established split:

- **`src/mousedroid/validation/preflight.py`** — async `run_preflight(cfg)` for hardware probes. Add a check by writing an `async _check_<name>(cfg) -> PreflightCheckResult` and registering it in `_CHECK_DISPATCH`. The dispatcher catches per-check exceptions and records them as `FAIL` so a single misbehaving driver never crashes the operator runbook.
- **`src/mousedroid/validation/pillars.py`** — async `validate_all_pillars(cfg)` over the 10 AGI pillars. Two patterns:
  - **Pattern A (factory builder)**: when a `build_<pillar>` factory exists, instantiate + smoke-assert. **Use explicit `if x is None: return _fail(...)` instead of `assert x is not None`** — assert is stripped under `-O` (PYTHONOPTIMIZE=1, the default Jetson Docker entrypoint).
  - **Pattern B (pytest delegation)**: when no factory builder exists yet, delegate to the pillar's existing unit-test module via in-process `pytest.main`. Paths in `_PYTEST_DELEGATION_PATHS` are repo-relative and resolved against module-level `_REPO_ROOT = Path(__file__).resolve().parents[3]` so the dispatcher works regardless of caller CWD.
- **`src/mousedroid/cli/{preflight,validate_pillars}.py`** — argparse wrappers over the async APIs. CLI exit-code contract: `0` on `OK` or `DEGRADED` (WARN-only); `1` only on `FAIL`. Don't return `1` on `DEGRADED` — CI uses these codes as the canonical "is the dispatcher broken?" signal.
- **`tools/lidar_telemetry_probe.py`** — standalone non-orchestrator probe for verifying the telemetry → dashboard pipeline when the rover is partially detached (ESP32 / CSI absent). Binds a non-default port (8090) so it never collides with a running orchestrator.

When adding ribbon-disconnect-style operator-actionable diagnostics, distinguish **WARN** (operator can fix without a code change) from **FAIL** (driver crash / wrong cfg / permission denied). Dashboards rely on this distinction.

## Dashboard live-verification surface (PR #104 — dashboard-stability sprint)

When working on the live-Jetson dashboard path, the schema-driven escape
hatches are non-negotiable contracts:

- **`ESP32Config.enabled: bool = True`** — flip to `False` (YAML or
  `MOUSEDROID_ESP32__ENABLED=false` env) to make `build_esp32_driver`
  resolve to `MockESP32Driver` even with `mock_hardware=False`. The
  resilience wrapper stays in place; only the *inner* driver changes.
  This avoids the prior workaround of monkey-patching
  `orchestrator.start()` to swallow connect failures.
- **`CameraConfig.v4l2_grayscale_extract: bool = True`** — IMX708 Bayer
  workaround for the V4L2 fallback path (the container ships without
  `nvarguscamerasrc`). When `True`, `capture_raw_jpeg` extracts the green
  channel as luma + clones to RGB so the operator sees the scene with
  mosaic artefacts instead of solid green. Flip to `False` once the
  container gains the GStreamer plugin.
- **`CameraConfig.snapshot_jpeg_quality: int = 90`** — Pillow quality for
  the snapshot path used by `scripts/verify_sensors.py --save-frame`.
  Range-gated `1..100` by Pydantic.

**Workstation reverse proxy:** `tools/dashboard_proxy.py` is the canonical
bridge from a Windows / macOS browser to the auth-gated Jetson telemetry
server. It accepts CLI positional args (`port upstream [token]`) AND env
vars (`PROXY_PORT` / `JETSON_HTTP` / `JETSON_TOKEN`) — never hardcode the
token. Three transport modes supported: plain HTTP, streaming (MJPEG /
SSE), WebSocket. Hop-by-hop headers (RFC-9110 §7.6.1) are stripped before
forward; the proxy injects the configured Bearer token at the upstream
edge so the browser never sees it. WebSocket forwarding uses
`asyncio.wait(..., return_when=FIRST_COMPLETED)` + explicit task
cancellation to avoid pool-slot leaks on one-sided close.

**Test surface mirror:** every dashboard-touching change should land
under the matching tier:

| Tier | Directory | When to add |
|------|-----------|-------------|
| Unit | `tests/unit/` | Single-function behaviour, mocked deps |
| Integration | `tests/integration/test_pr*_integration.py` | Multi-module wiring through the factory |
| E2E | `tests/e2e/test_pr*_e2e.py` | Full request path through proxy / camera / driver chain |
| Regression | `tests/regression/test_pr*_backwards_compat.py` | YAML / env / default-value invariants |
| AQA | `tests/regression/test_pr*_aqa.py` | Schema-field hygiene + protocol conformance |
| Sanity | `tests/smoke/test_pr*_sanity.py` | Sub-second import + parse smoke |
| Hardware | `tests/hardware/test_pr*_<surface>.py` | `@pytest.mark.hardware`-gated, runs on rover only |

The PR #104 test files are the reference implementations — copy their
docstring style + skip-gate pattern (`tests/_jetson_hardware.is_jetson_host`)
when adding new ones.

See `AGENTS.md` (agentic-worker behavioural contract) and `SKILLS.md`
(capability index keyed by trigger phrase) for additional context.
