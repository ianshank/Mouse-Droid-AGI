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
  commentary/         # Phase-0 grounded, novelty-gated spoken commentary (out-of-loop)
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

## USB-C smoke validation surface (PR #106 — rover smoke-stability sprint)

The Wave Rover USB-C wiring must be discoverable and stable across rover
swaps. Three non-negotiable contracts encode this:

- **`USBCDiscoveryConfig.enabled: bool = False`** — master switch lives in
  `src/mousedroid/config/schema.py`. Defaults `False` so pre-PR YAML files
  load unchanged. Flip to `True` only on the Jetson production overlay,
  where `required_endpoints` declares the named cables the rover expects.
  When `enabled=True` with an empty list, `_require_endpoints_when_enabled`
  raises at YAML-load time — a misconfigured gate never silently passes.
- **Factory override chain** — `_resolve_esp32_serial_via_usbc_discovery`
  in `src/mousedroid/factory.py` (commit `34ab760`) supersedes
  `cfg.esp32.serial_port` with the live `rover_esp32` by-id path **only
  when** (a) discovery is enabled AND (b) the literal `serial_port` does
  not exist on disk. A pinned, valid `serial_port` always wins so an
  operator override is never silently shadowed. Log the override fire
  via the `esp32_serial_port_overridden` structured event.
- **`ESP32Config.smoke_test_velocity_mps: float = 0.05` (`ge=0`)** —
  setpoint for the power-chain probe in
  `src/mousedroid/diagnostics/power_chain.py`. The `ge=0` (not `gt=0`)
  bound lets operators express a permanent zero-motion safe-bench config;
  the runtime `allow_motion` gate in `assert_power_chain` is still
  authoritative regardless of this setpoint.

**USB-C boot-race guard:** `enumerate_usbc_devices` and `resolve_endpoint`
in `src/mousedroid/diagnostics/usbc.py` MUST guard `Path.glob` against a
missing `by_id_root` directory. Without the guard, a pre-udev call (boot
race during container startup) raises `FileNotFoundError` and crashes the
smoke harness. The guard surfaces every required endpoint as `MISSING`
instead — the harness sees a structured FAIL list, not an unhandled
exception.

**Serial driver decode hygiene:** `SerialESP32Driver._read_line` in
`src/mousedroid/comms/serial_driver.py` MUST decode with
`errors="replace"`. A garbled byte (firmware churn, brown-out, UART
noise) under the default strict codec raises `UnicodeDecodeError` out of
the `asyncio.to_thread` wrapper, bypassing the adaptive-timeout state
machine. With `errors="replace"`, the replacement char flows into
`json.loads` and the existing `esp32_non_json_response` warning path
handles it cleanly.

**E2E inline scripts:** every smoke / e2e bash one-liner that asserts
factory-builder return types MUST use explicit `if not isinstance(x, ...):
raise RuntimeError(...)`, NOT `assert isinstance(x, ...)`. The Jetson
Docker entrypoint sets `PYTHONOPTIMIZE=1` which strips asserts. The
`scripts/jetson_smoke_test.sh` E2E stage demonstrates the correct shape.

**Operator triage:** see `docs/runbooks/jetson-rover-smoke.md` for the
warm-vs-cold smoke discipline, rover-swap by-id-drift symptom, and the
canonical structlog grep recipes (`usbc_endpoint_*`,
`esp32_serial_port_overridden`, `power_chain_probe_complete`,
`esp32_raw_line`). The C4 component diagram for the smoke gate is at
`docs/architecture/c4-usbc-smoke.md`.

## LLM gateway + cloud/local failover (PR #107 — Tier C-rover deliberative brain)

The deliberative mission-translation path now supports cloud Claude as
the primary brain with a local model as the off-network fallback. The
30 Hz reactive control loop (RSSM → MCTS → ESP32) stays deterministic
and LLM-free; only the natural-language → `GoalVector` translation goes
through this layer, OUTSIDE the hot loop.

Non-negotiable contracts:

- **`LLMConfig.backend: Literal["llama_cpp", "openai_compatible", "anthropic"]`**
  — single dispatch key. `llama_cpp` (default) preserves byte-identical
  legacy behaviour; `anthropic` selects the Claude Messages API
  backend; `openai_compatible` reaches any OpenAI-shaped HTTP endpoint
  (local Ollama / LM Studio / cloud).
- **`LLMConfig.fallback_backend: Literal["none", "llama_cpp", "openai_compatible"]`**
  — local-only fallback. Setting `anthropic` here is rejected at YAML
  parse time (cloud-to-cloud failover is anti-pattern — you'd lose the
  off-network autonomy). Default `none` skips the composite.
- **`LLMConfig.fallback_retry_cooldown_s: float = 30.0`** — operator-
  tunable seconds before the composite re-probes a degraded primary
  (mobile rover WAN dropouts). Lower (e.g. 5 s) for lab bench tests,
  higher (120 s) behind a flaky uplink. **Threaded through the factory**
  — `tests/unit/factory/test_build_llm_gateway_dispatch.py` pins this so
  a future refactor that swapped the kwarg for a literal fails fast.
- **`LLMConfig.api_key: SecretStr | None`** — Pydantic `SecretStr` wraps
  the Anthropic API key. **NEVER** `.get_secret_value()` in a log call
  or exception message. The factory passes the resolved value to the
  SDK constructor ONCE; everywhere else the wrapper masks repr to
  `SecretStr('**********')`. Operators supply the key via
  `ANTHROPIC_API_KEY` env var (preferred — SDK resolves natively) or
  `MOUSEDROID_LLM__API_KEY` schema-mapped override.
- **Prompt-injection filter pre-egress.** The `RegexInjectionFilter`
  MUST `.sanitize()` the NL command BEFORE any `messages.create` call
  reaches `api.anthropic.com`. This is the only place rover NL goes
  third-party — the filter envelope is what stops
  `"ignore all instructions and..."`-shaped commands from leaving the
  rover. Pinned by `tests/unit/llm_gateway/test_anthropic_gateway.py`.
- **`asyncio.CancelledError` propagates** in BOTH gateways and in the
  composite — explicit `except asyncio.CancelledError: raise` before
  the broad `except Exception`. The composite's "never raises on
  backend failure" contract is for *backend* failures, NOT for
  cooperative task cancellation. The composite also stamps
  `_last_primary_attempt` AFTER the await returns (not before) so a
  cancelled probe never poisons the cooldown timer.
- **Markdown-fence JSON resilience.** `_JSON_OBJECT_RE` extracts the
  first `{...}` span before `json.loads` — Claude routinely wraps
  responses in ```` ```json ... ``` ```` fences despite system-prompt
  instructions. Dict-shaped response blocks (mocks + alternative SDK
  clients) are also handled by `_extract_text`.
- **Concurrent + safe lifecycle.** `FallbackLLMGateway.start()` and
  `stop()` both use `await asyncio.gather(..., return_exceptions=True)`
  — cold-boot time is `max(T_primary, T_secondary)` (not the sum), and
  a primary start/stop crash never skips the secondary's cleanup.

**Operator deployment:** see `config/jetson_claude_pilot.yaml` for the
canonical anthropic-primary + llama_cpp-fallback overlay. Note
`latency_target_ms: 5000.0` — the default 500 ms (calibrated for local
GGUF) would spam `anthropic_gateway_slow` warnings on every normal
cloud round-trip.

**Architecture diagram:** `docs/architecture/c4-llm-gateway.md`.

## Live deployment + CI-gate contracts (PR #111/#112/#113 — Claude-pilot rover deploy)

The PR #107 gateway is now LIVE on the Jetson rover. Deploying it surfaced
deployment + CI-gate invariants an agent MUST respect:

- **`deployments/jetson-image.json` must point to a REACHABLE trunk
  commit** whose `config/schema.py` the deployed image actually has. The
  `config-compat` CI gate (`.github/workflows/config-compat.yml`) does
  `git worktree` against this `sha` and validates every changed
  `config/*.yaml` against that historical schema. NEVER pin a squash-source
  feature commit — it becomes unreachable once the feature branch is
  deleted post-merge, and the gate dies repo-wide. Update this record
  whenever the image is rebuilt OR the rover's tracked source SHA changes.
- **NEVER write a literal GitHub expression token `${{ ... }}` inside a
  workflow `run:` block — even in a comment.** GitHub evaluates the
  expression regardless of comment context; an empty `${{ }}` is an
  "invalid workflow file" startup failure that silently disables the whole
  workflow (this is the exact bug that left `config-compat` dead). The
  pinned `actionlint` job (CI Stage 0, `docker://rhysd/actionlint:1.7.12`,
  config `.github/actionlint.yaml` declaring the custom `jetson` runner
  label) now guards this.
- **Per-host rover overrides live ONLY in `/etc/mousedroid/docker.env`**
  (`MOUSEDROID_LLM__ENABLED=true`, `MOUSEDROID_LLM__N_GPU_LAYERS=0` — Phi-3
  fallback on CPU because the world model owns the shared iGPU). NEVER
  commit these; `config/docker.env.example` documents the secret surface
  (the `ANTHROPIC_API_KEY` slot included) without holding live values.
- **`scripts/translate_mission.py` is the operator dry-run probe** —
  NL→`GoalVector` via `build_llm_gateway` + `resolve_runtime_config_paths`,
  no motors engaged. It validates the deliberative path end-to-end without
  the (dead) ESP32. Use it to confirm cloud→local failover before a
  mission.
- **Test isolation:** NEVER `patch.dict("sys.modules", ...) +
  importlib.reload` a module that imports `cv2` (PR #112). The reload
  evicts the real `cv2` from the import cache and poisons every later test
  in the same process under `pytest tests/`. Use `patch.object` on the
  specific symbol instead.

The Dockerfile.jetson Stage 4b installs the `anthropic` SDK non-fatally
(the cloud tier survives `--force-recreate`); `config/jetson_production.yaml`
carries the `llm:` block (Claude-haiku primary + Phi-3 `llama_cpp`
fallback). Operator runbook: `docs/runbooks/jetson-claude-pilot-deploy.md`.

See `AGENTS.md` (agentic-worker behavioural contract) and `SKILLS.md`
(capability index keyed by trigger phrase) for additional context.
