# MouseDroid — Claude Code Project Instructions

## Overview

MouseDroid is a Star Wars MSE-6 autonomous-navigation system running on an NVIDIA Jetson Orin Nano (with a parked hierarchical robot-arm training platform) — an edge-AI / robotics engineering project, not a claim of general intelligence. Its cognitive stack is organised around a "10 Pillars of the Ideal Neural Network" research framing used as an engineering compass: every pillar is real, unit-tested code, and the honest axis is integration — seven pillars (world model, cognitive, memory, continual learning, reward, safety, curiosity) are wired into the 30 Hz runtime loop (curiosity via the memory subsystem), while three (meta, growth, scaling) are implemented and tested but not yet wired in.

> **Governance:** `docs/CHARTER.md` is the project constitution (vision, scope, invariants, roadmap) and sits above this document. When a change touches scope or an invariant, defer to the charter.

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
- **`src/mousedroid/validation/pillars.py`** — async `validate_all_pillars(cfg)` over the 10 cognitive pillars. Two patterns:
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
| Property | `tests/property/` | Hypothesis-driven invariants over input space |
| Performance | `tests/performance/` | Latency / throughput budget assertions |
| Sanity | `tests/smoke/test_pr*_sanity.py` | Sub-second import + parse smoke |
| Hardware | `tests/hardware/test_pr*_<surface>.py` | `@pytest.mark.hardware`-gated, runs on rover only |

The **Property** (`tests/property/`) and **Performance** (`tests/performance/`)
tiers already exist and run in `scripts/ci.sh` (property folds into the unit +
property + integration coverage stage; performance runs as its own stage) — they
are part of the mirror, not optional extras.

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

## LLM-gateway observability (PR #115 — Prometheus token/latency/served/budget metrics)

The cloud Claude tier now exports four config-gated Prometheus families
(namespaced via `cfg.metrics.namespace`). Non-negotiable contracts:

- **One flag, pure-add.** `MetricsConfig.track_llm_gateway: bool = True` gates
  all four families: `{ns}_llm_tokens_total{model,token_type}` (counter),
  `{ns}_llm_gateway_latency_ms` (label-free histogram, buckets from
  `MetricsConfig.llm_gateway_latency_buckets_ms`),
  `{ns}_llm_gateway_served_total{tier,outcome}` (counter), and
  `{ns}_llm_latency_budget_exceeded_total{model}` (counter). Families are
  **omitted from `/metrics` until first write** — a registry built without LLM
  activity (or with `metrics=None`) renders byte-identically to pre-feature.
- **Keep `anthropic_gateway_slow`.** The budget counter increments on the SAME
  branch as the existing `anthropic_gateway_slow` log — **do NOT rename the log
  event** (≥8 docs reference it). The metric name carries the budget semantics.
- **Shared registry threaded keyword-only.** `build_orchestrator` passes ONE
  `build_metrics_registry(cfg)` to BOTH `build_telemetry_server` AND
  `build_llm_gateway(..., metrics=…)`, so a translation through the running
  orchestrator surfaces on `/metrics`. The `metrics` param is keyword-only,
  defaults `None` (byte-identical legacy construction — pinned by
  `tests/unit/factory/test_build_llm_gateway_dispatch.py`).
- **Record on SUCCESS only**; `asyncio.CancelledError` propagates untouched.
  The per-tier served counter lives in `FallbackLLMGateway` (the only place that
  knows which tier answered).
- **Label-cardinality guard.** `inc_llm_tokens` / `inc_llm_gateway_served`
  validate against module-level frozensets (`_LLM_TOKEN_TYPES` /
  `_LLM_SERVED_TIERS` / `_LLM_SERVED_OUTCOMES` in `telemetry/metrics.py`) and
  **drop** out-of-set values with a DEBUG log — never label by mission text.
- **`generate_metrics_sample()` seeds all four** (promtool contract).
- **`_extract_token_usage` is defensive:** OUTER
  `getattr(response, "usage", None)` then inner `getattr`/`.get` for
  `input_tokens`/`output_tokens` — production NEVER touches `response.usage`
  directly (fakes/alt-SDK clients lack it).

Architecture diagram: `docs/architecture/c4-llm-gateway.md` (Observability
section).

## Full Jetson on-device validation (PR #116)

`scripts/jetson_full_validation.sh` is the single consolidated entry point that
**composes** the existing tooling (no duplication) into three phases:
static-CI → cold-hardware → warm-live. Contracts:

- **Cold-then-warm.** Phase 2 `docker stop`s the container for exclusive-device
  sensor + smoke checks (host venv) and a `trap` ALWAYS restarts it on exit —
  never leave the rover brain down. Phase 3 runs the warm checks (`/api/v1/health`,
  the auth-exempt live `/metrics` scrape, `translate_mission`, the LiDAR→WS probe,
  structlog greps) against the running container.
- **Validate-around the dead ESP32.** `serial`/`motor`/`power` smoke steps are
  **non-blocking** (WARN, not FAIL); the orchestrator e2e and hardware pytest run
  with `MOUSEDROID_ESP32__ENABLED=false`; **no motion is ever armed**.
- **No hardcoded values.** Every tunable is env-overridable: container name,
  report root, telemetry URL, config, lidar probe port/duration, health
  retries/interval, curl/pytest timeouts, log tail, and the metric namespace
  (`MOUSEDROID_METRICS__NAMESPACE`, mirroring the schema field). Secrets
  (`ANTHROPIC_API_KEY`, `MOUSEDROID_TELEMETRY_TOKEN`) are **presence-checked
  only — never echoed**.
- **Phase-1 ci.sh OOM guard (PR #161).** Jetson has 7.4 GB RAM; a running
  mousedroid daemon plus container ci.sh + pytest + coverage + torch + LMDB
  routinely SIGKILL'd (rc=137). `run_phase1_ci_container` in the wrapper
  applies `ulimit -v ${PHASE1_CI_ULIMIT_KB}` (default 6 GB) so Python raises
  MemoryError first; on rc=137 it retries once under
  `${PHASE1_CI_RETRY_ULIMIT_KB}` (default 5 GB) + `MOUSEDROID_CI_SLIM=1`,
  which makes ci.sh skip Performance / Regression / E2E stages (memory-
  heaviest). Records **WARN** on retry-success (never silent PASS). Retry is
  gated by `${PHASE1_CI_OOM_RETRY}` (default 1) so operators can opt out.
  Perf/Regression/E2E coverage is NOT lost — `jetson_full_validation.sh`
  Phase 2 already runs `pytest -m hardware` in a dedicated tier that owns
  the peripherals. Contract pinned by
  `tests/regression/test_jetson_phase1_oom_guard.py` (17 source-text pins);
  a future edit that removes the ulimit, unlocks the retry to non-137 rcs,
  or unwraps the Unit+Property+Integration+coverage core signal from the
  mandatory path will fail those tests.
- **`scripts/ci.sh` hardware-marker filter (PR #160).** The unit + property +
  integration, performance, and regression pytest stages all pass `-m "not
  hardware"` so `@pytest.mark.hardware`-marked tests (like
  `test_build_distance_sensor_real_hardware` which opens real GPIO) do NOT
  collect on hosts that don't own the peripherals — critical when the rover's
  `mousedroid` service is holding the GPIO line during Phase-1 container ci.sh.
  Hardware coverage still runs in Phase 2's `pytest -m hardware` tier.
- **Live `/metrics` population is proven in-process, not over HTTP.**
  `config/jetson_production.yaml` has no `openclaw:` block, so
  `POST /api/v1/mission` is unregistered — nothing drives the gateway over the
  wire. `tests/hardware/test_llm_gateway_metrics_live_jetson.py::test_inprocess_mission_populates_metric_families`
  builds the real orchestrator and drives a **guaranteed-UNKNOWN** command
  (`"navigate to the cantina"` — rule-parsed commands like `"go forward"` /
  `"patrol …"` never reach the LLM) through `process_mission`, asserting
  `orch._metrics` renders the families on live Claude. Operator runbook:
  `docs/runbooks/jetson-full-validation.md`.

## Validation-efficiency surface (latency stats + trend store + phase caching)

Runtime/resource-efficiency tooling layered on top of the existing validation
harness. Three additive, opt-in contracts — nothing changes the default
single-shot / full-run behaviour:

- **`src/mousedroid/validation/latency_stats.py`** — pure, deterministic
  `summarize(samples_ms) -> LatencySummary` (min/mean/p50/p95/p99/max) shared by
  the `tools/` probes. No I/O, no clock reads, no verdict — the caller gates
  against its config-supplied target. `llm_latency_probe.py --iterations N`
  (default 1 = byte-identical legacy single-shot gate; >1 emits
  `llm_latency_summary` and gates on **p95** to absorb cloud/GPU tail variance).
  `lidar_telemetry_probe.py` now emits `lidar_frame_interval_summary` (inter-
  arrival jitter — a high p95/p99 vs p50 means dropped/bunched dashboard frames).
  The pure `intervals_ms(timestamps_s)` helper (timestamps→inter-arrival gaps)
  lives here too so the jitter maths is unit-tested, not inlined in the probe.
  **Import-decoupling contract:** `mousedroid/validation/__init__.py` re-exports
  the heavy `runtime` sensor helpers (numpy/cv2/pyaudio) **lazily** via :pep:`562`
  `__getattr__`, so importing the pure modules never drags the sensor stack into
  the process (locked by `tests/regression/test_validation_import_decoupling.py`).
  Keep new pure helpers dependency-free; never add an eager `runtime` import to
  `__init__`.
- **`src/mousedroid/validation/report_store.py`** — persists each
  `PreflightReport` to the **existing** harness journal (no parallel store) as a
  `preflight_report` event, and `detect_regressions(history)` compares the two
  newest runs for status downgrade / new FAIL / latency creep (gated by BOTH a
  `slow_ratio` and an absolute `slow_floor_s` so sub-50 ms checks don't false-
  fire). Wired via `mousedroid.cli.preflight --journal-path PATH` (opt-in record)
  + `--trend` (print regressions; exit 1 on regression). `recorded_at_ns` is
  wall-clock `time.time_ns()` — stable across reboots, unlike the journal's
  monotonic entry stamp. Construct the JSONL config with
  `HarnessJournalConfig.model_validate({...})` (NOT direct construction — mypy
  strict treats Field-defaulted args as required without the pydantic plugin).
- **F-018 trend closure (PR #151):** the full-validation harness now FEEDS the
  trend store — Phase-2 preflight appends via `--journal-path/--trend/
  --journal-max-bytes` (journal under REPORT_ROOT, single-generation rotation,
  `max_bytes<=0` = disabled, failed replace degrades to `journal_rotate_failed`);
  Phase-4 SUMMARY.md is rendered by `validation/summary.py` (pure, tested) with a
  **Trend** section, `scripts/render_validation_summary.py` as the shim and a
  bash `write_summary_fallback` for python-less hosts; `mousedroid-trend.{service,
  timer}` samples hourly with a regression-pinned NON-EXCLUSIVE check subset
  (`config,host_env_keys`) and a SEPARATE journal path — never point the timer at
  camera/lidar/esp32/audio and never share the full-run journal.
- **`scripts/jetson_full_validation.sh` Phase-1 caching** — Phase 1 (static CI)
  is a pure function of the committed source. `git_clean_sha` echoes the HEAD
  sha ONLY when the tree under `src/tests/scripts/config/pyproject.toml` is
  clean; a dirty tree returns empty → forced cache miss → never masks an
  uncommitted edit. A matching cached sha SKIPs Phase 1 (recorded
  `PASS "static CI (cached)"`); the cache is written only on a fully-green run.
  `--no-cache` forces re-run; `--phases 0,1,3` runs an ordered subset (`--phase`
  kept as the single-phase alias). Cache lives under
  `<report-root>/.cache/phase1_pass_sha` (gitignored). Hardware (Phase 2) + live
  (Phase 3) are NEVER cached.

## Unified dashboard + sensor-fusion summary (full rover bring-up)

The telemetry server serves a single overview page at `/` (→ `/dashboard`)
rendering camera + lidar + sensor-fusion + status, reachable over WiFi.
Contracts:

- **`TelemetryFrame.fused` is purely additive.** It defaults to an empty dict
  (mirrors `sensor_liveness`); `to_dict()` is `asdict`, so it auto-serialises on
  `/ws`. A registry/frame built without an observation is byte-identical to
  pre-feature. Shape: `{n_valid, n_modalities, lidar_present, modalities{…},
  fused_norm}`.
- **The `valid_mask` is length 4 (no lidar) OR 5 (with lidar)** — slot order
  `[vision, ultrasonic, motor, audio, (lidar)]` (`constants.N_SENSOR_MODALITIES`
  / `_WITH_LIDAR`). `_build_fused_summary` in `telemetry/frame_builder.py` zips
  `_MODALITY_NAMES` against the ACTUAL length and never indexes a fixed slot —
  a 4-element mask must never IndexError. It is a pure function of the frame's
  existing inputs (no sensor reads, no hot-loop cost) — NOT a new fusion
  algorithm.
- **Dashboard has no hardcoded host/port/token.** `static/dashboard.html`
  derives its origin from `window.location` and carries the bearer token via
  `?token=` (the existing `/camera`+`/lidar` pattern), persisting it to the WS +
  MJPEG URLs. `/dashboard` stays behind the auth middleware.
- **Real-motor bring-up is probe-first.** A dead ESP32 with `enabled=True` makes
  `orchestrator.start()`→`esp32.connect()` retry-then-raise → the container
  crash-loops. The bring-up runbook probes the ESP32 first and only keeps motors
  live if it responds; otherwise `MOUSEDROID_ESP32__ENABLED=false`. Operator
  runbook: `docs/runbooks/jetson-full-bringup.md`.

## Operator Q&A + full backend telemetry (PR #118)

The deliberative path gained an operator natural-language Q&A route, and the
telemetry backend now surfaces the full sensor/LLM/served picture. As with every
deliberative surface, the 30 Hz reactive loop stays LLM-free — Q&A runs OUTSIDE
the hot loop through the same gateway + prompt-injection-filter envelope as
mission translation, and answers are served on the unified dashboard.

## Skill-validation surface (rover-hardening sprint)

Two skill families are now validated, each pinned by its own test so neither can
silently drift:

- **`.claude/skills/<name>/SKILL.md` (project skills; migrated from the legacy
  `.claude/commands/*.md` layout per foundry plan WS-F7a — the legacy dir stays
  deleted)** — `tools/validate_skill_commands.py`
  is the reusable library + CLI. Per skill file it asserts the YAML front-matter
  carries a non-empty `description`, every backtick-wrapped repo path it
  references actually exists, and no hardcoded host/IP leaks in. Paths are
  **discovered** from the body (never enumerated) and format/glob tokens
  (`{}`, `*`, `$`, `<>`) are excluded so illustrative patterns like
  `weights/arm/{task}_final.pt` are not false-flagged. The contract is pinned by
  `tests/regression/test_skill_commands_aqa.py` (the PR gate — runs in the
  regression stage) and mirrored as a fast local signal in `scripts/ci.sh`
  (`tools/` is now in the `ruff check` / `ruff format --check` scope too).
- **Builtin `SkillSpec` ↔ publishable doc pairing** — `tests/unit/skills/builtin/test_skill_specs_match_docs.py`
  enforces the pairing the `src/mousedroid/skills/builtin/__init__.py` docstring
  has long promised: every builtin spec has a `docs/openclaw_skills/<name>/SKILL.md`
  whose H1 (`# <name>`) names the skill, and every published doc maps back to a
  registered spec (no orphans). It asserts the H1, **not** YAML front-matter —
  these publishable docs intentionally have none.
- **Optional `status:` front-matter lifecycle (PR #151)** — skills may carry
  `status: active|frozen|deferred` (+ a free-form `unfreeze:` note); the
  validator flags any OTHER value as `invalid-status` so a typo like `forzen`
  can't silently un-freeze a paused skill. Absent `status` stays valid
  (backwards compatible with `.github/skills/` and external layouts). The three
  arm/sim skills are `frozen` pending the F-008 hardware gate + 30-day soak.

## On-device incremental learning (Phase 6 — functional, default-OFF, soak-gated)

The rover can refine its own **RSSM world model** *between* cloud retraining
cycles from fresh on-device experience — safe by construction and default-OFF.
The WS-E2/E3 learning + gate seams are **CLOSED** (#135): the learner refines the
**live RSSM** (not a config-sized stand-in) and the gate scores it on a **held-out
replay batch** by recon+KL loss (the `score_policy`/`PolicyProtocol`/seed-states
diagnostic scaffolding was retired post-WS-E3). The 30 Hz reactive loop stays
training-free; the bounded refinement + gate run at the slow-cadence / POST_TICK
seam OUTSIDE the hot loop, all torch work offloaded via `asyncio.to_thread`.
**Still soak-gated** — keep `enabled`/`enable_hot_swap` off on the live rover
until a soak gate passes. Non-negotiable contracts:

- **Default-OFF `Optional`/`None`.** `OnDeviceLearningConfig` is an `Optional`
  field on `Settings` (default `None`; `enabled: bool = False`). Absent or
  disabled ⇒ `build_on_device_coordinator` returns `None`, no background task
  is spawned, and the orchestrator is byte-identical to pre-Phase-6. New fields
  keep defaults so existing YAML loads unchanged.
- **Counter is pure-add + gated.** `{ns}_on_device_learning_reverted_total{reason}`
  (`telemetry/metrics.py`) is gated by `MetricsConfig.track_on_device_learning`
  (default `True`) and omitted from `/metrics` until the first revert. `reason`
  is a low-cardinality frozenset `_ON_DEVICE_REVERT_REASONS`
  (`regression_bound`, `integrity_mismatch`, `exception`) — out-of-set values
  are dropped with a DEBUG log; seeded in `generate_metrics_sample()`. Keep the
  `on_device_candidate_reverted` log event name (docs reference it).
- **`slot_dir` resolved under the experience root + validated.** The candidate
  slot is `<ExperienceConfig.path>/<slot_dir>/<digest>.pt` — NEVER an absolute
  host path. A `field_validator` rejects absolute / `..`-traversal / empty
  `slot_dir` at YAML-load. SHA-256 integrity (ADR-010 / `verify_sha256`) is
  reused: the digest stamps the filename and is re-verified on load
  (`integrity_mismatch` on failure). On-device weights NEVER overwrite the
  cloud-pulled slot.
- **Hot loop untouched.** The coordinator (`learning/on_device/replay_trigger.py`)
  offloads the trigger probe, batch load, learner update, AND the gate scoring
  via `asyncio.to_thread`. The base model is deep-copied before any gradient
  flows (base bitwise-unchanged); the candidate is a separate object.
- **RSSM-vs-RSSM recon-loss gate + auto-revert is authoritative (WS-E3).**
  `RegressionGate.evaluate` (`regression_gate.py`) scores the candidate RSSM and
  the live baseline RSSM by their held-out **reconstruction+KL loss** on a SHARED
  FIXED `(B, T, ...)` batch with SHARED decoders + `scoring_seed`, via the
  deterministic `score_dynamics` (`scoring.py`) — **LOWER IS BETTER**. PROMOTE iff
  `candidate_loss` is finite AND `candidate_loss <= baseline_loss +
  regression_tolerance` (marks the slot active); otherwise REVERT + increment the
  counter. This REPLACED the retired self-gaming imagined-return metric (it summed
  the model's OWN `reward_head`, so reward-head inflation gamed it — WS-E-SPIKE).
  The metrics param to `build_on_device_coordinator` is keyword-only (defaults
  `None`).
- **WS-E2/E3 refine the LIVE RSSM (λ=0).** The learner (`RSSMRefiner`,
  `rssm_refiner.py`) deep-copies the live RSSM and refines the candidate via
  `train_sequence` over a replay sequence batch using an `autograd.grad`
  manual-SGD loop (`allow_unused=True` is MANDATORY — `reward_head`/`prior`/
  `observation_decoder` are off the recon/KL graph). **λ=0: no EWC penalty**
  (`ewc_lambda` accepted but ignored; `held_out_fraction` is a future seam not
  wired into the gate). The throwaway `RawModalityDecoders` are refined jointly
  but NEVER persisted — only the refined RSSM `state_dict` round-trips into a
  fresh `build_world_model`. The gate's held-out batch is built ONCE over a slice
  DISJOINT from the refine window; too few records ⇒ logged no-op
  (`on_device_gate_skipped_no_held_out_batch`).
- **Trigger arms on NEW records, not store size.** `build_on_device_coordinator`
  keeps an in-memory `consumed_offset`; `count_new_records` counts beyond it and
  the coordinator's `on_consumed` advances it after a fired cycle, so the trigger
  DISARMS until fresh experience accumulates (else it re-fires the refine+gate
  every cadence on stale data). Per-process — resets on restart.
- **WS-E4 hot-swap is off-loop + default-OFF.** `OnDeviceWeightUpdateSource`
  (`hot_swap.py`, wired only when `enable_hot_swap=True`) materialises a promoted
  slot OFF the hot loop (`refresh_once` via `asyncio.to_thread`: re-verify
  SHA-256 fail-closed → `inc("integrity_mismatch")`; build + device-place); the
  in-`tick()` `take_materialized` is a PURE ref lookup. A newer digest evicts the
  prior unacknowledged pending engine (`_evict_pending_engine`) so the cache
  never strands an unreachable engine. When wired it SUPERSEDES the cloud OTA
  world-model poller (`on_device_hot_swap_supersedes_cloud_world_model_poller`).
- **Determinism is load-bearing.** `score_dynamics` + `RSSMRefiner.update`
  capture/restore the CPU AND CUDA RNG keyed on `torch.cuda.is_available()` ALONE
  (NOT the candidate's device — `torch.manual_seed` reseeds EVERY CUDA generator
  regardless of where the model lives, so a CPU candidate on a GPU host must still
  restore CUDA RNG or it leaks a reseed into the shared process stream); the
  gate-runner's decoder-init seed is confined to
  `torch.random.fork_rng(devices=range(torch.cuda.device_count()))` (forks ALL
  CUDA generators, not just device 0); the refiner builds recon heads on the
  candidate's device. Same `scoring_seed` + inputs + weights ⇒ byte-identical loss
  ⇒ reproducible promote/revert.
- **Grep events** (live RSSM path, NOT the unwired #134 `ewc_online`
  `on_device_update_*`): `on_device_refine_start`/`_complete`,
  `on_device_dynamics_score_computed`, `on_device_candidate_promoted`/`_reverted`
  (keep these names — docs reference them), `on_device_consumed_offset_advanced`,
  and (hot-swap) `on_device_hot_swap_pending`/`_slot_integrity_mismatch`/
  `_refresh_failed`/`_supersedes_cloud_world_model_poller`. The full pipeline
  (promote + revert + `_tick_count==0`) is pinned by
  `tests/integration/test_on_device_sim_soak.py`. See
  `docs/runbooks/jetson-on-device-learning.md` (+ soak-gate framing) and
  `docs/architecture/c4-on-device-learning.md`.

## Portfolio reframe + large-artifact handling (PR #167)

The project is framed as an **edge-AI / robotics portfolio** ("MouseDroid"), not a claim of
general intelligence. The
cognitive-stack tables (README / `docs/CHARTER.md`) are split on the **runtime-integration** axis,
not implemented-vs-stub: seven pillars are wired into the 30 Hz loop (`world_model`, `cognitive`,
`learning`, `memory`, `reward`, `safety`, `curiosity` — the last via the memory subsystem), three
are implemented + unit-tested but not yet wired (`meta`, `growth`, `scaling`), and `arm/` is
parked. Non-negotiable contracts, pinned by `tests/regression/test_portfolio_reframe_aqa.py`:

- **Brand is docs-only.** The rename is a case-sensitive whole-token sweep of the legacy brand
  token to `MouseDroid`; it never touches the `mousedroid` package, `MOUSEDROID_*` env prefixes, or
  config keys, and
  it excludes dated/append-only history (`CHANGELOG.md`, `progress.md`, `docs/superpowers/plans/*`).
  No forward-facing doc may re-assert the old general-intelligence "cohesive-agentic" framing (the
  regression AQA greps for the literal phrase, so describe it hyphenated as here).
- **Large binaries are untracked, not deleted.** `training/data/bdi_annotations.npz` (generated) and
  `docs/3D_printing_files/*.stl|*.FCStd` are gitignored + `.dockerignore`d with pointer READMEs.
  `scripts/fetch_data.sh` is **regeneration-first** (`python -m training.run_pipeline --config <cfg>
  --phases 0`; `--from-hf` is an opt-in HF-dataset mirror) and resolves the output dir from the
  config's `training.data_dir` — passing `CONFIG` as `sys.argv` (NEVER interpolated into Python
  source) and letting a load error propagate rather than masking it with a wrong-path fallback.
- **History purge is operator-run + complete.** `scripts/purge_history.sh` (runbook:
  `docs/runbooks/history-purge.md`; C4: `docs/architecture/c4-artifact-storage.md`) does a
  `git clone --mirror` (rewrites ALL refs — a blob reachable from any un-rewritten branch defeats the
  purge), `git filter-repo` globbing the CAD *binaries* so the pointer README survives, a commit-map
  re-pin of `deployments/jetson-image.json` (NOT HEAD — preserves the `config-compat` gate's schema),
  a config-compat verify in a worktree, then `git push --force --all`/`--tags`. Dry-run by default;
  `--push` is the opt-in gate. Default branch is resolved from the TARGET remote (`ORIGIN_URL`).

See `AGENTS.md` (agentic-worker behavioural contract) and `SKILLS.md`
(capability index keyed by trigger phrase) for additional context.
