# MouseDroid — Implementation Plan: Next Steps

> **Date**: 2026-05-16 (Tier C merged) — last updated; originally 2026-03-19
> **Author**: System Architect Agent
> **Status**: Post-Tier-C — see "Tier C Status (2026-05-16)" below for the
> active milestone pointer. The pre-Tier-A historical plan + the 2026-04-26
> Jetson/voice/training baseline note are preserved as-is for archaeology.
> **Branch**: `claude/create-implementation-plan-aQp9z`

---

## Tier C Status (2026-05-16) — Closed-Loop Autonomy MERGED

All four Tier C tracks have landed on the integration branch:

| Track | PR | Scope | Status |
|---|---|---|---|
| **C3.1** | [#93](https://github.com/ianshank/Mouse-Droid-AGI/pull/93) | CI matrix promotion + B2 telemetry follow-through | ✅ Merged |
| **C1** | [#94](https://github.com/ianshank/Mouse-Droid-AGI/pull/94) | Closed-loop cloud retraining + Jetson OTA puller | ✅ Merged |
| **C2** | [#95](https://github.com/ianshank/Mouse-Droid-AGI/pull/95) | Mission lifecycle + geometric safety projection | ✅ Merged |
| **C4** | [#96](https://github.com/ianshank/Mouse-Droid-AGI/pull/96) | Isaac Lab env body (B3.2-B3.5) + RoverRewardConfig | ✅ Merged |

Phase status post-Tier-C:

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — Consolidate in-flight work | ✅ COMPLETE | Pre-Tier-A baseline |
| Phase 2 — Self-healing resilience | ✅ COMPLETE | Pre-Tier-A baseline |
| Phase 2.1 — BC into PPO + replay/VLM/VLA telemetry | ✅ COMPLETE | Tier A (PRs #85–#89) |
| Phase 3 — Code quality + type safety | ✅ COMPLETE | mypy strict + ruff clean |
| Phase 3b — VLA inference + ONNX | ✅ COMPLETE | Tier A + Tier B2 |
| Phase 4 — Training pipeline execution | ✅ COMPLETE (cloud side) | C1 cloud trainer wired; HF Hub upload step deferred to PR-A1.5 follow-up |
| Phase 5 — Real-physics sim-to-real foundation | ✅ COMPLETE | Tier B3 (constants + URDF→USD) + Tier C4 (env body). Operator runs `.usd` conversion on Linux to fully unlock. |
| **Phase 6 — On-device incremental learning** | 🔜 **ACTIVE** | See [NEXT_STEPS.md](NEXT_STEPS.md) §"Next Major Milestone — Phase 6" for scope. Multi-week. Plan file TBD. |

**Code-level follow-ups** (small PRs, do these first):

1. ✅ **Tier C2.1 — MissionLifecycle orchestrator wiring** — COMPLETED in
   `claude/tier-c-closeout-harden`. The lifecycle now ticks at POST_TICK and
   `process_mission` calls `start_mission()`.
2. ✅ **`training/upload_weights.py`** — COMPLETED in
   `claude/tier-c-closeout-harden`. `sync_gcs_to_hf()` + CLI `--from-gcs` close
   the cloud -> HF Hub leg of the OTA loop.
3. ✅ **Tier C2.3 — Mission Lifecycle Activation** — COMPLETED on
   `claude/tier-c2-3-mission-lifecycle-activation`. Adds the new
   `OpenAICompatibleLLMGateway` (HTTP, Ollama-default), `LLMGatewayMissionReplanner`
   adapter, `build_vlm_progress` + `build_mission_replanner` factories, and threads
   both new dependencies through `build_orchestrator → build_mission_lifecycle`.
   The lifecycle is no longer a permanent `None` in production once the three
   new flags (`mission.vlm_progress_enabled`, `mission.llm_replanner_enabled`,
   `llm.enabled`) are turned on. All defaults preserve byte-identical
   pre-Tier-C2.3 behaviour. Env-driven via `MOUSEDROID_LLM__*`.

**Operator follow-ups** (post-merge, hardware/Linux access required):

3. Run `scripts/convert_urdf_to_usd.py` on a Linux workstation + commit
   `assets/rover/mse6_4wd.usd` (unblocks the 9 Isaac Lab unit tests on Linux CI).
4. Validate the C1 / C2 / C4 Jetson E2E smokes (each PR body documents a
   per-track smoke runbook).
5. Configure branch protection to require `vla-extras (3.11)` (and promote
   `onnx-world-model-extras (3.11)` once it accumulates 7 green runs).
6. Build + push `mousedroid-cloud:tier-c1` Docker image to GCR for the
   Vertex AI custom-training job spec.

---

## Context

MouseDroid is a Star Wars MSE-6 droid replica running on NVIDIA Jetson Orin Nano with:

- **173 Python source files** across **30+ modules** (up from 121 at plan creation)
- **1299+ tests** at **85%+ branch coverage** (85% gate enforced)
- **5-stage CI pipeline**: lint → typecheck → test → security → Docker (Python 3.10 + 3.11 + 3.12)
- **Architecture**: Protocol-based DI, asyncio everywhere, Pydantic v2 config, factory pattern
- **Zero hardcoded values**: all config in YAML, constants centralized in `constants.py`
- **4 active Jetson production modalities**: camera, audio (Wonrabai USB), LiDAR (FHL-LD19), encoders

> **2026-04-26 rebaseline**: PR 54's Phase 1 domain-randomization baseline is now merged into the
> active line. The current operational roadmap is Jetson + voice rollout on camera/LiDAR/USB audio
> and ESP32. HC-SR04 ultrasonic work and the robot-arm platform are deferred from the active plan.

### Sources Analyzed

- `NEXT_STEPS.md` — 7 priority areas with 25+ action items
- `PLAN.md` — Self-healing resilience plan (6 phases, not started)
- `PLANNING.md` — L4T container deployment plan (partially done)
- `CHANGELOG.md` — recent work history
- `../analysis/COVERAGE_ANALYSIS.md` — coverage gap analysis
- `agent.md` — architectural invariants
- **11 remote branches** — 5 with in-flight commits
- `config/default.yaml`, `pyproject.toml`, `config/schema.py` — current config state

### In-Flight Branches (as of 2026-04-14 — all resolved)
| Branch | Status | Content |
|--------|--------|---------|
| `pr-16` | ✅ Merged | Offline RL (CQL/IQL) + ICM novelty decay + quality fixes |
| `plan-drone-migration` | ✅ Merged | Multi-platform MAVLink drone support |
| `feat/post-refactor-retrain` | ✅ Merged | AI perception pipeline v0.3.0 |
| `integrate-suzepi-microphone` | ✅ Closed (superseded by PR #32) | USB mic integration done via Wonrabai USB sound card |
| `plan-rssm-pretraining` | ✅ Closed (superseded by PR #34) | RSSM pretraining done via dual-stream CfC/GRU hybrid |

### Key Gaps Identified (original list — see status below)
1. ✅ **Self-healing resilience**: RESOLVED — `circuit_breaker.py`, `retry.py`, `resilient_driver.py` fully implemented and factory-wired (PR history)
2. ✅ **Sensor staleness**: RESOLVED — `SafetyConfig.sensor_stale_s` now actively checked in `safety/monitor.py`; `max_loop_time_ms` replaces hardcoded value
3. ✅ **Hardcoded value**: RESOLVED — `_MAX_LOOP_TIME_MS = 200.0` replaced with `cfg.max_loop_time_ms` from `SafetyConfig`
4. ✅ **Mypy strict**: RESOLVED — 0 errors remaining (was 50); CI passes clean
5. **Training pipeline**: Scripts exist; end-to-end execution still pending (Phase 4)
6. **Observability**: No Grafana dashboard, no Loki aggregation; Prometheus `/metrics` endpoint is live (priority 5)
7. **Packaging**: No PyPI release workflow (Phase 9)
8. ✅ **USB microphone**: RESOLVED — Wonrabai USB sound card integrated (PR #32), 5th sensor modality FHL-LD19 LiDAR added (PR #31)

---

## Phase 1: Consolidate In-Flight Work (Sprint 1)

**Status: COMPLETE** — All PRs merged (pr-16, pr-18, post-refactor-retrain). Empty branches closed.

**Goal**: Merge or close all pending branches to establish a clean baseline.

### 1.1 Review & Merge PR-16 (Offline RL + ICM Novelty Decay)
- **Branch**: `origin/pr-16`
- **Commits**: Offline RL (CQL/IQL), ICM novelty decay, quality gate fixes, HF hub coverage
- **Verify**: All tests pass, coverage >= 85%, ruff + mypy clean
- **Action**: Merge to master after review

### 1.2 Review & Merge Drone Migration
- **Branch**: `origin/claude/plan-drone-migration-PnYa5`
- **Commits**: MAVLink multi-platform support, constant centralization, 99.55% coverage
- **Verify**: Backwards compatibility — existing ESP32/Wave Rover paths unaffected
- **Verify**: New platform abstraction follows Protocol-based DI pattern
- **Action**: Merge to master after review

### 1.3 Review & Merge Post-Refactor Retrain
- **Branch**: `origin/feat/post-refactor-retrain`
- **Commits**: AI perception pipeline, config-driven training, code review fixes
- **Verify**: Integration with existing `training/` scripts
- **Action**: Merge to master after review

### 1.4 Close Empty Branches
- **Close**: `integrate-suzepi-microphone` (work scoped in Phase 5)
- **Close**: `plan-rssm-pretraining` (work scoped in Phase 4)

**Gate**: Run full suite after each merge — `pytest tests/ --cov --cov-fail-under=85`

---

## Phase 2: Self-Healing Core Resilience (Sprint 2)

**Status: COMPLETE** — `circuit_breaker.py` (200 LOC), `retry.py` (143 LOC), `resilient_driver.py` (187 LOC) fully implemented and factory-wired at `factory.py` lines 62-79. Sensor staleness live in `safety/monitor.py`.

**Goal**: Implement the 6-phase resilience plan from `PLAN.md` — the largest identified gap.

### 2.1 Circuit Breaker
- **File**: `src/mousedroid/resilience/circuit_breaker.py`
- **Design**:
  - Generic async circuit breaker: `CLOSED → OPEN → HALF_OPEN → CLOSED`
  - All thresholds from `CircuitBreakerConfig` (failure_threshold, recovery_timeout_s, half_open_max_calls)
  - Thread-safe via `asyncio.Lock`
  - Structured log events on every state transition via `structlog`
- **API**:
  ```python
  class CircuitState(enum.Enum):
      CLOSED = "closed"
      OPEN = "open"
      HALF_OPEN = "half_open"

  class CircuitBreaker:
      def __init__(self, name: str, cfg: CircuitBreakerConfig) -> None: ...
      async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T: ...

  class CircuitOpenError(Exception): ...
  ```
- **Tests** (`tests/unit/test_circuit_breaker.py`): 12 tests
  - Initial state, success stays closed, failures below/at threshold
  - Open rejects, half-open transitions, reset, concurrency, config-driven behavior

### 2.2 Retry with Exponential Backoff
- **File**: `src/mousedroid/resilience/retry.py`
- **Design**:
  - Async retry function + decorator form
  - Backoff: `min(base_delay * (exp_base ** attempt), max_delay) + jitter`
  - All params from `RetryConfig` (max_attempts, base_delay_s, max_delay_s, exponential_base)
  - Configurable retryable exception types
- **API**:
  ```python
  async def retry_async(func, *args, cfg: RetryConfig, retryable_exceptions=...) -> T: ...
  def with_retry(cfg: RetryConfig, retryable_exceptions=...) -> Callable: ...
  class RetryExhaustedError(Exception): ...
  ```
- **Tests** (`tests/unit/test_retry.py`): 12 tests
  - First try success, retry then success, exhaustion, timing, jitter, max delay cap, decorator form

### 2.3 Resilient ESP32 Driver Wrapper
- **File**: `src/mousedroid/resilience/resilient_driver.py`
- **Design**:
  - Wraps any `ESP32CommProtocol` with CB + retry (decorator pattern)
  - Implements `ESP32CommProtocol` itself — transparent drop-in replacement
  - `emergency_stop()` **bypasses** circuit breaker (safety-critical path)
  - Separate circuit breakers per operation type (command vs query)
- **Tests** (`tests/unit/test_resilient_driver.py`): 8 tests
  - Delegation, retry on failure, CB opens, emergency stop bypass, protocol conformance

### 2.4 Sensor Staleness Detection
- **Modify** `src/mousedroid/config/schema.py`:
  ```python
  class SafetyConfig(BaseModel):
      max_loop_time_ms: float = Field(200.0, gt=0)  # NEW — backwards-compatible default
      # sensor_stale_s already exists (0.5)
  ```
- **Modify** `src/mousedroid/safety/monitor.py`:
  - Track per-sensor last-valid timestamps in `dict[int, float]`
  - Check staleness against `cfg.sensor_stale_s`
  - Replace hardcoded `_MAX_LOOP_TIME_MS = 200.0` with `cfg.max_loop_time_ms`
- **Tests**: Extend `tests/unit/test_safety_monitor.py` — 5 new tests
  - Stale sensor reduces valid count, fresh sensor OK, threshold from config, backwards-compatible default

### 2.5 Factory Wiring
- **Modify** `src/mousedroid/factory.py`:
  - `build_esp32_driver()` wraps raw driver with `ResilientESP32Driver(inner, cfg.retry, cfg.circuit_breaker)`
- **Modify** `src/mousedroid/resilience/__init__.py`:
  - Export `CircuitBreaker`, `CircuitOpenError`, `CircuitState`, `RetryExhaustedError`, `retry_async`, `with_retry`, `ResilientESP32Driver`
- **Tests**: Extend `tests/unit/test_factory.py` — 2 new tests

### 2.6 Integration Tests
- **File**: `tests/integration/test_self_healing_orchestrator.py`
  - `test_orchestrator_survives_esp32_transient_failures`
  - `test_orchestrator_emergency_stops_on_sustained_failure`
  - `test_sensor_staleness_triggers_safety`
  - `test_full_tick_with_resilient_driver`

**Estimated new tests**: ~40
**Dependencies**: None (Phase 1 should merge first but not blocking)
**Backwards compat**: Default config values match current behavior; new resilience is additive

---

## Phase 3: Code Quality & Type Safety (Sprint 2-3)

**Status: NEARLY COMPLETE** — mypy strict passes with 0 errors (was 50). Some hypothesis property-based tests remain.

**Goal**: Resolve mypy strict errors and add property-based tests.

### 3.1 Mypy Strict Fixes (50 errors)
- **Files**: Various across `src/mousedroid/`
- **Common fixes**:
  - Untyped `backward()`, `trace()` — add stubs or `type: ignore[no-untyped-call]`
  - Unused `type: ignore` comments — remove
  - Missing return type annotations — add
  - Add `[[tool.mypy.overrides]]` in `pyproject.toml` for third-party libs (torch, lmdb, picamera2)
- **Target**: Zero mypy errors in `scripts/ci.sh`

### 3.2 Hypothesis Property-Based Tests
- **File**: `tests/property/test_math_properties.py`
  - `@given` tests for `_safe_softmax`, `_bayesian_normalise`, `_clamp`
  - Invariant: output sums to 1.0, values in [0,1], clamped within bounds
- **File**: `tests/property/test_experience_properties.py`
  - `ExperienceRecord.serialize() → deserialize()` round-trip with arbitrary data
  - Invariant: deserialized == original
- **File**: `tests/property/test_constitutional_properties.py`
  - `ConstitutionalChecker.check()` with arbitrary action vectors
  - Invariant: output always within safety bounds
- **File**: `tests/property/test_action_properties.py`
  - `ActionNormalizer` bounds invariants
  - Invariant: `|vx| <= max_velocity_mps`, `|omega| <= max_omega_rads`

### 3.3 Memory Consolidation Integration Test
- **File**: `tests/integration/test_memory_consolidation.py`
  - Pipeline: log 100 episodes → episodic memory → consolidation → semantic search
  - Verify: cosine similarity > 0.7 for retrieved concepts

### 3.4 EWC + Progressive Networks Training Loop
- **File**: `training/train_continual.py`
  - Config-driven (batch size, lr, EWC lambda from `LearningConfig`)
  - Test forward-transfer: Task B performance after Task A
- **Tests**: `tests/unit/test_train_continual.py`

**Estimated new tests**: ~25
**Dependencies**: Phase 1 (clean baseline)

---

## Phase 4: Training Pipeline Execution (Sprint 3-4)

**Goal**: Execute the complete model training pipeline end-to-end.

### 4.1 RSSM Pretraining on Synthetic Data
- **Script**: `training/train_rssm.py`
- **Data**: Generate synthetic sequences via mock orchestrator (`training/data_generator.py`)
- **Config**: All hyperparams from `config/local_training.yaml` (batch=32, lr=3e-4)
- **Validation**: Reconstruction loss converges, surprise calibration
- **Checkpoints**: Every 10 epochs per `cfg.training.checkpoint_every_n`
- **Tests**: `tests/unit/test_rssm_training.py` — 1-epoch smoke test with tiny data

### 4.2 MCTS Policy Warm-Start
- **Script**: `training/warmstart_policy.py`
- Initialize `PolicyMLP` from RSSM latent statistics
- Tune `cfg.mcts.ucb_c` via 1000-episode simulated rollout
- **Target**: <50ms per MCTS search at 200 simulations
- **Tests**: `tests/performance/test_mcts_latency.py`

### 4.3 BDI Weight Training
- **Script**: `training/train_bdi.py`
- Train: BeliefEncoder, DesireEncoder, IntentionPredictor, AffectEstimator
- Save `.npz` to `weights/bdi/`; load via `NeuralBDI(weights_dir=...)`
- **Tests**: `tests/unit/test_bdi_training.py` — shape validation, convergence check

### 4.4 Constitutional RL Fine-Tuning
- **Script**: `training/train_constitutional_rl.py`
- PPO with `ConstitutionalChecker` as safety constraint layer
- **Validation**: Zero constitutional violations in 1000 held-out episodes
- **Tests**: `tests/unit/test_constitutional_rl_training.py`

### 4.5 Upload Weights to HuggingFace
- **Script**: `training/upload_weights.py` → `ianshank/mousedroid-weights`
- Verify round-trip via `weights_manager.py` auto-download

**Estimated new tests**: ~15
**Dependencies**: Phase 2 (resilient driver), Phase 3 (clean types)

---

## Phase 5: USB Microphone Integration (Sprint 3)

**Status: COMPLETE** — Wonrabai USB sound card merged in PR #32. Audio modality fully integrated.

**Goal**: Complete SuziePi USB microphone integration (empty `integrate-suzepi-microphone` branch).

### 5.1 Config
- **Verify/modify** `src/mousedroid/config/schema.py`:
  - `MicrophoneConfig`: `sample_rate`, `channels`, `chunk_size`, `device_name`, `format`
- **Verify/modify** `config/default.yaml`: `microphone:` section with defaults

### 5.2 USB Microphone Driver
- **Verify/modify** `src/mousedroid/hardware/audio/usb_microphone.py`:
  - Implements `AudioProtocol`
  - Config-driven: sample rate, channels, device name from `MicrophoneConfig`
  - Async capture via `asyncio.to_thread()`

### 5.3 Factory Wiring
- **Modify** `src/mousedroid/factory.py`: `build_microphone()` uses `MicrophoneConfig`
- **Modify** `src/mousedroid/sensing/manager.py`: audio modality ring buffer

### 5.4 Tests
- `tests/unit/test_usb_microphone.py` — mock PyAudio, verify config-driven behavior
- `tests/hardware/test_usb_microphone_hw.py` — `@pytest.mark.hardware` real device

**Estimated new tests**: ~8
**Dependencies**: None (can run parallel to Phases 2-4)
**Backwards compat**: Audio already optional — no changes to non-audio paths

---

## Phase 6: CI/CD Expansion (Sprint 4)

**Goal**: Strengthen the CI pipeline for broader coverage and hardware testing.

### 6.1 Python 3.12 Matrix ✅ COMPLETE
- `.github/workflows/ci.yml` runs matrix `["3.10", "3.11", "3.12"]` for lint, typecheck, and test stages
- Deprecation warnings (`datetime.utcnow()` → `datetime.now(UTC)`) resolved

### 6.2 Prometheus Metrics Validation
- **Modify** `.github/workflows/ci.yml`: add `promtool check metrics` step
- **File**: `config/prometheus/alerts.yml` — sample alert rules
- **Tests**: `tests/smoke/test_prometheus_format.py`

### 6.3 TensorRT Compilation CI (Jetson Runner)
- **File**: `.github/workflows/jetson-ci.yml`
- Test `TensorRTCompiler.compile()` with dummy model
- Cache compiled engines between runs
- **Requires**: Jetson self-hosted runner infrastructure

### 6.4 Hardware-in-the-Loop Nightly
- **File**: `.github/workflows/hardware-nightly.yml`
- Run `@pytest.mark.hardware` tests on Jetson runner
- Report latency metrics as CI artifacts
- **Requires**: Jetson runner + mock chassis

**Estimated new tests**: ~5
**Dependencies**: Phase 1 (clean baseline)

---

## Phase 7: Observability & Monitoring (Sprint 4-5)

**Goal**: Production-ready monitoring stack.

### 7.1 Grafana Dashboard
- **File**: `docs/grafana_dashboard.json`
- Panels: loop latency, GPU temp, curiosity drive, memory utilization, battery, safety violations
- Wire `loop_time_ms` from WebSocket → Grafana Live

### 7.2 Telemetry Security
- **Modify** `src/mousedroid/telemetry/server.py`:
  - Bearer-token auth via `cfg.telemetry.auth_token`
  - Document mTLS for production
- **Tests**: `tests/unit/test_telemetry_auth.py` — 401 on unauthenticated when token configured

### 7.3 Prometheus Scrape Configs
- `config/prometheus/scrape_localhost.yml`
- `config/prometheus/scrape_docker.yml`
- `config/prometheus/scrape_jetson_systemd.yml`

### 7.4 Loki Log Aggregation
- `config/loki/promtail.yml` — Grafana Agent config for Jetson
- Log queries: emergency stops, constitutional violations, sensor failures

**Estimated new tests**: ~5
**Dependencies**: Phase 1, Phase 6.2

---

## Phase 8: LLM Gateway Deployment (Sprint 5)

**Goal**: Deploy natural language command interface.

### 8.1 Llama-3 GGUF Model Download
- **File**: `scripts/download_model.sh`
- Select Q4_K_M quantized 7B model for Jetson Orin Nano (8GB RAM)
- Config-driven model path via `LLMConfig`

### 8.2 LLM Gateway Integration Tests
- **File**: `tests/integration/test_llm_gateway.py`
  - 10 diverse NL commands → `GoalVector` responses
  - Validate `vx/vy/omega` in `[-1, 1]`
  - Latency target: <500ms on Jetson

### 8.3 Mission Parser Evaluation
- **File**: `scripts/eval_mission_parser.py`
  - 100 NL → expected-velocity benchmark
  - Cosine similarity scoring
  - Iterate on `_SYSTEM_PROMPT`

**Estimated new tests**: ~10
**Dependencies**: Phase 4 (trained models)

---

## Phase 9: Packaging & Distribution (Sprint 6)

**Goal**: Production-ready packaging and release.

### 9.1 PyPI Release Workflow
- **File**: `.github/workflows/release.yml`
  - Trigger on version tag push (`v*`)
  - Build wheel + sdist
  - Publish `mousedroid==0.2.0` to PyPI
- Bump version in `pyproject.toml`

### 9.2 Docker Dev Image
- Verify `Dockerfile.jetson` includes all optional deps
- **File**: `Dockerfile.dev` — CPU-only dev image (non-Jetson)
- Docker build + smoke test in CI

### 9.3 HuggingFace Hub Weights
- Automate weight upload in CI post-training
- **File**: `scripts/download_weights.sh` — first-run setup
- Verify `weights_manager.py` auto-download round-trip

**Estimated new tests**: ~3
**Dependencies**: Phase 4 (trained weights), Phase 6 (CI infrastructure)

---

## Phase 10: Dual-Stream CfC/GRU RSSM Maturation (Sprint 3-4)

**Status: MERGED** — PR #34 merged the liquid neural network hybrid world model. Core implementation landed; training, benchmarking, and Jetson deployment remain.

**Goal**: Mature the dual-stream CfC/GRU RSSM from proof-of-concept to production-ready world model.

### 10.1 Extended Training Run
- Run 50+ epoch training on RTX 5060 Ti with dual-stream CfC/GRU RSSM
- Validate convergence; compare loss curves to classic RSSM baseline
- Upload trained weights to `ianshank/mousedroid-dual-stream-rssm` on HuggingFace Hub
- **Estimated effort**: 2 days

### 10.2 CfC Hyperparameter Sweep
- Sweep `cfc_backbone_units` (32, 64, 128) and `cfc_backbone_layers` (1, 2, 3)
- Compare training speed and final loss across configurations
- Select optimal config for Jetson inference budget (8 GB RAM, 30 Hz target)
- **Estimated effort**: 3 days

### 10.3 Fusion Strategy Comparison
- Implement attention-based and gating-based alternatives to concat fusion in `stream_fusion.py`
- Benchmark concat vs attention vs gating on navigation task rollouts
- **Estimated effort**: 1 week

### 10.4 Online CfC Adaptation on Jetson
- Enable real-time CfC parameter updates from live sensor data
- Validate inference latency stays within 30 Hz budget with adaptation enabled
- **Estimated effort**: 1 week

### 10.5 Dual-Stream vs Classic RSSM Benchmarks
- Compare prediction accuracy, planning quality (MCTS search depth), and inference latency
- Document findings in `docs/architecture.md` as an ADR
- **Estimated effort**: 3 days

**Estimated new tests**: ~10
**Dependencies**: Phase 4 (training pipeline), Phase 6 (CI infrastructure for benchmarks)

---

## Verification Plan

### Per-Phase Gate (must pass before advancing)

| Check | Command | Threshold |
|-------|---------|-----------|
| Lint | `ruff check src/ tests/` | Zero violations |
| Types | `mypy --strict src/mousedroid/` | Zero errors (Phase 3+) |
| Coverage | `pytest tests/ --cov --cov-fail-under=85` | >= 85% branch |
| Tests | `pytest tests/ -v --tb=short` | All pass |
| Hardcoded | Grep for magic numbers outside `constants.py`/`schema.py` | Zero matches |

### End-to-End Validation (after all phases)

1. **Mock-mode run**: `python -m mousedroid.main --mock-hardware --config config/default.yaml`
2. **Docker GPU**: `docker-compose -f docker-compose.jetson.yml up`
3. **Training pipeline**: `python training/run_pipeline.py --config config/local_training.yaml`
4. **Telemetry**: Hit `/api/v1/status`, `/metrics`, `/ws` endpoints
5. **Coverage HTML**: `pytest --cov --cov-report=html` — verify 85%+ per module

### Architectural Invariants (from agent.md)

- All interfaces are `@runtime_checkable Protocol`
- All thresholds/dims/pins come from Pydantic config or `constants.py`
- Factory functions are the only place importing concrete types
- `structlog` for all logging — never `print()`
- `asyncio` everywhere — no threading
- `torch.no_grad()` for all inference paths
- `deque(maxlen=N)` for all sensor ring buffers

---

## Summary

| Phase | Sprint | Focus | Status | New Tests | Key Deliverables |
|-------|--------|-------|--------|-----------|------------------|
| **1** | 1 | Merge in-flight PRs | ✅ **COMPLETE** | Existing | Clean baseline on master — all 5 branches resolved |
| **2** | 2 | Self-healing resilience | ✅ **COMPLETE** | ~40 | `circuit_breaker.py`, `retry.py`, `resilient_driver.py`, sensor staleness |
| **3** | 2-3 | Code quality + types | ✅ **NEARLY COMPLETE** | ~25 | Mypy strict at 0 errors (was 50); Hypothesis tests remain |
| **4** | 3-4 | Training pipeline | Not started | ~15 | RSSM, MCTS, BDI, Constitutional RL trained models |
| **5** | 3 | USB microphone | ✅ **COMPLETE** | ~8 | Wonrabai USB sound card integrated (PR #32) |
| **6** | 4 | CI/CD expansion | **6.1 ✅ COMPLETE**, rest pending | ~5 | Python 3.10/3.11/3.12 matrix in CI; TensorRT CI + hardware nightly pending |
| **7** | 4-5 | Observability | Partial | ~5 | `/metrics` endpoint live; Grafana, Loki, alert rules pending |
| **8** | 5 | LLM gateway | Not started | ~10 | Llama-3 deployment, NL command interface |
| **9** | 6 | Packaging | Not started | ~3 | PyPI release, Docker dev, HF Hub weights |
| **10** | 3-4 | Dual-Stream CfC/GRU RSSM | **MERGED** (PR #34), maturation in progress | ~10 | Extended training, hyperparameter sweep, Jetson benchmarks |

**Total estimated new tests**: ~121 (was 111 before Phase 10)
**Current test count**: 1299+ passing (was 752+ at plan creation)
**Coverage target**: Maintain >= 85% at every phase gate
**Total sprints**: 6 (parallel tracks where noted)

### Dependency Graph

```
Phase 1 (merge PRs) ✅ COMPLETE
  ├─→ Phase 2 (resilience) ✅ COMPLETE
  │     └─→ Phase 4 (training)
  │           ├─→ Phase 8 (LLM gateway)
  │           │     └─→ Phase 9 (packaging)
  │           └─→ Phase 10 (CfC maturation) ← PR #34 merged
  ├─→ Phase 3 (code quality) ✅ NEARLY COMPLETE
  │     └─→ Phase 4 (training)
  ├─→ Phase 5 (microphone) ✅ COMPLETE ← independent
  ├─→ Phase 6 (CI/CD) ← 6.1 done, rest pending
  │     └─→ Phase 7 (observability) ← /metrics live, rest pending
  │           └─→ Phase 9 (packaging)
  ├─→ Phase 7 (observability)
  └─→ Phase 10 (CfC/GRU RSSM maturation) ← parallel track
```

### Backlog (Future — post Phase 9)

| Item | Description |
|------|-------------|
| ROS 2 bridge | `/cmd_vel` + `/scan` for ecosystem compatibility |
| Sim-to-Real | Isaac Sim for RSSM training before hardware |
| Multi-agent | Coordinate multiple Mouse Droids |
| Voice interface | Whisper STT → LLM gateway |
| Anomaly detection | Autoencoder on observation bundles |
| OTA updates | Differential ESP32 firmware updates |
| Energy harvesting | Adaptive power modes |
