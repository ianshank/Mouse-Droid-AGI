# MouseDroidAGI — Next Steps

> **Last updated**: 2026-04-14 | **Version**: 0.3.0 | **Test count**: 2505 passing | **Coverage**: 85%+ maintained

This document tracks planned enhancements, organised by priority and category.

## Recently Completed — v0.3.0 Production Readiness (2026-04-14)

All 7 phases of the Production Readiness milestone are complete:

- ✅ **Phase 1 — Deployment Hardening**: Docker device passthrough, tick timeout + emergency stop,
  systemd watchdog, pre-flight validation script, service hardening
- ✅ **Phase 2 — Memory & Curiosity Wiring**: `MemoryTier` factory, experience logging in orchestrator,
  consolidation background loop, curiosity key in obs_dict
- ✅ **Phase 3 — Voice & Rocky E2E**: startup/shutdown voice events, enriched obstacle context,
  audio-level enrichment, full mock-hardware voice pipeline validated
- ✅ **Phase 4 — Sensor Fusion Resilience**: `recovery_attempt()` on sensor manager, self-healing
  orchestrator, cascading failure tests (11 scenarios)
- ✅ **Phase 5 — LLM Gateway Integration**: rule-based parser + LLM fallback chain, degraded mode,
  prompt injection detection
- ✅ **Phase 6 — Jetson On-Device Validation**: hardware E2E test suite, 5-minute endurance tests,
  updated sensor verification script
- ✅ **Phase 7 — Production Telemetry**: new Prometheus metrics for memory, curiosity, voice, LLM, recovery

---

## Recently Completed (2026-04-14 checkpoint)

- ✅ **Green build restored**: 14 previously failing tests fixed; all 1299+ tests now pass
- ✅ **ruff + mypy cleanup**: Zero linting violations; `mypy --strict` passes with 0 errors (was 50)
- ✅ **Dual-Stream CfC/GRU RSSM merged** (PR #34): Liquid neural network hybrid world model; CfC backbone fused with classic GRU RSSM stream; full config and factory wiring
  - `DualStreamRSSM` (331 LOC), `CfCWrapper` (110 LOC), `StreamFusion` (102 LOC)
  - Factory dispatch: `cfc_hidden_dim > 0` → DualStreamRSSM, else classic RSSM
  - Dual-stream training script (712 LOC) with dual optimizers and CfC loss warmup
  - 57 new tests; 5-epoch validation training converging on RTX 5060 Ti
  - HuggingFace upload: `ianshank/mousedroid-dual-stream-rssm` (experimental)
- ✅ **FHL-LD19 LiDAR added as 5th sensor modality** (PR #31): `hardware/lidar/ld19_driver.py`, `ld19_protocol.py`, `feature_extractor.py`, `resilient_lidar.py`; factory-wired into sensor fusion pipeline
- ✅ **Wonrabai USB Sound Card integrated** (PR #32): `hardware/audio/usb_microphone.py` implementing `AudioProtocol`; replaces placeholder USB mic branch
- ✅ **Python 3.12 added to CI matrix**: All three lint, typecheck, and test stages run on 3.10, 3.11, and 3.12 in parallel
- ✅ **Jetson Docker deployment** — image rebuilt with ncps, smoke tests passing
- ✅ **SSH key deployment** to Jetson via serial console

---

## Recently Completed (Phase A checkpoint — pr-18)

- **Wired `resume_from`** through `training/run_pipeline.py` and CLI (`--resume`) to allow RSSM checkpoint continuation
- **Training-surface mypy clean**: replaced `np.ndarray` with `NDArray[Any]` across all `training/` modules; `mypy training/` passes clean
- **Extended test coverage**: new tests for `run_warmstart()` latent stats, artifact creation, UCB config propagation, and partial-pipeline combinations (resume forwarding, phase-2/3/4 with missing upstream)
- **Jetson production config activated**: `config/jetson_production.yaml` now enables cognitive core, HF auto-download, telemetry, Prometheus metrics, and Jetson safety overrides
- **Weights published**: BDI, MCTS warmstart, constitutional-RL, and RSSM weights uploaded to `ianshank/mousedroid-weights` (28 files)
- **HF subfolder download fixed**: `CognitiveConfig.huggingface_subfolder` field added; download path now resolves `bdi/belief.npz` correctly; local production smoke passes end-to-end (`health_check: ok`)

---

## Recently Completed (PR-14)

- Added Prometheus-compatible `/metrics` endpoint to telemetry server with config-driven metric namespace/toggles
- Added `scripts/check_branch_coverage.py` changed-line coverage gate (min 85%) and wired it into `scripts/ci.sh`
- Added local pre-commit hook to run branch coverage gate automatically before commits
- Added targeted telemetry tests for `/metrics` endpoint and broadcast-loop metric updates
- Hardened hardware/integration tests by replacing hardcoded serial/config values with environment-driven settings

---

## Priority 1 — Hardware Integration ✅ COMPLETE (software side)

### 1.1 HC-SR04 GPIO Integration Tests ✅
- `tests/hardware/test_hc_sr04_edge_cases.py` — 7 edge-case tests (GPIO cleanup-on-exception,
  re-init after cleanup, stale-read detection, max-range clamping, read latency, config-driven
  trigger/echo pins and speed-of-sound constant)
- Existing: `tests/hardware/test_hc_sr04_integration.py` — 9 tests (config sanity, single read,
  timeout, rapid polling, concurrent reads)

### 1.2 IMX500 Camera Integration ✅
- `tests/hardware/test_imx500_edge_cases.py` — 8 edge-case tests (double-start idempotency,
  capture-after-stop raises, stop-without-start no-op, determinism, fallback feature dim,
  `feature_dim` property, repeated start/stop no-leak, concurrent captures all valid)
- Existing: `tests/hardware/test_imx500_integration.py` — 9 tests (shape/dtype/norm, fallback,
  framerate, concurrent)

### 1.3 ESP32 Serial Driver Loopback Test ✅
- `tests/hardware/test_esp32_edge_cases.py` — 7 edge-case tests (reconnect after disconnect,
  emergency stop bypasses open circuit breaker, concurrent velocity no corruption,
  battery voltage stability, encoder fields all float, resilient driver stats, velocity
  within config limits)
- Existing: `tests/hardware/test_esp32_loopback.py` — 8 tests (loopback, e-stop latency,
  battery, circuit breaker)

### 1.4 End-to-End Sense-Plan-Act Smoke Test ✅
- `tests/hardware/test_e2e_edge_cases.py` — 7 timing regression tests (P95 tick latency ≤ 2x
  budget, jitter ≤ 50% budget, tick count monotonic, health after burst, start/stop/restart
  cycle, e-stop during burst, min 50% throughput)
- Existing: `tests/hardware/test_e2e_sense_plan_act.py` — 8 tests (burst mean latency, miss
  rate, outlier detection, tick count, run loop)

**Supporting infrastructure added:**
- `tests/hardware/conftest.py` — `autouse=True` real-hardware env override (overrides root
  `_mock_hardware_env`; sets `MOUSEDROID_MOCK_HARDWARE=false`) + session-scoped `jetson_settings`
  fixture loading from `config/jetson_production.yaml` with fallback to `config/default.yaml`
- **Bug fix**: `src/mousedroid/constants.py` — added `MILLIDEGREE_DIVISOR = 1000.0` and
  `GPU_LOAD_PERCENTAGE_DIVISOR = 10.0` (were imported by `health/monitor.py` and
  `efficiency/profiler.py` but undefined, causing `ImportError` and 3 test failures)

---

## Priority 2 — Model Training Pipeline

### 2.1 RSSM Pretraining on Simulated Data
- Generate synthetic observation sequences via `MouseDroidOrchestrator` in mock mode
- Train RSSM encoder + dynamics using the `training/` config (batch 32, lr 3e-4)
- Evaluate latent quality: reconstruction loss, surprise calibration
- Save checkpoints every 10 epochs as per `cfg.training.checkpoint_every_n`
- **Effort**: 1 week | **Owner**: ML team

### 2.2 MCTS Policy Warm-Start
- Initialise `PolicyMLP` weights from RSSM latent statistics
- Run 1000-episode simulated rollout to tune `cfg.mcts.ucb_c`
- Target: <50 ms per MCTS search at 200 simulations
- **Effort**: 3 days | **Owner**: ML team

### 2.3 BDI Weight Training
- Collect labelled intention annotations from 500 navigation episodes
- Train `BeliefEncoder`, `DesireEncoder`, `IntentionPredictor`, `AffectEstimator`
- Save as `.npz` files in `weights/bdi/`; load via `NeuralBDI(weights_dir=...)`
- **Effort**: 1 week | **Owner**: ML team

### 2.4 Constitutional RL Fine-tuning
- Define reward signal: `cfg.reward.weight_*` as per `RewardConfig`
- Run PPO with `ConstitutionalChecker` as safety constraint layer
- Validate: no constitutional violations in 1000 held-out episodes
- **Effort**: 2 weeks | **Owner**: ML team

---

## Priority 3 — LLM Gateway Deployment

### 3.1 Llama-3 GGUF Model Download
- Select a 7B-parameter quantised model (Q4_K_M) for Jetson Orin Nano
- Upload to Hugging Face Hub under project namespace
- Add download script to `scripts/download_model.sh`
- **Effort**: 1 day | **Owner**: ML team

### 3.2 LLM Gateway Integration Test
- Test 10 diverse natural language commands → `GoalVector` responses
- Validate `vx/vy/omega` are in `[-1, 1]` range
- Measure inference latency on Jetson (target: <500 ms)
- **Effort**: 2 days | **Owner**: ML team

### 3.3 Mission Parser Evaluation
- Build a benchmark of 100 NL → expected-velocity pairs
- Run `LLMGateway.translate_mission()` and score cosine similarity
- Iterate on `_SYSTEM_PROMPT` to improve accuracy
- **Effort**: 3 days | **Owner**: ML team

---

## Priority 4 — CI/CD Pipeline

### 4.1 GitHub Actions Workflow ✅ COMPLETE
- ✅ Lint (`ruff check`), type-check (`mypy`), test (`pytest --cov`) run on every push
- ✅ Coverage enforced at ≥85% via `--cov-fail-under=85`
- ✅ Smoke tests (`pytest -m smoke`) run as a fast pre-flight gate
- ✅ Matrix expanded to Python 3.10 + 3.11 + 3.12 in parallel (as of 2026-04-14)
- **Effort**: Done | **Owner**: DevOps

### 4.2 TensorRT Compilation CI
- Add Jetson-hosted CI runner for TensorRT engine compilation tests
- Test `TensorRTCompiler.compile()` with a small dummy model
- Cache compiled engines between runs
- **Effort**: 3 days | **Owner**: DevOps

### 4.3 Hardware-in-the-Loop CI
- Dedicated Jetson Orin Nano CI runner connected to mock chassis
- Run `@pytest.mark.hardware` test suite nightly
- Report latency metrics as CI artefacts
- **Effort**: 1 week | **Owner**: DevOps + hardware team

---

## Priority 5 — Observability & Monitoring

### 5.1 Prometheus Metrics Export
- ✅ Implemented in telemetry server (`/metrics`, Prometheus text format 0.0.4)
- Next: add per-deployment scrape config examples for localhost, Docker, and Jetson systemd
- **Effort**: 0.5 days | **Owner**: backend team

### 5.2 Grafana Dashboard
- Pre-built dashboard JSON for: loop latency, GPU temp, curiosity drive, memory utilisation
- Wire `loop_time_ms` from `/api/v1/sensors` WebSocket into Grafana Live
- Ship as `docs/grafana_dashboard.json`
- **Effort**: 1 day | **Owner**: backend team

### 5.3 Structured Log Aggregation
- Ship logs to Loki via Grafana Agent on Jetson
- Define log queries for: emergency stops, constitutional violations, sensor failures
- **Effort**: 2 days | **Owner**: backend team

### 5.4 Telemetry Server — Connect to Prometheus Scraping
- ✅ `/metrics` route is live and includes frame drops, ws clients, loop time, battery, safety violations, GPU temp
- Next: add `promtool check metrics` validation in CI and publish sample alert rules
- **Effort**: 1 day | **Owner**: backend + DevOps

### 5.5 Telemetry Server — Production Security
- Add optional bearer-token authentication (`cfg.telemetry.auth_token`)
- Document mTLS setup for production deployments on private networks
- Add CI test: unauthenticated request returns 401 when token is configured
- **Effort**: 1 day | **Owner**: security + backend team

---

## Priority 6 — Code Quality & Architecture

### 6.1 Mypy Strict — Resolve Pre-existing Errors
- ✅ **Resolved** — 0 mypy strict errors remaining (was 50 errors)
- Fixed untyped `backward()`, `trace()`, unused `type: ignore` comments
- Added `[[tool.mypy.overrides]]` stubs and proper type annotations
- **Effort**: 3 days | **Owner**: any engineer

### 6.2 Hypothesis Property-Based Tests
- Add `@given` tests for `_safe_softmax`, `_bayesian_normalise`, `_clamp`
- Test `ExperienceRecord.serialize/deserialize` round-trip with arbitrary data
- Test `ConstitutionalChecker.check` with arbitrary action vectors
- **Effort**: 2 days | **Owner**: any engineer

### 6.3 Memory Consolidation Integration Test
- Full pipeline: log 100 episodes → episodic memory → consolidation → semantic search
- Verify retrieved concepts have cosine similarity > 0.7 to query
- **Effort**: 1 day | **Owner**: ML team

### 6.4 Retry + Circuit Breaker Wiring
- ✅ **COMPLETE** — Implemented in `src/mousedroid/resilience/` with full factory wiring. CircuitBreaker (CLOSED→OPEN→HALF_OPEN), retry with exponential backoff + jitter, ResilientESP32Driver wrapper.
- `circuit_breaker.py` (200 LOC), `retry.py` (143 LOC), `resilient_driver.py` (187 LOC)
- Factory-wired at `factory.py` lines 62-79; emergency_stop bypasses circuit breaker
- **Effort**: 2 days | **Owner**: backend team

### 6.5 EWC + PNN Training Loop
- Write a training script `training/train_continual.py` using `EWC` + `ProgressiveNetwork`
- Test forward-transfer: performance on Task B after training on Task A
- **Effort**: 1 week | **Owner**: ML team

---

## Priority 7 — Packaging & Distribution

### 7.1 Docker Image for Development
- `Dockerfile` based on `nvcr.io/nvidia/l4t-pytorch` for Jetson
- Include all optional dependencies: hardware, jetson, llm
- **Effort**: 2 days | **Owner**: DevOps

### 7.2 Hugging Face Hub Integration
- Upload trained model weights (RSSM, BDI, policy) to `ianshank/mousedroid-weights`
- Add `scripts/download_weights.sh` to fetch from Hub on first run
- **Effort**: 1 day | **Owner**: ML team

### 7.3 PyPI Release Workflow
- Automated release on version tag push via GitHub Actions
- Publish `mousedroid==0.2.0` to PyPI
- **Effort**: 1 day | **Owner**: DevOps

---

## Priority 8 — Dual-Stream CfC/GRU Maturation

### 8.1 Extended Training on Real Data
- Train dual-stream RSSM on real sensor data from Jetson (not synthetic)
- Target: 100+ episodes from physical navigation runs
- Compare CfC vs GRU-only quality metrics on held-out episodes
- Upload trained weights to `ianshank/mousedroid-dual-stream-rssm`
- **Effort**: 1 week | **Owner**: ML team

### 8.2 CfC Time-Delta Integration & Hyperparameter Sweep
- Pass real `dt` from observation timestamps to CfC cell (currently uses unit time)
- Sweep `cfc_backbone_units` (32, 64, 128), `cfc_backbone_layers` (1, 2, 3)
- Select optimal config for Jetson inference budget
- **Effort**: 3 days | **Owner**: ML team

### 8.3 Attention-Based Stream Fusion
- Replace concat fusion with learned attention gate: `alpha * h_gru + (1-alpha) * h_cfc`
- Allow model to adaptively weight GRU (planning) vs CfC (reflexes) per timestep
- Benchmark concat vs attention vs gating on navigation task
- **Effort**: 1 week | **Owner**: ML team

### 8.4 TensorRT Export & Online CfC Adaptation
- Export DualStreamRSSM to TensorRT for Jetson inference acceleration
- Handle CfC cell's variable-time dynamics in TRT compilation
- Enable real-time CfC parameter updates from live sensor data
- Target: <10ms per observe_step on Orin Nano within 30 Hz budget
- **Effort**: 1 week | **Owner**: ML + DevOps

### 8.5 Full Activation Decision
- After 8.1-8.2 complete, review CfC contribution metrics
- Compare prediction accuracy, planning quality, and inference latency vs classic RSSM
- If CfC improves >5% over GRU-only: permanently enable in production config
- If CfC degrades: archive as experimental, keep GRU-only
- Document results in architecture decisions (docs/architecture.md)
- **Effort**: 3 days | **Owner**: Ian

---

## Backlog (Future)

| Item | Description |
|------|-------------|
| ROS 2 bridge | Publish `/cmd_vel` and subscribe to `/scan` for ecosystem compatibility |
| Sim-to-Real | Isaac Sim environment for training RSSM before hardware deployment |
| Multi-agent | Coordinate multiple Mouse Droids in a corridor |
| Voice interface | Whisper STT → LLM gateway for hands-free commanding |
| Anomaly detection | Autoencoder on observation bundles for unseen obstacle detection |
| OTA updates | Differential firmware updates for ESP32 over WiFi |
| Energy harvesting | Adaptive power mode selection based on `cfg.jetson.power_mode` |

---

## Known Limitations

| Limitation | Mitigation |
|-----------|-----------|
| Single-camera vision (no stereo depth) | HC-SR04 provides 1-D depth; MCTS imagines around uncertainty |
| No GPU-accelerated BDI inference | BDI uses numpy MLPs; acceptable at 1 Hz slow loop |
| MCTS action space is discrete (9 candidates) | Sufficient for corridor navigation; extend for open environments |
| LLM gateway adds 200–500 ms latency | Runs async; does not block 30 Hz main loop |
| No loop closure / SLAM | Odometry drift accumulates over long runs; reset via landmarks |
