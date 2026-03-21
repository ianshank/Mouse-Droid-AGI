# MouseDroidAGI — Next Steps

This document tracks planned enhancements, organised by priority and category.

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

## Priority 1 — Hardware Integration (Immediate)

### 1.1 HC-SR04 GPIO Integration Tests
- Add `@pytest.mark.hardware` tests for `HcSr04` on real Jetson GPIO pins
- Verify `trigger_pin`/`echo_pin` pulse timing at configured `timeout_s`
- Test `max_range_m` cutoff and invalid echo handling
- **Effort**: 1 day | **Owner**: hardware team

### 1.2 IMX500 Camera Integration
- Verify `IMX500Camera.capture_features()` returns correct `feature_dim` (256)
- Test onboard inference pipeline with a pre-loaded `.rpk` model file
- Validate frame rate matches `cfg.camera.fps`
- **Effort**: 2 days | **Owner**: hardware team

### 1.3 ESP32 Serial Driver Loopback Test
- Add hardware test that sends `send_velocity(0.1, 0.0, 0.0)` and reads encoder feedback
- Validate `EncoderReading` fields match physical wheel movement
- Test emergency stop latency (<10 ms)
- **Effort**: 1 day | **Owner**: robotics team

### 1.4 End-to-End Sense-Plan-Act Smoke Test
- Single 5-second run on real hardware with motion log
- Verify orchestrator loop achieves 30 Hz with <5% deadline misses
- Measure total sense → ESP32 send latency
- **Effort**: 2 days | **Owner**: robotics team

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

### 4.1 GitHub Actions Workflow
- ✅ Lint (`ruff check`), type-check (`mypy`), test (`pytest --cov`) run on every push
- ✅ Coverage enforced at ≥85% via `--cov-fail-under=85`
- ✅ Smoke tests (`pytest -m smoke`) run as a fast pre-flight gate
- Next: expand matrix to Python 3.11 + 3.12 in parallel
- **Effort**: 0.5 days | **Owner**: DevOps

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
- 50 strict errors currently fail full `scripts/ci.sh` mypy gate
- Caused by untyped `backward()`, `trace()`, unused `type: ignore` comments
- Fix by adding `[[tool.mypy.overrides]]` stubs or proper type annotations
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
- `cfg.retry` and `cfg.circuit_breaker` are configured but not yet wired into drivers
- Wrap `ESP32CommProtocol` calls with exponential backoff retry
- Add circuit breaker to fail-fast after `failure_threshold` consecutive errors
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
