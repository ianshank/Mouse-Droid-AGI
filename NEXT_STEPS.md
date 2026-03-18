# MouseDroidAGI — Next Steps

This document tracks planned enhancements, organised by priority and category.

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

> **Status**: Full pipeline executing on `feat/post-refactor-retrain` branch.
> RSSM Phase 1 training in progress (epoch ~73/200). See `training/run_pipeline.py`.

### 2.1 RSSM Pretraining on Simulated Data — ✅ COMPLETE
- ✅ Generate synthetic observation sequences (3000 episodes, `sequences.pt` 342 MB)
- ✅ Train RSSM encoder + dynamics (200 epochs, AMP, RTX 5060 Ti, ~2.5h)
- ✅ Losses fully converged: kl~1e-05, recon~0 by epoch 65
- ✅ `weights/rssm/final.pt` — 513,185 params, uploaded to HuggingFace

### 2.2 MCTS Policy Warm-Start — ✅ COMPLETE
- ✅ `weights/mcts/policy_init.npz` generated from RSSM latent statistics
- ✅ UCB tuned to c=1.41, p50=**16 ms** (target <50 ms — 3.1× under budget)
- ✅ `weights/mcts/tuned_config.json` uploaded to HuggingFace

### 2.3 BDI Weight Training — ✅ COMPLETE
- ✅ `weights/bdi/{belief,desire,intention,affect}.npz` + `belief_norm_stats.npz`
- ✅ Intention accuracy: **75.87%** (threshold 60%)
- ✅ All files uploaded to HuggingFace

### 2.4 Constitutional RL Fine-tuning — ✅ COMPLETE
- ✅ `weights/constitutional_rl/{policy,value}.npz` (PPO with Three Laws constraint)
- ✅ Shapes validated, uploaded to HuggingFace

---

## Priority 3 — LLM Gateway Deployment

> **Status**: Schema integration and Docker config complete on `feat/post-refactor-retrain`.
> Phi-3 Mini 4K q4_K_M selected (best quality/RAM fit for 8 GB Jetson unified memory).

### 3.1 LLM Model Selection & Provisioning — ✅ DONE
- ✅ Selected Phi-3 Mini 4K Instruct q4_K_M GGUF (~2.2 GB) for Jetson Orin Nano
- ✅ `GatewayConfig` added to `schema.py` with model_path, n_threads, n_gpu_layers
- ✅ Docker Compose updated with models volume + env var
- ✅ `deploy_remote.sh --with-llm` auto-downloads via huggingface_hub on Jetson
- ✅ Jetson Docker deployment completed with baked audio dependencies and container smoke tests
- ✅ `src/mousedroid/ai/` package complete: VisionAI (YOLOv8n/CLIP/MediaPipe), AudioAI (Whisper/OpenWakeWord/YAMNet), SensorFusion (MiDaS + Kalman)
- ⏳ Remaining: camera + ESP32 validation on the real hardware path

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
- Lint (`ruff check`), type-check (`mypy --strict`), test (`pytest --cov`) on every push
- Fail if coverage drops below 85% (`--cov-fail-under=85`)
- Matrix: Python 3.11, 3.12
- **Effort**: 1 day | **Owner**: DevOps

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
- Instrument key metrics: loop_hz, surprise_ema, battery_v, safety_violations_total
- Expose via HTTP `/metrics` endpoint on port 9090
- `cfg.metrics.enabled` / `cfg.metrics.export_interval_s` already configured
- **Effort**: 2 days | **Owner**: backend team

### 5.2 Grafana Dashboard
- Pre-built dashboard JSON for: loop latency, GPU temp, curiosity drive, memory utilisation
- Ship as `docs/grafana_dashboard.json`
- **Effort**: 1 day | **Owner**: backend team

### 5.3 Structured Log Aggregation
- Ship logs to Loki via Grafana Agent on Jetson
- Define log queries for: emergency stops, constitutional violations, sensor failures
- **Effort**: 2 days | **Owner**: backend team

---

## Priority 6 — Code Quality & Architecture

### 6.1 Mypy Strict — Resolve Pre-existing Errors
- 44 pre-existing strict errors in torch-dependent modules
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

### 7.2 Jetson Post-Deploy Verification
- Use `scripts/jetson_smoke_test_deploy.sh` after each Docker deploy to verify container health, CUDA, weights, LLM, ultrasonic, microphone, and recent logs
- Run `MOUSEDROID_CHECK_MIC=false` or `MOUSEDROID_CHECK_ULTRASONIC=false` only when a sensor is intentionally disconnected
- **Effort**: 0.5 day | **Owner**: robotics team

### 7.3 Hugging Face Hub Integration
- Upload trained model weights (RSSM, BDI, policy) to `ianshank/mousedroid-weights`
- Add `scripts/download_weights.sh` to fetch from Hub on first run
- **Effort**: 1 day | **Owner**: ML team

### 7.4 PyPI Release Workflow
- Automated release on version tag push via GitHub Actions
- Publish `mousedroid==0.3.0` to PyPI
- **Effort**: 1 day | **Owner**: DevOps

---

## Priority 8 — AI Perception Pipeline Hardware Validation

> **Status**: `src/mousedroid/ai/` package merged (`feat/v0.3.0-ai-pipeline`). All 1077 unit tests pass with mocks. Hardware validation on Jetson pending.

### 8.1 VisionAI Jetson Validation
- Run `JetsonYOLODetector` with a compiled `yolov8n.engine` TensorRT file on the Jetson
- Verify `ObjectDetection` bounding boxes + confidence scores against IMX500 live frames
- Confirm `half_precision=True` path achieves target FPS (>10 fps at 640×640)
- Run `CLIPEmbedder` end-to-end: capture frame → 512-dim embedding → cosine similarity
- **Effort**: 2 days | **Owner**: robotics team

### 8.2 AudioAI Jetson Validation
- Verify `OpenWakeWordDetector` triggers reliably within 1 s of saying “hey Jarvis” in corridor
- Record 10-second clips; confirm `WhisperASR` transcription WER < 20% on simple commands
- Test `YAMNetClassifier` smoke detection on pre-recorded WAV samples
- **Effort**: 1 day | **Owner**: ML team

### 8.3 SensorFusion Validation
- Compare `KalmanDepthFusion.center_distance_m` to tape-measure ground truth for 0.5 m, 1 m, 2 m targets
- Verify `FusedDepthResult.depth_map` shape matches frame resolution
- Test with ultrasonic only (MiDaS disabled) — confirm fallback to raw HC-SR04 reading
- **Effort**: 1 day | **Owner**: robotics team

### 8.4 Three Laws Integration Smoke Test
- Walk in front of the robot at 1 m distance — confirm `human_detected=True`, `human_dist_m` ≈ 1.0
- Hold up open palm facing camera — confirm `gesture_stop_commanded=True`
- Say “stop” into microphone — confirm `voice_stop_commanded=True`
- Verify orchestrator halts motion within 100 ms of any Three Laws override
- **Effort**: 1 day | **Owner**: robotics + hardware teams

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
