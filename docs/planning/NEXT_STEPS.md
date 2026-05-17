# MouseDroidAGI — Next Steps

> **Last updated**: 2026-05-17 | **Version**: 0.4.1-dev (smoke-stability pass + live-Jetson verification) | **Pre-PR validation**: Ruff clean, mypy strict clean, 4792 tests pass on host + dashboard pipeline live-verified on rover

## Smoke-stability sprint findings — operator-actionable follow-ups

The intermediate smoke-test stability sprint (`claude/smoke-test-stability-pass`,
2026-05-17) live-verified the rover stack at `192.168.55.1` and surfaced 14 findings
(F-001…F-014 in `SMOKE_REPORT.md`). Most are documented expected states; these are
the ones that still need action.

### F-006 (HIGH, operator-actionable) — LLM inference too slow for real-time control

- **Symptom:** `translate_mission("turn left slowly")` on the live Jetson took **260 s**
  (Phi-3-mini-q4 at 0.52 tok/s, eval). Orchestrator latency budget is 500 ms.
- **Root cause:** `jetson_production.yaml` ships `llm.n_gpu_layers: 0` — CUDA KV cache
  is allocated but matmul stays on CPU.
- **Fix (no code):** set `llm.n_gpu_layers: -1` (offload all) in
  `config/jetson_production.yaml` and re-deploy. Validate via `tools/lidar_telemetry_probe`-style
  one-shot probe that calls `LLMGateway.translate_mission` and asserts elapsed ≤ 500 ms.
- **Code follow-up:** add a runtime guard in `LLMGateway` that emits a
  `llm_latency_budget_exceeded` structured log when a single call exceeds
  `cfg.llm.latency_target_ms`. Operator dashboard then alerts before this regresses
  silently again.

### F-013 + F-014 (HIGH, ops) — Dashboard plumbing fragility

The dashboard data path is correct end-to-end (verified live) but two deployment fragilities
must be closed in a code follow-up:

- **F-013:** there are **two copies** of `jetson_production.yaml` on each Jetson —
  `config/jetson_production.yaml` (in-repo, the source of truth) and
  `/etc/mousedroid/jetson_production.yaml` (operator-deployed, drifts). No sync job
  closes the gap. The live verification needed an `scp config/jetson_production.yaml
  jetson:/etc/mousedroid/jetson_production.yaml` step before the dashboard would come up.
- **F-014:** `docker-compose.jetson.yml:31` substitutes
  `MOUSEDROID_MOCK_HARDWARE=${VAR:-true}` from the **host shell env at compose-up time**,
  ignoring the `MOUSEDROID_MOCK_HARDWARE=false` setting inside
  `/etc/mousedroid/docker.env`. Production thus defaults to mock unless the operator
  explicitly exports the var.

**Proposed code follow-up sprint** (a single PR after this one ships):

1. Add `scripts/deploy_config_jetson.sh` that runs `scp config/*.yaml
   jetson:/etc/mousedroid/` + verifies checksums + restarts the orchestrator. Document
   in `JETSON_SMOKE_RUNBOOK.md` as the canonical config-refresh path.
2. Flip the compose default to `:-false` so production deployments default to real
   hardware. Tests pass `MOUSEDROID_MOCK_HARDWARE=true` explicitly.
3. Add `MOUSEDROID_TELEMETRY_TOKEN=${MOUSEDROID_TELEMETRY_TOKEN}` to the compose
   `environment:` block so the token defined in `docker.env` actually reaches the
   container at startup.
4. Add a `_log.info("mock_hardware_resolved", value=cfg.mock_hardware)` line at
   orchestrator boot so the resolved boolean is always visible in container logs.

### F-009 (INFO, observability) — TensorRT silent mock fallback

`build_tensorrt_compiler` returns `MockTensorRTCompiler` when the `tensorrt` Python
wheel fails to import, with the choice only logged at DEBUG. On a production Jetson
this means the operator can't tell from logs whether the real compiler or the mock is
selected. Promote `tensorrt_compiler_built` to INFO with a `backend` field whose
value is `real | mock`, and add a one-liner to the operator runbook to pin the real
class.

### F-010 (MED, Tier C3 sprint) — VLM progress is constant mock

The Tier C2.3 mission-replanner loop runs against `MockVLMProgress` which returns
`cfg.mission.vlm_mock_progress_value` regardless of input. The plumbing works
(closed-loop test passes) but the operator should know "replanner decisions are
evaluating the wiring, not the real VLM signal." Document in `CHANGELOG.md` until
Tier C3 lands a real BLIP-2 / SmolVLM backend behind the same
`VLMProgressBackendProtocol`.

---

## Next Major Milestone — Phase 6: On-Device Incremental Learning

With Tier C merged (cloud retraining + OTA + closed-loop missions + safety
projection + Isaac Lab sim foundation), the rover now learns continuously
from cloud retraining. The remaining autonomy gap is **on-device incremental
learning** — i.e. the rover updating its own weights between cloud
retraining cycles using fresh experience without going through GCP.

Scope (multi-week, deferred from Tier C):

- **Continual learning hooks** — extend `learning/ewc.py` + `learning/progressive_net.py`
  with online update paths gated by the same SHA-256 integrity contract C1 wired
  for cloud OTA (per ADR-010). On-device updates land in a separate slot from the
  cloud-pulled weights so the orchestrator can A/B between them.
- **Online experience replay** — the existing `harness/replay_buffer.py` + the
  Tier A replay/VLM/VLA telemetry families gate the trigger condition for an
  on-device update step.
- **Safety gate** — a regression bound on the cloud-validated policy: if the
  on-device-updated policy drops below it on a held-out replay sample, the
  orchestrator reverts to the cloud weights and emits a
  `mousedroid_on_device_learning_reverted_total{reason}` counter (new family).
- **Operator follow-up** — `training/upload_weights.py::sync_gcs_to_hf` shipped
  in the Tier C Closeout branch (`claude/tier-c-closeout-harden`); the cloud
  loop is now fully closed (GCS -> HF Hub via `--from-gcs`).

Estimated scope: 3–4 sprints. Plan file: TBD (will be authored under
`.claude/plans/` once the upload module + operator validation of Tier C
land on the rover).

---

## Recently Completed — 2026-05-16 Tier C Closeout + Harden

Closed the three documented Tier C follow-ups and added two test-suite improvements:

- ✅ **C1.2 dual weight-update pollers** — `WeightUpdatePollConfig.world_model_enabled`
  config field (default `False`, backwards-compatible) gates a second
  `HuggingFaceWeightUpdatePoller` slot for the world-model engine. New
  `build_weight_update_pollers()` factory returns a `Mapping[str, ...]` keyed
  by `engine_type`. Orchestrator's `_apply_pending_weight_update` now iterates
  the mapping; the legacy `weight_update_poller=` kwarg is retained for one
  minor-version deprecation cycle and folded into the mapping at construction.
- ✅ **C1.1 cloud-trainer GCS->HF Hub leg** — `training/upload_weights.py::sync_gcs_to_hf`
  downloads blobs under a GCS prefix and pushes to HF Hub via the existing
  `upload_weights()`. CLI `--from-gcs --gcs-bucket --gcs-prefix` flags resolve
  defaults from `Settings.gcp.training.training_bucket` and
  `Settings.cloud.weight_update.policy_repo_id`.
- ✅ **C2.1 MissionLifecycle wiring** — `mission_lifecycle` constructor kwarg +
  `_maybe_tick_mission_lifecycle()` helper fires at the POST_TICK seam (after
  `task_tracker.evaluate_active` so terminal-state forwarding is correctly
  ordered). `process_mission` calls `start_mission()` so the lifecycle actually
  transitions PENDING -> RUNNING in production (not a permanent no-op).
- ✅ **Hygiene: numpy soft pin** — `numpy>=1.24,!=2.0.0,!=2.0.1` locks out the
  initial NumPy 2.x releases that broke transitive deps without blocking future
  2.x once the ecosystem catches up. Tree audit confirmed no in-repo usage of
  deprecated symbols (`np.float_`, `np.NAN`, `np.in1d`, etc.).
- ✅ **Integration test** — `tests/integration/test_tier_c_closeout_integration.py`
  exercises lifecycle + dual poller + projector together on a single tick AND
  round-trips through `build_orchestrator()` to prove the factory DI graph.
- ✅ **Property test** — `tests/property/test_mission_lifecycle_property.py`
  uses Hypothesis (100 examples) to verify the lifecycle's state machine never
  makes illegal transitions across arbitrary VLM-score sequences.

**Deferred from this PR:** `onnx-world-model-extras (3.11)` CI promotion to
required — the gate (≥7 consecutive green per-job runs on the integration
branch) was only met 6/7 times. Re-attempt once accumulated.

---

## Recently Completed — 2026-05-16 Tier C: Closed-Loop Autonomy + Cloud Retraining + Isaac Lab

Four parallel tracks merged after Tier B (PRs #90–#92):

- ✅ **C3.1 (PR #93)** — CI matrix promotion + B2 telemetry follow-through.
  Promoted `vla-extras (3.11)` from advisory to required after 6+ green runs.
  Wired the deferred `MetricsRegistry.observe_world_model_observe_step_seconds`
  helper (was using a defensive `getattr(..., None)` fallback in
  `DualStreamRSSMOnnx`). Grafana panel + Prometheus alert + Tier C dashboard
  E2E smoke scaffold (`tests/smoke/test_prometheus_format_tier_c.py`).
- ✅ **C1 (PR #94)** — Closed-loop cloud retraining + Jetson-side OTA puller.
  New `HuggingFaceWeightUpdatePoller` polls HF Hub on a config-driven
  cadence, verifies SHA-256 integrity against a published manifest, and
  surfaces a `PendingWeightUpdate` for the orchestrator to swap atomically
  after `_select_action` returns. `reset_state_on_swap=True` default avoids
  one-tick cross-model contamination via `torch.zeros_like(...)` (device +
  dtype preserved). Four new Prometheus families (downloads / SHA256
  mismatches / swaps / download_seconds). `Dockerfile.cloud` (x86 GPU,
  separate from `Dockerfile.jetson`'s ARM L4T base) + Vertex AI job spec.
  HF Hub upload step deferred to a follow-up (no `training/upload_weights.py`
  yet — operators hand off via `huggingface-cli upload`).
- ✅ **C2 (PR #95)** — Closed-loop mission lifecycle + geometric safety
  projection. `MissionLifecycle` state machine (PENDING → RUNNING →
  SUCCEEDED | FAILED | REPLANNING → RUNNING) polls `VLMProgressHead` per
  tick and triggers LLM-driven adaptive replan on stall (with operator-
  tunable replan limit). `GeometricSafetyProjector` clamps the policy's
  proposed action onto the safety-feasible set at a single seam in
  `tick()` covering ALL four `_select_action` return branches
  (cognitive / VLA / VLA-strict-timeout / nav_agent) so no learned
  safety violation can leak through. Four new Prometheus families
  (safety clamps / mission state transitions / replans / mission
  active-duration). Both seams default-disabled (byte-identical pre-C2
  behavior).
- ✅ **C4 (PR #96)** — Isaac Lab env body (B3.2-B3.5) replacing the three
  `TODO(Phase B)` markers in `sim/isaaclab/rover_env.py`. Real Isaac Lab
  `>=0.20,<0.30` API: `Articulation(ArticulationCfg(...))` live handle,
  `SimulationCfg.dt`, `ImplicitActuatorCfg` wrappers, `ContactSensor`
  wiring with `contact.data.net_forces_w` collision detection. 9 unit
  tests under `pytest.importorskip("isaaclab")` skip cleanly on CI hosts
  without Isaac Sim. Wheel fan-out matches `MockRoverEnv` exactly
  (`[FL, FR, RL, RR] = [left, right, left, right]` alternating).
  `RoverRewardConfig` Pydantic block with `Field(..., ge=0)` constraints
  preventing negative collision weights inverting the safety sign.

**Operator follow-ups (post-merge, hardware-only)**:

- Configure branch protection on the integration branch to require
  `vla-extras (3.11)` (and promote `onnx-world-model-extras (3.11)` once
  it accumulates 7 green runs since PR #92).
- Run `scripts/convert_urdf_to_usd.py` on a Linux workstation against
  `assets/rover/mse6_4wd.urdf` + commit the resulting `.usd` so the C4
  tests run for real on Linux (they skip cleanly on Windows / CI).
- Validate the C1 / C2 / C4 Jetson E2E smokes documented in each track's
  PR body (Tier C plan §"Jetson E2E Smoke" per track).

See [CHANGELOG.md](../../CHANGELOG.md) `## [v0.4.0] — 2026-05-16 — Tier C` for
the full surface diff and rollback paths per track.

---

## Recently Completed — 2026-05-15 Phase 2.1 PR-A1 — Close the Policy Learning Loop

- ✅ **Dedicated `bc_optimizer`** wired into `OfflineRLTrainer` via new optional `OfflineRLConfig.bc_lr` and `OfflineRLConfig.bc_batch_size` fields. When `bc_lr is None` (default), the BC optimizer is aliased to `policy_optimizer` (byte-identical legacy behavior). When `bc_lr` is set, a dedicated Adam over policy parameters is built — BC steps no longer corrupt the actor PPO optimizer state.
- ✅ **Sim/real `RealSimMixer` integration** behind the new `OfflineRLConfig.use_replay_mixer` toggle. When enabled with a distinct `cfg.training.replay.source_path`, `train_offline_rl` deterministically interleaves sim + real LMDB batches via the existing `MixerConfig.from_settings` pipeline. Graceful fallback when the real path is missing or identical.
- ✅ **Activation log assertion** patched into `tests/integration/test_phase21_bc_into_offline_rl.py::test_bc_loss_recorded_in_stats` (closes prior gap where `caplog` was passed but never asserted).
- ✅ **Performance budget regression** at `tests/performance/test_offline_rl_bc_overhead.py` — operator-tunable via `MOUSEDROID_BC_OVERHEAD_BUDGET` (default 2.5×).
- ✅ **Backwards-compatible checkpoints** — legacy `.pt` files (no `bc_optimizer` key) load cleanly into trainers with or without a dedicated optimizer.
- See [CHANGELOG.md](../../CHANGELOG.md) `## [Unreleased]` for full surface diff and rollback paths.



This document tracks planned enhancements, organised by priority and category.

> **2026-04-26 rebaseline**: the active production baseline is Jetson + camera + LiDAR + USB
> audio + ESP32. Overlay-sync automation and the early voice-config rollout are complete. HC-SR04
> ultrasonic items and robot-arm work in this document should be treated as deferred unless they
> are explicitly reactivated.

## Recently Completed — 2026-04-24 CI Determinism + Config Compatibility Hardening

- ✅ **Deterministic CI interpreter resolution**: `scripts/ci.sh` now resolves Python in a stable order
  (`MOUSEDROID_PYTHON` → workspace virtualenv → PATH) and exports `PYTHONNOUSERSITE=1` to avoid
  user-site package drift on Windows/Git Bash
- ✅ **Pre-test class-identity guard**: `scripts/check_settings_identity.py` verifies canonical
  `Settings` imports before running pytest, preventing intermittent `is_instance_of` failures
- ✅ **Changed-line coverage stability**: `tests/unit/test_config_migration.py` now exercises
  migration helper branches so `check_branch_coverage.py` remains green when schema alias logic evolves
- ✅ **Documentation structure professionalization**: planning/analysis documents moved to
  `docs/planning/` and `docs/analysis/` with updated references

## Recently Completed — 2026-04-23 Jetson Runtime Validation Alignment

- ✅ **Shared runtime validation layer**: `src/mousedroid/validation/runtime.py` now resolves the
  same config overlays used by the application and exposes reusable camera, microphone, speaker,
  and LiDAR checks for smoke and host-driven validation flows
- ✅ **Camera fallback hardening**: `JetsonCSICamera` now degrades from the primary Jetson path to
  GStreamer and then config-driven V4L2 `camera.device_path`, keeping ribbon-camera deployments usable
- ✅ **LiDAR completeness alignment**: scan acquisition timeout and minimum coverage are now
  config-driven (`scan_acquisition_timeout_s`, `min_scan_coverage_deg`), and validation uses the
  same coverage semantics as the driver
- ✅ **CI cleanup**: strict mypy is clean again, performance and E2E suites no longer bypass the
  mock-hardware test environment, and the pre-PR test run holds the 85% coverage gate

## Immediate Follow-up

- Install `promtool` on the Windows validation host so the Prometheus rule stage becomes enforced
  instead of skipped
- Re-run `tests/performance/test_jetson_endurance.py::test_endurance_30hz_loop` on non-mock Jetson
  hardware to validate the real 30 Hz deadline rather than the mock-runtime path

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

## Priority 9 — GCP Digital Twin

### 9.0 Phase 1: Telemetry Bridge + Cloud Storage ✅ COMPLETE (2026-04-15)
- ✅ `src/mousedroid/cloud/` module — Pub/Sub sink, GCS experience exporter, Cloud Logging,
  Cloud Monitoring, Firestore sync, credential resolver
- ✅ 8 GCP Pydantic config models (`Settings.gcp: GCPConfig | None = None`)
- ✅ 4 `build_cloud_*()` factory functions with graceful ImportError fallback
- ✅ Orchestrator integration (cloud sink + experience exporter in start/stop/tick)
- ✅ `config/gcp_digital_twin.yaml` overlay, Docker GCP SDK stage, credentials volume mount
- ✅ 88 cloud unit tests at 88.77% coverage

### 9.1 Phase 2: Cloud Training Pipeline
- Implement `CloudOfflineRLDataset` — GCS shards → PyTorch tensors (same interface as `OfflineRLDataset`)
- Build Vertex AI Pipeline (KFP v2) mirroring the local 5-phase `run_pipeline.py`
- Add `--data-source gcs://` flag to `training/train_rssm.py`
- Add `--cloud` flag to `training/run_pipeline.py` for Vertex AI execution
- Cloud Scheduler for nightly retraining (cron: `0 2 * * *` UTC)
- Vertex AI Model Monitoring for RSSM prediction drift
- EWC Fisher matrix update step in cloud pipeline
- **Effort**: 4 weeks | **Owner**: ML + cloud team

### 9.2 Phase 3: Parallel Simulation + Safety Validation
- GKE Autopilot cluster running mock-hardware containers
- Scenario generation pipeline (battery/distance/thermal sweeps, grammar-based fuzzing)
- Safety validation campaigns (500+ parallel scenarios)
- Red-team LLM prompt injection testing
- **Effort**: 6 weeks | **Owner**: ML + DevOps + security team

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
