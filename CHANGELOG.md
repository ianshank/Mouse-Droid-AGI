# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [0.3.0] — 2026-04-14 — Production Readiness

This release completes the **MouseDroidAGI Production Readiness** milestone across 7 phases,
bringing all cognitive, memory, voice, safety, and deployment subsystems to a production-ready
state on the NVIDIA Jetson Orin Nano. 2505 tests pass; branch-coverage gate ≥ 85%.

### Added — Phase 1: Deployment Hardening

- **Docker device passthrough** (`docker-compose.jetson.yml`)
  - All device mappings now active with env-var overrides: `${MOUSEDROID_ESP32_DEV:-/dev/ttyUSB0}`,
    `${MOUSEDROID_CAMERA_DEV:-/dev/video0}`, `${MOUSEDROID_LIDAR_DEV:-/dev/ttyUSB1}`, GPIO, audio
  - `group_add: [audio, video, dialout, gpio]` for correct device permissions
  - Docker `HEALTHCHECK` directive polling `/api/v1/health` (30s interval, 3 retries)
  - Persistent `promtail_positions` volume to survive restarts

- **Tick timeout + emergency stop** (`src/mousedroid/config/schema.py`, `orchestrator.py`)
  - `LoopConfig.tick_timeout_s` — configurable per-tick timeout (default 1.0 s, `gt=0`)
  - `asyncio.wait_for(self.tick(), timeout=tick_timeout)` wraps every orchestrator tick
  - `asyncio.TimeoutError` → `emergency_stop()` + critical log + voice error event
  - Unhandled exception in `tick()` → `emergency_stop()` + voice error event
  - `LoopConfig.watchdog_enabled`, `watchdog_interval_s` fields added

- **Systemd watchdog integration** (`src/mousedroid/health/watchdog.py`)
  - `WatchdogProtocol` — `@runtime_checkable Protocol` with `notify()` method
  - `SystemdNotifier` — sends `WATCHDOG=1` via `sdnotify` package or `systemd-notify` subprocess fallback
  - `FileHeartbeatNotifier` — writes monotonic timestamp to configurable path for Docker HEALTHCHECK
  - `NullNotifier` — no-op for mock/dev mode
  - `build_watchdog(cfg)` factory function auto-selects notifier based on environment + config
  - Orchestrator calls `watchdog.notify()` after each successful tick

- **systemd service hardening** (`scripts/mousedroid.service`, `scripts/mousedroid-docker.service`)
  - `Type=notify` + `WatchdogSec=30` on both service units
  - `ExecStartPre=/opt/mousedroid/scripts/preflight_check.sh` blocks startup on hardware failure
  - `MOUSEDROID_LOOP__WATCHDOG_ENABLED=true` injected into service environment

- **Pre-flight validation script** (`scripts/preflight_check.sh`)
  - Checks ESP32, camera, GPIO (required); LiDAR, audio (optional warnings)
  - Validates Docker/NVIDIA runtime, disk space (configurable `MOUSEDROID_MIN_DISK_GB`), config YAML syntax
  - Checks model weights presence (LLM + BDI)
  - Coloured PASS/FAIL/WARN output, exits non-zero on any required failure
  - Fully configurable via env vars: `MOUSEDROID_ESP32_DEV`, `MOUSEDROID_CAMERA_DEV`, `MOUSEDROID_LIDAR_DEV`

- **Docker env documentation** (`config/docker.env.example`)
  - Documents all device path env vars with default values and required/optional annotations

- **Pre-commit coverage hook extended** (`scripts/check_branch_coverage.py`)
  - Detects Pydantic Settings + coverage.py class-identity false-failure pattern
  - Falls through gracefully with `ALLOW_PYTEST_COLLECTION_SKIP=1` bypass

- **Tests** — `tests/unit/test_watchdog.py` (12 tests), `tests/unit/test_tick_timeout.py` (7 tests),
  `tests/integration/test_preflight_validation.py` (11 tests)

### Added — Phase 2: Memory & Curiosity Pipeline Wiring

- **`MemoryTier` dataclass** (`src/mousedroid/memory/tier.py`)
  - Groups `episodic`, `semantic`, `working`, and `consolidation` managers into a single injectable unit
  - `build_memory_tier(cfg)` factory function; enabled via `cfg.memory.enabled` (default `False`)

- **Orchestrator memory integration** (`src/mousedroid/orchestrator/orchestrator.py`)
  - Optional `memory_tier: MemoryTier | None` parameter in `MouseDroidOrchestrator.__init__`
  - Each tick: creates `ExperienceRecord` from obs + action + safety context; pushed to episodic + working memory
  - Background `asyncio.Task` runs `MemoryConsolidation.consolidate()` on `consolidation_interval_s` interval

- **Curiosity wiring**
  - ICM intrinsic reward computed from previous/current latent states each tick
  - `"curiosity"` key injected into `obs_dict` with per-channel curiosity scores
  - `SemanticIndex.retrieve()` queried for epistemic novelty when memory enabled

- **Tests** — `tests/unit/test_memory_tier.py` (8 tests), `tests/integration/test_memory_pipeline.py` (6 tests),
  `tests/unit/test_curiosity_wiring.py` (5 tests)

### Added — Phase 3: Voice & Rocky End-to-End

- **Startup/shutdown voice events** (`src/mousedroid/orchestrator/orchestrator.py`)
  - `start()` fires `"startup"` voice event after voice engine initialises
  - `stop()` fires `"shutdown"` voice event before teardown

- **Enriched voice context**
  - Emergency stop paths fire `"error"` voice event with safety context
  - `lidar_min_dist_m` included in obstacle voice events
  - Audio level RMS included in voice context when microphone available

- **Tests** — `tests/integration/test_orchestrator_voice_events.py` (8 tests)

### Added — Phase 4: Sensor Fusion Resilience

- **Sensor recovery protocol** (`src/mousedroid/sensing/manager.py`)
  - `async recovery_attempt() -> int` — tries to reinitialise failed sensors; returns recovered count
  - Orchestrator attempts recovery before triggering emergency stop on sensor degradation

- **Config additions** (`src/mousedroid/config/schema.py`)
  - `SafetyConfig.sensor_recovery_attempts` (default 1)
  - `SafetyConfig.sensor_recovery_delay_s` (default 0.5 s)

- **Self-healing orchestrator tests** — `tests/integration/test_self_healing_orchestrator.py` (9 tests)
- **Cascading sensor failure tests** — `tests/integration/test_cascading_sensor_failure.py` (11 tests)

### Added — Phase 5: LLM Gateway Integration

- **LLM gateway wired into orchestrator** (`src/mousedroid/factory.py`, `orchestrator.py`)
  - `build_llm_gateway(cfg)` called in `build_orchestrator()` when `cfg.llm.enabled`
  - `process_mission(nl_command)` method on orchestrator for NL → `GoalVector` translation
  - Rule-based parser first (< 1 ms for common commands); LLM fallback for complex/unknown commands
  - Prompt injection detection rejects malicious inputs

- **Degraded mode** (`src/mousedroid/llm_gateway/gateway.py`)
  - `start()` enters degraded mode (log warning, `_degraded=True`) instead of raising when
    `llama-cpp-python` or model file is missing — service continues operating safely

- **Tests** — `tests/integration/test_llm_gateway_wiring.py` (6 tests),
  updated `tests/unit/test_llm_gateway.py` (degraded-mode tests)

### Added — Phase 6: Jetson On-Device Validation Suite

- **Hardware E2E tests** — `tests/e2e/test_jetson_hardware_e2e.py` (marked `@pytest.mark.jetson`)
  - Camera, ultrasonic, ESP32, LiDAR, microphone, speaker, full 5-tick orchestrator loop with real sensors

- **Endurance tests** — `tests/performance/test_jetson_endurance.py`
  - 5-minute 30 Hz run; GPU temp < 85 °C; RSS stable within 10 %; loop p95 < 33 ms

- **Sensor verification script** (`scripts/verify_sensors.py`)
  - Updated with LiDAR + speaker checks, `--json` output flag for CI integration

### Added — Phase 7: Production Telemetry & Metrics

- **New Prometheus metrics** (`src/mousedroid/telemetry/metrics.py`)
  - `{ns}_memory_episodic_size`, `{ns}_memory_semantic_size` — episodic and semantic index size gauges
  - `{ns}_memory_working_size` — working memory context window size gauge
  - `{ns}_curiosity_intrinsic_reward` — intrinsic curiosity reward gauge per tick
  - `{ns}_voice_events` — voice event counter labelled by event type
  - `{ns}_llm_requests`, `{ns}_llm_latency_ms` — LLM gateway request counter + latency gauge (ms)
  - `{ns}_sensor_recoveries`, `{ns}_sensor_recovery_failures` — sensor recovery counters
  - All metric names use `{ns}` = `MetricsConfig.namespace` (default: `mousedroid`)

### Fixed

- **LLM gateway RuntimeError regression** — `gateway.py` `start()` no longer raises when
  `llama-cpp-python` is absent; uses degraded mode so tests relying on `build_orchestrator()`
  default config continue to pass
- **Pydantic Settings + coverage.py false failure** — `check_branch_coverage.py` pre-commit hook
  extended to detect `is_instance_of` + `Settings` coverage fingerprint and bypass cleanly
- **`test_file_heartbeat_notify_updates_timestamp` flakiness** — sleep increased to 50 ms
  to avoid race under load

### Changed

- **`config/schema.py`** — `LoopConfig` gains `tick_timeout_s`, `watchdog_enabled`,
  `watchdog_interval_s`; `SafetyConfig` gains `sensor_recovery_attempts`, `sensor_recovery_delay_s`;
  all new fields have defaults preserving full backward compatibility
- **`orchestrator.py`** — `run()` loop restructured around `asyncio.wait_for`; adds optional
  `watchdog` and `memory_tier` constructor parameters; enriches voice event context
- **`factory.py`** — adds `build_watchdog()`, `build_memory_tier()`, `build_llm_gateway()`
  (when `cfg.llm.enabled`) wired into `build_orchestrator()`
- **`docker-compose.jetson.yml`** — all devices uncommented with env-var overrides;
  healthcheck added; `group_add` permissions granted; Promtail positions volume added
- **`.gitignore`** — adds Serena workspace, heartbeat runtime files, LLM model dir,
  pre-flight output, validation output patterns

### Added

- **Dual-Stream CfC/GRU RSSM world model** — liquid neural network hybrid for adaptive reflexes
  - `DualStreamRSSM` — dual-stream architecture: GRU (slow planning, 256-dim) + CfC (fast reflexes, 64-dim) with concat fusion producing 320-dim combined hidden state
  - `CfCWrapper` — Closed-form Continuous-time cell wrapping `ncps.torch.CfC` with configurable backbone (units, layers, sparsity)
  - `StreamFusion` — concatenation-based fusion layer with `fuse()`, `extract_gru_state()`, `extract_cfc_state()` operations
  - `WorldModelProtocol` + `SafetyTraceProtocol` — `@runtime_checkable` protocol interfaces for world model DI
  - `DualStreamTrainingConfig` — Pydantic config for dual optimizers, gradient clipping, CfC loss warmup schedule
  - `ModelConfig` gains CfC fields: `cfc_hidden_dim`, `cfc_backbone_units`, `cfc_backbone_layers`, `cfc_mode`, `cfc_sparsity_level`
  - `build_world_model()` factory dispatch: `cfc_hidden_dim > 0` → `DualStreamRSSM`, else classic `RSSM`
  - `gru_parameters()` / `cfc_parameters()` — separate parameter groups for dual optimizer training
  - `get_safety_trace()` — extracts CfC hidden state from combined state for independent safety monitoring
- **Dual-stream training script** — `training/train_dual_stream_rssm.py` (712 LOC)
  - Dual Adam optimizers: GRU params (lr=3e-4) + CfC params (lr=1e-4)
  - Separate gradient clipping: GRU (max_norm=10.0) + CfC (max_norm=1.0)
  - Linear CfC loss weight warmup from 0.1→1.0 over 10k steps
  - Periodic fallback monitoring: logs CfC contribution quality, warns on >5% degradation
  - Full AMP support, checkpoint resume with dual optimizer states
  - CLI: `--config`, `--data`, `--device`, `--resume`, `--validate-only`
- **Jetson dual-stream config** — `config/jetson_dual_stream.yaml` with CfC activation gate
- **Human activation gate** — CfC disabled by default (`cfc_hidden_dim=0`); requires explicit `MOUSEDROID_MODEL__CFC_HIDDEN_DIM=64` to enable
- **HuggingFace model repo** — `ianshank/mousedroid-dual-stream-rssm` with 5-epoch validation weights + training metadata
- **57 new dual-stream tests**:
  - `test_cfc_cell.py` — CfC wrapper unit tests (initialization, forward, hidden dims)
  - `test_dual_stream_rssm.py` — DualStreamRSSM observe/imagine, protocol conformance, safety trace
  - `test_stream_fusion.py` — fusion layer, extract/fuse roundtrip
  - `test_dual_stream_training.py` — dual optimizer construction, warmup schedule, gradient clipping, checkpoint roundtrip
  - `test_dual_stream_compat.py` — factory dispatch, config backward compatibility, regression suite
  - `test_world_model_property.py` — Hypothesis property tests for rollout stability
  - `test_factory_integration.py` — integration tests for factory dispatch paths
- **ncps dependency** — `ncps>=0.0.7` added to `pyproject.toml` `[cfc]` extra and `Dockerfile.jetson`

### Changed

- **`Dockerfile.jetson`** — added `ncps>=0.0.7` install step (non-fatal graceful fallback)
- **`world_model/__init__.py`** — exports `DualStreamRSSM`, `CfCWrapper`, `StreamFusion`, protocol types
- **`factory.py`** — `build_world_model()` gains dual-stream dispatch branch

- **FHL-LD19 2D LiDAR sensor** — 5th modality integrated end-to-end through the cognitive stack
  - `LD19LidarDriver` — async UART driver with CRC8-validated binary protocol parsing
  - `LD19FrameParser` — LD19 packet parser with angle interpolation (n-1 intervals)
  - `LidarFeatureExtractor` — sector-binned distance features normalised to `[0, 1]`, vectorised via `np.minimum.at`
  - `MockLidar` — configurable mock driver for CI/testing
  - `ResilientLidarDriver` — circuit-breaker + retry wrapper for production reliability
  - `LidarScan` dataclass for typed scan data (angles, distances, confidences)
  - `LidarProtocol` — `@runtime_checkable Protocol` for DI
  - `LidarConfig` — Pydantic config with range validation, sector count, feature dim
  - `build_lidar()` / `build_lidar_feature_extractor()` factory functions
  - `SensorManager` gains LiDAR ring buffer + concurrent `_safe_lidar_read()`
  - `MultimodalEncoder` gains optional `lidar_proj` layer (enabled when `ModelConfig.lidar_dim > 0`)
  - `RSSM.observe_step()` threads LiDAR features through observation pipeline
  - `SafetyMonitor` evaluates LiDAR clearance via `SafetyConfig.lidar_max_range_m`
  - `TelemetryFrame.lidar_min_dist_m` — LiDAR distance surfaced in telemetry
  - `lidar_diagnostics` tool registered in tool registry
  - 12 new test files with 200+ LiDAR-specific tests
- **Wonrabai USB Sound Card** — combo mic + 8Ω 5W speaker on single USB interface
  - Speaker and voice engine enabled in `config/default.yaml` and `config/jetson_production.yaml`
  - Docker ALSA audio passthrough (`/dev/snd` + `group_add: [audio]`) in `docker-compose.jetson.yml`
  - 6 combo audio device tests verifying both mic and speaker discover the same USB device
- **Audio constants** — `POWER_CLIP_MAX` and `LOG_FLOOR` extracted to `hardware/audio/constants.py`

### Changed

- **`UsbMicrophone`** — renamed from "SuziePi" to generic USB; added graceful degradation
  matching `UsbSpeaker` pattern (try/except ImportError + OSError, return silence on failure)
- **`AudioFeatureExtractor`** — magic numbers `1e20` / `1e-10` replaced with named constants
- **`SafetyMonitor`** — `lidar_max_range_m` accessed directly from `SafetyConfig` field
  (was `getattr` with hardcoded `12.0` fallback)
- **`build_telemetry_frame()`** — uses `safety_ctx.lidar_min_dist_m` (actual metres)
  instead of raw normalised feature minimum
- **`MultimodalEncoder`** — missing LiDAR mask slot now treated as invalid (zeroed out)
  instead of silently passing unvalidated projection
- **`SensorManager._safe_lidar_read()`** — returns `ok=False` when feature extractor
  is missing (was `True`, feeding fake all-ones data marked valid)

### Fixed

- **LD19 angle interpolation** — fixed n-1 intervals formula (`step = diff / (n_points - 1)`)
- **3 mypy strict errors** — `torch.jit.save` untyped call, `depth_processor` Any return, stale `cv2` type-ignore
- **CRC test flakiness** — replaced probabilistic different-inputs-differ assertion with deterministic known test vectors (`0x74`, `0x4C`)
- **`usb_microphone.py` coverage** — removed from `pyproject.toml` coverage omit list

### Added (previous)

- **Audio integration into world model** — microphone data now flows end-to-end through the cognitive stack
  - `MultimodalEncoder` gains optional `audio_proj` layer (enabled when `ModelConfig.audio_dim > 0`)
  - `RSSM.observe_step()` extracts `audio_chunk` from observations and passes it to the encoder
  - `ModelConfig` gains `audio_dim` (default 0, backwards-compatible) and `audio_proj_dim` (default 32) fields
  - `config/default.yaml` enables microphone and sets `audio_dim: 1024`
- **Reusable camera feature extraction** — `FeatureExtractorProtocol` with pluggable backends
  - `src/mousedroid/hardware/camera/feature_extractor.py` — new module with `MeanPoolExtractor`, `TensorRTExtractor`, and `build_feature_extractor()` factory
  - `CameraConfig` gains `feature_extractor` (Literal `"mean_pool"` / `"tensorrt"` / `"auto"`) and `l2_normalize` (bool) fields
  - `TensorRTExtractor` loads ONNX models via `onnxruntime` with graceful fallback to mean-pool
- **Audio pipeline tests** — `tests/integration/test_audio_pipeline.py` (3 tests), 10 new encoder tests, 3 new RSSM tests
- **Feature extractor tests** — `tests/unit/test_feature_extractor.py` (13 tests) covering protocol compliance, L2-norm, TRT fallback
- **Config tests** — 9 new tests for `audio_dim`, `audio_proj_dim`, `feature_extractor`, `l2_normalize` fields

### Changed

- **`MicrophoneConfig.device_name`** — default changed from `"SuziePi"` to `"USB"` (matches common USB mics like TI PCM2902)
- **`JetsonCSICamera` / `IMX500Camera`** — feature extraction delegated to `FeatureExtractorProtocol`; 15 lines of duplicate mean-pool code removed from each driver
- **`MultimodalEncoder`** — docstring updated to reflect up-to-4-modality valid mask; `_AUDIO_IDX = 3` constant added
- **`telemetry/server.py`** — `isinstance(gpu_temp, (int, float))` → `isinstance(gpu_temp, int | float)` (UP038 lint fix)

### Fixed

- **Stale `type: ignore` comments** — removed `[no-redef]` on `jetson_csi.py` optional imports and `[untyped-decorator]` on `rssm.py` `@torch.no_grad()`
- **Duplicate feature extraction** — identical 15-line `_extract_features()` in both camera drivers replaced with shared `MeanPoolExtractor`
- **Ruff UP038 violation** — `isinstance(gpu_temp, (int, float))` in `telemetry/server.py` now uses union syntax

### Removed

- Duplicate `_extract_features()` implementations from `jetson_csi.py` and `imx500.py` (replaced by shared `feature_extractor.py`)

---

- **Phase A — Training pipeline resume + type cleanup** (`training/`)
  - `run_pipeline()` and `run_phase_1_rssm()` now accept `resume_from: Path | None`; CLI gains `--resume` flag to resume RSSM training from an existing checkpoint
  - `training/training_utils.py`, `train_bdi.py`, `collect_annotations.py`, `data_generator.py`, `warmstart_policy.py`, `train_constitutional_rl.py` — replaced bare `np.ndarray` annotations with `numpy.typing.NDArray[Any]`; added local `# type: ignore[attr-defined]` on factory-returned `object` access; `mypy training/ --ignore-missing-imports` now reports `Success: no issues found in 12 source files`
  - `warmstart_policy.py` — `tune_ucb()` now correctly passes `ucb_target_ms` when constructing the nested `MCTSConfig` candidate
- **Training test surface extended**
  - `tests/unit/test_run_pipeline.py` — added `test_resume_from_is_forwarded_to_phase_1`, `test_phase_0_and_2_runs_without_prior_rssm_artifact_if_phase_1_skipped`, `test_phases_2_3_4_require_missing_upstream_artifacts`
  - `tests/unit/test_warmstart_policy.py` — added `TestComputeLatentStatistics.test_returns_correct_shapes`, `TestRunWarmstart.test_run_warmstart_creates_artifacts`, `TestRunWarmstart.test_run_warmstart_passes_ucb_target_from_config`
- **Jetson production config activated** (`config/jetson_production.yaml`)
  - Cognitive core enabled: `cognitive.enabled: true`, `weights_dir: /opt/mousedroid/weights/bdi`, HF auto-download with up to 5 retries
  - Telemetry enabled: host `0.0.0.0:8080`, 10 Hz, mDNS broadcast, JSON serialisation
  - Prometheus metrics enabled at `/metrics` under `mousedroid` namespace
  - Jetson safety overrides: `min_valid_sensors: 1`, `battery_critical_v: 9.5`
- **HuggingFace weight publishing** — BDI, constitutional-RL, MCTS warmstart, and RSSM checkpoint weights uploaded to `ianshank/mousedroid-weights` (28 files, ~30 MB)
- **HuggingFace download subfolder fix** (`src/mousedroid/`)
  - `CognitiveConfig.huggingface_subfolder` field added (default `"bdi"`) — determines which subfolder of the HF repo contains the BDI `.npz` files
  - `download_weights_from_huggingface()` / `_download_file_with_retry()` gain `subfolder` and `local_dir` kwargs; `hf_hub_download` is now called with `subfolder=` and `local_dir=weights_dir.parent` so files land exactly at `weights_dir/belief.npz` etc.
  - `config/default.yaml` gains `cognitive.huggingface_subfolder: "bdi"`
  - Local production smoke confirms end-to-end: config overlay loads → BDI weights auto-downloaded from HF → `NeuralBDI` initialised with `weights_source=huggingface` → orchestrator `health_check` returns `status: ok`

- **WiFi/Ethernet Telemetry Server** — `src/mousedroid/telemetry/` — real-time remote monitoring over the local network
  - `TelemetryServer` — aiohttp-based REST + WebSocket server (`/api/v1/status`, `/api/v1/sensors`, `/api/v1/health`, `/api/v1/logs`, `/api/v1/network`, `/metrics`, `/ws`)
  - `TelemetryPublisher` — non-blocking async queue bridge; rate-limiting (≤60 Hz); drop-on-full semantics
  - `TelemetryFrame` — immutable frozen dataclass snapshot (all plain Python types; JSON/msgpack serialisable)
  - `LogRingBuffer` — structlog processor that captures the last *N* log entries for `/api/v1/logs`
  - `FrameBuilder` — converts `ObservationBundle` → `TelemetryFrame` each control-loop cycle
  - `NetworkInterface` discovery — uses stdlib `socket` only; no external dependencies
  - Optional API-key authentication (`X-API-Key` header), CORS middleware, mDNS/Zeroconf registration
  - Optional msgpack serialisation for binary-efficient WebSocket streaming
  - `MockTelemetryServer` — zero-dependency stub that satisfies `TelemetryServerProtocol` for CI/unit tests
- **Telemetry configuration** — `TelemetryConfig` Pydantic model added to `config/schema.py`
  - Fields: `enabled`, `host`, `port`, `publish_hz` (≤60), `queue_size`, `api_key`, `cors_origins`, `max_clients`, `mdns_enabled`, `mdns_service_name`, `serialization`, `log_stream_buffer`
- **`common/actions.py`** — `ActionNormalizer` utility extracted from orchestrator for reuse
- **Prometheus metrics registry** — `src/mousedroid/telemetry/metrics.py`
  - Pure-stdlib Prometheus text-format exporter (no third-party metrics dependency)
  - Config-driven metric namespace and per-metric toggles (no hardcoded metric names)
  - Tracks loop time, battery voltage, websocket clients, frame drops, safety violations, and GPU temperature
- **Telemetry smoke tests** — `tests/smoke/test_telemetry_smoke.py` (43 tests)
  - Covers full stack: `TelemetryFrame` → `LogRingBuffer` → `TelemetryPublisher` → `TelemetryServer` REST + WebSocket → E2E integration
  - All network I/O mocked to avoid DNS hangs on Windows; Windows-only `socket.getaddrinfo` test skipped with `@pytest.mark.skipif`
- **Telemetry unit tests** — 10 config, 14 network, 20+ server unit tests in `tests/unit/`
- **Telemetry integration test** — `tests/integration/test_telemetry_e2e.py`
- **Modular refactor** — `ab6b01c` — hard-coded values eliminated; `constants.py` expanded; dependency injection improved across `orchestrator`, `factory`, `cognitive_core`, `sensing/manager`

### Changed

- **`pyproject.toml`** — added `smoke` pytest marker; `aiohttp` added to `[server]` extras
- **Coverage config** — removed `src/mousedroid/telemetry/server.py` from coverage omit list so telemetry route changes are gated
- **`config/default.yaml`** — `telemetry` section with sensible defaults
- **`factory.py`** — `build_telemetry_server()` wires `TelemetryPublisher` → `TelemetryServer` → `Orchestrator`
- **`orchestrator.py`** — publishes `TelemetryFrame` each tick when telemetry enabled; lifecycle `start()`/`stop()` for server
- **`sensing/manager.py`** — `SensorManager` injects `TelemetryPublisher` for frame forwarding
- **`scripts/ci.sh`** — adds branch changed-line coverage gate (`scripts/check_branch_coverage.py --min 85`)
- **Git pre-commit hook** — local hook runs branch coverage gate automatically before commit when `src/mousedroid` Python files are modified

### Fixed

- **`tests/unit/test_cognitive_core.py`** — fixed I001 import sort
- **`tests/unit/test_telemetry_config.py`** — added `# noqa: S104` for `0.0.0.0`; `PT011` match patterns on all `pytest.raises`
- **`tests/unit/test_telemetry_network.py`** — SIM117 nested `with` blocks combined; Windows DNS-hang test skipped
- **`tests/unit/test_telemetry_server.py`** — E402 noqa after `importorskip`; network endpoints mocked to avoid real socket I/O
- **`tests/integration/test_docker_gpu.py`** — Jetson/container-specific assertions now guarded with `skipif` on non-Jetson hosts or non-L4T containers
- **`scripts/check_branch_coverage.py`** — branch coverage enforcement now based on changed executable lines instead of whole-file percentage
- **17 ruff violations** resolved across 4 PR test files

---

## [0.12.0] — Previous unreleased work

### Added

- **GPU Pre-Training Pipeline** — end-to-end orchestration for running phases natively on Jetson Orin Nano
  - `run_pipeline.py` orchestrator and native AMP support in `train_rssm.py`
  - GPU-accelerated MCTS rollouts in `warmstart_policy.py`
  - Native fallback logic and memory limit checks (6 GB default) via `GPUConfig`
  - Automated HuggingFace Hub artifact uploading via `upload_weights.py`
  - Full CI test-suite coverage (24 new unit tests added)

- **CognitiveCore integration** — dual-cadence BDI + metacognitive + constitutional loops wired into `MouseDroidOrchestrator`
  - Fast path (30 Hz): `PolicyMLP` + `ConstitutionalChecker` via `tick_fast()`
  - Slow path (~1 Hz): `NeuralBDI` inference + metacognitive updates via background `asyncio.Task`
  - Graceful fallback to MCTS agent on cognitive failure
- **`CognitiveConfig`** — Pydantic config in `schema.py` with HuggingFace auto-download, weights dir, fallback settings
- **`build_cognitive_core()`** — factory function with weight loading strategy (local → HuggingFace → random init)
- **`weights_manager.py`** — HuggingFace weight download with exponential backoff retry logic
- **21 new tests** — orchestrator cognitive paths (7), factory cognitive (4), weights manager (10)
- **`COVERAGE_ANALYSIS.md`** — coverage gap analysis and 85% enforcement plan
- **`TEST_SUITE_SUMMARY.md`** — detailed breakdown of all 21 cognitive test cases
- **`VALIDATION_CHECKLIST.md`** — step-by-step validation and CI/CD simulation guide
- **Docker GPU deployment** — `Dockerfile.jetson` using NVIDIA L4T PyTorch base (`dustynv/l4t-pytorch:r36.4.0`) with CUDA 12.6, TensorRT 10.4, and pycuda pre-installed
- **Docker Compose** — `docker-compose.jetson.yml` with NVIDIA runtime, optional hardware passthrough, and volume mounts
- **CI/CD pipeline** — `.github/workflows/ci.yml` with 5-stage pipeline (lint → typecheck → test → security → Docker) across Python 3.10/3.11 matrix
- **Docker GPU integration tests** — `tests/integration/test_docker_gpu.py` with auto-skip outside L4T container
- **Container test runner** — `scripts/jetson_test_runner.sh` for running categorised tests inside the container
- **Docker deploy script** — `scripts/docker_deploy.sh` for automated container deployment
- **Systemd Docker service** — `scripts/mousedroid-docker.service` for automatic container startup on boot
- **`.dockerignore`** — optimised Docker build context (excludes `.git`, caches, docs)
- **L4T container ADR** — `docs/architecture/ADR-l4t-container.md` documenting containerisation decision
- **Pre-built AI container ADR** — `docs/architecture/ADR-004-prebuilt-ai-containers.md` documenting multi-stage Docker build
- **Product requirements** — `docs/prd/prd-l4t-container-deployment.md`, `docs/prd/prd-prebuilt-llm-container.md`
- **Common utilities** — `src/mousedroid/common/math/numpy_ops.py` and `src/mousedroid/common/tools/registry.py` (reusable module extraction)
- **NVMe SSD support** — 500 GB NVMe partition, mount, 16 GB swap, Docker data-root, containerd symlink to SSD
- **4 GB → 16 GB swap** — SSD-backed swap file for memory-intensive builds (replaces zram-only swap)

### Changed

- **Coverage** — 54% → 97.34% (959 tests, 85% gate enforced by `pyproject.toml`)
- **Orchestrator** — cognitive core as primary action source with MCTS fallback; `start()`/`stop()` lifecycle for cognitive core
- **Factory** — `build_orchestrator()` now builds and injects `CognitiveCore` with graceful error handling
- **`bdi_model.py`** — replaced private `_relu`/`_safe_softmax_impl` with shared `numpy_ops` imports
- **`constitutional_rl.py`** — replaced private `_relu`/`_layer_norm` with shared `numpy_ops` imports
- **`tools/registry.py`** — added import from canonical `common.tools.registry` (backward compatible)
- **`tools/__init__.py`** — import from canonical `common.tools.registry`
- **Python compatibility** — ruff target `py311` → `py310`, mypy `python_version` 3.11 → 3.10 (Jetson JetPack 6.x ships Python 3.10)
- **`pyproject.toml`** — added `huggingface-hub` to `[llm]` extras
- **`factory.py`** — explicit `UltrasonicConfig` default values for all fields (mypy strict compliance)
- **`loader.py`** — removed stale `type: ignore[import-untyped]` on yaml import
- **`jetson_csi.py`** — fixed optional import types (`Any` annotation for `_jetson_utils` / `_cv2`)

### Fixed

- **`test_bdi_model.py`** — fixed stale `_relu` import (renamed to `relu` in `numpy_ops`)
- **`common/tools/registry.py`** — added `_mic_diagnostics` handler (9th tool, matching tests)
- **`weights_manager.py`** — fixed mypy `no-redef` via `_hf_hub_download` alias pattern
- **`test_docker_gpu.py`** — `_has_cuda()` moved before `pytestmark` (was undefined F821)
- **`numpy_ops.py`** — removed unused imports (F401), sorted `__all__`
- **`record.py`** — fixed import sort order (I001)
- **`registry.py`** — sorted `__all__` (RUF022)
- **21 lint errors** resolved (20 auto-fixed, 1 manual)
- **7 mypy errors** resolved across 4 source files

### Removed

- Deprecated modules consolidated into `common/` package with backward-compatible shims
