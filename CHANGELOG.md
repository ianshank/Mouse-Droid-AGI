# Changelog

All notable changes to MouseDroidAGI are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Real-hardware GPIO integration tests (marked `@pytest.mark.hardware`)
- TensorRT engine compilation CI workflow for Jetson
- LLM gateway integration with Llama-3 GGUF model
- Training pipeline: RSSM pretraining on simulated trajectories
- Weights upload and versioning via Hugging Face Hub
- Dashboard: Grafana + Prometheus metrics export

---

## [0.2.0] — 2026-03-08

### Added
- **Shared utility**: `comms/_utils.py` — single `clamp()` function replacing duplicated logic in `serial_driver.py` and `wifi_driver.py` (DRY principle)
- **Protocol coverage tests**: `tests/unit/test_protocol_coverage.py` — all 8 pure-interface protocol modules now covered at 100%
- **Experience logger tests**: `tests/unit/test_experience_logger.py` — edge cases: log-without-open, flush-on-threshold, read-missing-key
- **Factory coverage tests**: `tests/unit/test_factory.py` — real-hardware branch paths for serial/WiFi ESP32, IMX500 camera, HC-SR04
- **LLM gateway tests**: `tests/unit/test_llm_gateway.py` — start/translate-with-model, parse-response, stop, error paths
- **Orchestrator full-coverage tests**: `tests/unit/test_orchestrator.py` — tick cycle, emergency stop, all 3 sensor failure modes, run-loop exception recovery
- **BDI weight-loading tests**: `tests/unit/test_bdi_model.py` — `.npz` file loading for all 4 sub-networks (BeliefEncoder, DesireEncoder, IntentionPredictor, AffectEstimator)
- **Cognitive core slow-loop test**: `tests/unit/test_cognitive_core.py` — verifies BDI inference runs in slow background task
- **Tool registry builtin dispatch tests**: `tests/unit/test_tool_registry.py` — all 8 built-in tool handlers exercised
- **Serial/WiFi driver full coverage**: `tests/unit/test_serial_driver.py`, `tests/unit/test_wifi_driver.py` — all async methods mocked and exercised

### Changed
- **`cognitive/metacognitive.py`**: Extracted magic numbers `33.0` and `100.0` to named constants `_TARGET_LOOP_MS` and `_LOOP_SCORE_SCALE`
- **`cognitive/constitutional_rl.py`**: Extracted `64` (hidden layer size) to `_POLICY_HIDDEN_DIM` constant; applies to both `PolicyMLP` and `ValueMLP`
- **`orchestrator/orchestrator.py`**: Added `exc_info=True` to 3 sensor-read warning log calls for full stack trace capture in production logs
- **`comms/serial_driver.py`**: Imports `clamp` from shared `comms/_utils.py`; local `_clamp` definition removed
- **`comms/wifi_driver.py`**: Imports `clamp` from shared `comms/_utils.py`; local `_clamp` definition removed

### Fixed
- Bare warning logs in orchestrator `_sense()` method now capture full exception context (`exc_info=True`)

### Metrics
| Metric | Before (0.1.0) | After (0.2.0) |
|--------|---------------|---------------|
| Tests | 457 | **530** |
| Coverage | 89% | **98.6%** |
| Ruff errors | 0 | 0 |
| Mypy strict errors | 44 (pre-existing) | 44 (unchanged) |

---

## [0.1.0] — 2026-03-07

### Added — Full Initial Implementation

#### Core Infrastructure
- `pyproject.toml` with Hatchling build system, full dependency matrix, ruff/mypy/pytest/coverage configuration
- `config/schema.py` — Pydantic v2 settings schema: 20+ nested config models, all with defaults for backward compatibility
- `config/loader.py` — YAML config overlay loader with type-safe merging
- `logging/setup.py` — structlog JSON logger factory with colour console dev mode
- `factory.py` — Platform-agnostic dependency injection factory; sole point of concrete-type imports

#### Hardware Layer
- `hardware/protocols.py` — `VisionProtocol`, `DistanceSensorProtocol` runtime-checkable interfaces
- `hardware/camera/imx500.py` — Raspberry Pi IMX500 AI Camera driver (onboard inference)
- `hardware/camera/mock_camera.py` — Zero-dependency mock for CI and dev
- `hardware/sensors/ultrasonic.py` — HC-SR04 GPIO driver via Jetson.GPIO
- `hardware/sensors/mock_ultrasonic.py` — Configurable mock distance sensor
- `hardware/motors/mock_motors.py` — Motor state mock

#### Communication
- `comms/protocol.py` — `ESP32CommProtocol` + `EncoderReading` dataclass
- `comms/serial_driver.py` — UART serial driver, all I/O via `asyncio.to_thread`
- `comms/wifi_driver.py` — HTTP driver using stdlib `urllib`, all I/O via `asyncio.to_thread`
- `comms/mock_driver.py` — In-memory mock with configurable encoder/battery responses

#### Sensing
- `sensing/bundle.py` — `MouseDroidObservationBundle` (timestamp, vision, distance, motor, validity mask)
- `sensing/protocol.py` — `ObservationProtocol` structural interface
- `sensing/manager.py` — Concurrent sensor fan-out with per-sensor `deque` ring buffers and graceful degradation

#### World Model (Pillar 1)
- `world_model/encoder.py` — Multi-modal observation encoder (vision + ultrasonic + motor)
- `world_model/rssm.py` — Recurrent State-Space Model latent dynamics
- `world_model/mcts.py` — Adaptive Monte Carlo Tree Search planner (50–200 simulations)
- `world_model/protocol.py` — `WorldModelProtocol` interface

#### Cognitive Architecture (Pillar 2)
- `cognitive/bdi_model.py` — Neural Belief-Desire-Intention pipeline (4 numpy MLPs, optional `.npz` weight loading)
- `cognitive/metacognitive.py` — 8-dimensional capability tracker with EMA + NetworkX causal graph
- `cognitive/constitutional_rl.py` — Constitutional safety checker, `PolicyMLP`, `ValueMLP`, `CuriosityAggregator`, `FlowCalculator`
- `cognitive/cognitive_core.py` — Dual-cadence controller: 30 Hz fast tick + 1 Hz slow BDI/metacog background loop

#### Memory Systems (Pillar 3)
- `memory/working.py` — Fixed-size token context window
- `memory/episodic.py` — FAISS-indexed episodic memory with similarity retrieval
- `memory/semantic.py` — Concept graph with FAISS semantic search
- `memory/consolidation.py` — Async consolidation from episodic → semantic
- `memory/protocol.py` — `MemoryProtocol` + `ReplayBufferProtocol` interfaces

#### Continual Learning (Pillar 4)
- `learning/ewc.py` — Elastic Weight Consolidation Fisher-information regularisation
- `learning/progressive.py` — Progressive neural network with lateral connections

#### Meta-Learning (Pillar 5)
- `meta/maml.py` — Model-Agnostic Meta-Learning (MAML) with inner/outer loop
- `meta/in_context.py` — In-context few-shot adaptation

#### Curiosity (Pillar 6)
- `curiosity/icm.py` — Intrinsic Curiosity Module: forward + inverse models

#### Growth (Pillar 7)
- `growth/distillation.py` — Knowledge distillation from teacher to student network

#### Reward Modelling (Pillar 8)
- `reward/model.py` — Constitutional multi-objective reward (truthfulness, helpfulness, safety, engagement)
- `reward/aggregator.py` — Weighted reward aggregation with configurable weights

#### Scaling (Pillar 9)
- `scaling/moe.py` — Mixture-of-Experts with top-K routing
- `scaling/adaptive.py` — Adaptive compute: early-exit based on surprise threshold

#### Safety (Pillar 10)
- `safety/context.py` — `SafetyContext` dataclass (is_emergency, violations, scores)
- `safety/monitor.py` — `MouseDroidSafetyMonitor` with clearance, velocity, sensor-staleness checks

#### Supporting Modules
- `efficiency/tensorrt.py` — TensorRT engine compilation wrapper
- `efficiency/profiler.py` — Latency profiler with ring-buffer stats
- `experience/logger.py` — LMDB-backed experience logger with auto-flushing
- `experience/record.py` — `MouseDroidExperienceRecord` msgpack serialisation
- `llm_gateway/gateway.py` — Natural language → GoalVector via local Llama GGUF model
- `llm_gateway/config.py` — Gateway configuration model
- `llm_gateway/protocol.py` — `LLMGatewayProtocol` + `GoalVector`
- `health/monitor.py` — Jetson sysfs GPU temperature and load monitor
- `tools/registry.py` — Agentic tool registry with 8 built-in handlers
- `agents/base.py` — `AgentProtocol` interface
- `agents/navigation.py` — `MouseDroidNavigationAgent` (MCTS + RSSM)
- `agents/_planning.py` — Planning utilities
- `orchestrator/orchestrator.py` — Main 30 Hz sense-plan-act loop

#### Configuration Files
- `config/default.yaml` — Full default config with all 20+ sections
- `config/jetson_production.yaml` — Jetson Orin Nano production overrides
- `config/mock_hardware.yaml` — Mock hardware preset

#### Scripts
- `scripts/ci.sh` — CI validation script
- `scripts/deploy_jetson.sh` — Jetson deployment script
- `scripts/flash_esp32.sh` — ESP32 firmware flash script
- `scripts/mousedroid.service` — systemd unit file

#### Tests
- 457 tests across `tests/unit/` and `tests/integration/`
- 89% coverage at initial release
- All tests pass in mock-hardware mode (no GPIO required)
