# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

- **WiFi/Ethernet Telemetry Server** — `src/mousedroid/telemetry/` — real-time remote monitoring over the local network
  - `TelemetryServer` — aiohttp-based REST + WebSocket server (`/api/v1/status`, `/api/v1/sensors`, `/api/v1/health`, `/api/v1/logs`, `/api/v1/network`, `/ws`)
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
- **Telemetry smoke tests** — `tests/smoke/test_telemetry_smoke.py` (43 tests)
  - Covers full stack: `TelemetryFrame` → `LogRingBuffer` → `TelemetryPublisher` → `TelemetryServer` REST + WebSocket → E2E integration
  - All network I/O mocked to avoid DNS hangs on Windows; Windows-only `socket.getaddrinfo` test skipped with `@pytest.mark.skipif`
- **Telemetry unit tests** — 10 config, 14 network, 20+ server unit tests in `tests/unit/`
- **Telemetry integration test** — `tests/integration/test_telemetry_e2e.py`
- **Modular refactor** — `ab6b01c` — hard-coded values eliminated; `constants.py` expanded; dependency injection improved across `orchestrator`, `factory`, `cognitive_core`, `sensing/manager`

### Changed

- **`pyproject.toml`** — added `smoke` pytest marker; `aiohttp` added to `[server]` extras
- **`config/default.yaml`** — `telemetry` section with sensible defaults
- **`factory.py`** — `build_telemetry_server()` wires `TelemetryPublisher` → `TelemetryServer` → `Orchestrator`
- **`orchestrator.py`** — publishes `TelemetryFrame` each tick when telemetry enabled; lifecycle `start()`/`stop()` for server
- **`sensing/manager.py`** — `SensorManager` injects `TelemetryPublisher` for frame forwarding

### Fixed

- **`tests/unit/test_cognitive_core.py`** — fixed I001 import sort
- **`tests/unit/test_telemetry_config.py`** — added `# noqa: S104` for `0.0.0.0`; `PT011` match patterns on all `pytest.raises`
- **`tests/unit/test_telemetry_network.py`** — SIM117 nested `with` blocks combined; Windows DNS-hang test skipped
- **`tests/unit/test_telemetry_server.py`** — E402 noqa after `importorskip`; network endpoints mocked to avoid real socket I/O
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
