# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added — Post-Refactor Retrain (feat/post-refactor-retrain)

- **LLM Gateway schema integration** — `GatewayConfig` model added to `src/mousedroid/config/schema.py`
  with `llm_gateway: GatewayConfig | None` field on `Settings`. Fully optional with `None` default for
  backward compatibility. Config sections added to `config/jetson_production.yaml` (active) and
  `config/default.yaml` (commented template).
- **6 regression tests** — `TestTrainBdiNormStatsRegression` (3 tests in `test_bdi_training.py`) and
  `TestValidateBDIAccuracyNormRegression` (3 tests in `test_validate_weights.py`) exercising the
  BDI norm_stats and validation normalization fixes from PR #12.
- **Deploy script enhancements** — `scripts/deploy_remote.sh` gains `--weights` and `--with-llm`
  modes with `rsync_weights()` and `provision_llm()` helpers for selective Jetson deployment.
- **Git LFS setup** — `.gitattributes` tracking `*.npz`, `*.npy`, `*.pt`, `*.stl`, `*.FCStd` via
  Git LFS. 22 binary objects migrated (82 MB total).

### Changed — Post-Refactor Retrain

- **Docker v0.3.0** — `Dockerfile.jetson` version bumped from v0.2.0 to v0.3.0.
- **Docker Compose LLM support** — `docker-compose.jetson.yml` adds `MOUSEDROID_LLM_GATEWAY__MODEL_PATH`
  environment variable and `models` volume mount for Phi-3 Mini GGUF serving.
- **`torch.amp` migration** — `training/train_rssm.py` migrated from deprecated `torch.cuda.amp`
  to `torch.amp` API (`GradScaler("cuda")`, `autocast("cuda", ...)`). Forward-compatible with
  PyTorch 2.x deprecation timeline.

### Added

- **MCTS early-exit convergence** (2026-03-14): `early_exit_value_threshold` and `early_exit_patience`
  in `MCTSConfig` — search terminates early when best-child value stabilises. Default `0.01` threshold,
  `3` patience (backward-compatible: `0.0` disables).
- **MCTS multi-dim action sampling** (2026-03-14): `action_sampling: "uniform"` in `MCTSConfig` replaces
  1-D linspace broadcast with independent per-dimension uniform samples. Eliminates correlated
  action candidates that were limiting search diversity. Legacy `"linspace"` still available.
- **MCTS time-budget simulation** (2026-03-14): `simulation_budget_ms: float` in `MCTSConfig` — when
  set > 0, `plan()` exits early if wall-clock time exceeds the budget (uses `time.monotonic()`).
- **MCTS tree reuse config** (2026-03-14): `reuse_tree: bool = False` in `MCTSConfig` — plumbed for
  future warm-start implementation (requires human sign-off, see ADR-007).
- **8 new MCTS tests** — early-exit convergence (3), time-budget (2), action diversity (3)
- **`BDITrainingConfig`** (2026-03-14): dedicated training config for BDI phase — `epochs=300` (vs
  shared 100), `accuracy_threshold=0.60`, `balance_classes`, `normalise_observations`. Wired into
  `Settings.bdi_training` and accepted by `train_bdi()` with full backward-compat fallback.
- **BDI class balancing** (2026-03-14): `audit_class_balance()` logs distribution ratios. Opt-in
  `balance_classes()` inside `collect_annotations.py` deterministically oversamples minority classes
  to within 20% of the majority.
- **BeliefEncoder z-score normalisation** (2026-03-14): Opt-in `normalise` flag loads stats from
  `belief_norm_stats.npz`. Default stays `False` for backward compatibility. Emits deprecation
  warning if disabled but stats file is present.
- **CI Accuracy Gate** (2026-03-14): Added `scripts/check_report.py` to enforce `0.60` minimum
  BDI intention accuracy during GitHub Actions pipeline. Added new `bdi-accuracy` job after testing.
- **BDI test coverage** (2026-03-14): Added comprehensive `test_bdi_features.py` (31 total BDI
  module tests), bringing combined branch coverage of `bdi_model.py`, `train_bdi.py`, and
  `collect_annotations.py` to 93%.

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

- **`validate_weights.py` — BDI matmul dimension crash** (2026-03-14): `validate_bdi_accuracy` was
  concatenating `belief` (128-dim) and `desire` (64-dim) to produce a 192-dim vector, then multiplying
  by `intention.w1` which has input dimension 64. Fixed to use only `desire` output for the intention
  layer, matching the trained weight shape. Root cause: wrong network topology assumption in the forward
  pass. Regression test: `TestBdiBugRegression`.
- **`validate_weights.py` — wrong path for constitutional_rl weights** (2026-03-14):
  `validate_constitutional_rl` looked for `policy.npz` / `value.npz` in `weights/` root but the files
  live in `weights/constitutional_rl/`. Fixed path to use `weights_dir / "constitutional_rl" / "policy.npz"`.
  Regression test: `TestConstitutionalRlPathRegression`.
- **`validate_weights.py` — MCTS latency regression not detected** (2026-03-14): Added
  `validate_mcts_latency()` which reads `mcts/tuned_config.json` and raises an error when best-UCB p50
  latency exceeds the 50 ms target. Last run showed 219 ms (4.4× over). Regression test:
  `TestMctsLatencyRegression`.


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
