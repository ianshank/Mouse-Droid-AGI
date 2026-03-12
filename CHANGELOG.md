# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

- **Docker GPU deployment** — `Dockerfile.jetson` using NVIDIA L4T PyTorch base (`dustynv/l4t-pytorch:r36.4.0`) with CUDA 12.6, TensorRT 10.4, and pycuda pre-installed
- **Docker Compose** — `docker-compose.jetson.yml` with NVIDIA runtime, optional hardware passthrough, and volume mounts
- **CI/CD pipeline** — `.github/workflows/ci.yml` with 5-stage pipeline (lint → typecheck → test → security → Docker) across Python 3.10/3.11 matrix
- **Docker GPU integration tests** — `tests/integration/test_docker_gpu.py` with auto-skip outside L4T container
- **Container test runner** — `scripts/jetson_test_runner.sh` for running categorised tests inside the container
- **Docker deploy script** — `scripts/docker_deploy.sh` for automated container deployment
- **Systemd Docker service** — `scripts/mousedroid-docker.service` for automatic container startup on boot
- **`.dockerignore`** — optimised Docker build context (excludes `.git`, caches, docs)
- **L4T container ADR** — `docs/architecture/ADR-l4t-container.md` documenting containerisation decision
- **Product requirements** — `docs/prd/prd-l4t-container-deployment.md`
- **Common utilities** — `src/mousedroid/common/math/numpy_ops.py` and `src/mousedroid/common/tools/registry.py` (reusable module extraction)
- **NVMe SSD support** — 500 GB NVMe partition, mount, 16 GB swap, Docker data-root, containerd symlink to SSD
- **4 GB → 16 GB swap** — SSD-backed swap file for memory-intensive builds (replaces zram-only swap)

### Changed

- **Coverage** — 54% → 98.01% (752 tests, 85% gate enforced by `pyproject.toml`)
- **Python compatibility** — ruff target `py311` → `py310`, mypy `python_version` 3.11 → 3.10 (Jetson JetPack 6.x ships Python 3.10)
- **`pyproject.toml`** — added `huggingface-hub` to `[llm]` extras
- **`factory.py`** — explicit `UltrasonicConfig` default values for all fields (mypy strict compliance)
- **`loader.py`** — removed stale `type: ignore[import-untyped]` on yaml import
- **`jetson_csi.py`** — fixed optional import types (`Any` annotation for `_jetson_utils` / `_cv2`)

### Fixed

- **`test_docker_gpu.py`** — `_has_cuda()` moved before `pytestmark` (was undefined F821)
- **`numpy_ops.py`** — removed unused imports (F401), sorted `__all__`
- **`record.py`** — fixed import sort order (I001)
- **`registry.py`** — sorted `__all__` (RUF022)
- **21 lint errors** resolved (20 auto-fixed, 1 manual)
- **7 mypy errors** resolved across 4 source files

### Removed

- Deprecated modules consolidated into `common/` package with backward-compatible shims
