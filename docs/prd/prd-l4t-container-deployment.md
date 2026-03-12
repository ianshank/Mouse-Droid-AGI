# PRD: L4T Container Deployment for GPU-Accelerated PyTorch

> **Epic**: `l4t-container-deployment`
> **Date**: 2026-03-11
> **Status**: Draft

---

## User Story

**As a** MouseDroidAGI developer,
**I want** to run the MouseDroid application inside an NVIDIA L4T PyTorch container,
**So that** I get GPU-accelerated PyTorch (CUDA) on the Jetson Orin Nano without building torch from source.

---

## Background

The Jetson Orin Nano runs Ubuntu 22.04 (JetPack 6.2, L4T R36.4.7) with Python 3.10.12. PyTorch wheels from PyPI and all NVIDIA pip indexes ship CPU-only builds for aarch64. NVIDIA provides pre-built L4T container images that include CUDA-compiled PyTorch — this is the only practical way to get GPU acceleration without a multi-hour source build.

### Current State

- mousedroid installed in `/opt/mousedroid/venv` with torch 2.5.1 **CPU-only**
- GPU confirmed: Orin (nvidia-smi 540.4.0), CUDA toolkit 12.6 installed
- 241 project files deployed, systemd service configured, 134 packages installed

---

## Acceptance Criteria

### AC1: Container Build

**Given** Docker with NVIDIA runtime is available on the Jetson,
**When** I run `docker compose -f docker-compose.jetson.yml build`,
**Then** a `mousedroid:jetson` container image is created based on L4T PyTorch.

### AC2: GPU Verification

**Given** the container is running,
**When** I execute `python3 -c "import torch; print(torch.cuda.is_available())"` inside the container,
**Then** it returns `True`.

### AC3: mousedroid Runs

**Given** the container is running with the project mounted,
**When** I execute `mousedroid --health-check --config /etc/mousedroid/default.yaml`,
**Then** the health check completes successfully.

### AC4: Hardware Passthrough

**Given** hardware devices are mapped into the container,
**When** the container starts with `--device` flags for UART, camera, and GPIO,
**Then** sensor reads (ESP32, camera, ultrasonic) succeed from inside the container.

### AC5: systemd Integration

**Given** the `mousedroid.service` is updated to manage the Docker container,
**When** I run `systemctl start mousedroid`,
**Then** the Docker container starts and the mousedroid application begins its sense-plan-act loop.

### AC6: Deploy Script

**Given** a developer runs `scripts/docker_deploy.sh`,
**When** the script completes,
**Then** the container is built, config is deployed, and the service is ready to start.

---

## Out of Scope

- Multi-node / swarm orchestration
- Remote container registry (NGC pull only, no push)
- Kubernetes deployment
- Container-based CI pipeline (Sprint 2)

---

## Success Metrics

| Metric | Target |
|--------|--------|
| `torch.cuda.is_available()` | `True` inside container |
| Container startup time | < 10s |
| 30 Hz loop latency | No regression vs. venv deployment |
| Disk usage | < 15 GB for image + layers |
