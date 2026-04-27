# MouseDroidAGI — Project Plan: L4T Container Deployment

> **Date**: 2026-03-11
> **Author**: Antigravity Agent
> **Status**: Historical containerization plan — baseline implemented; follow-up is operational validation

> **2026-04-26 rebaseline**: containerization milestones M1-M5 are complete on the current branch,
> overlay sync is automated via `scripts/sync_jetson_overlay.sh` as a non-fatal `ExecStartPre`
> step, and the active Jetson production scope excludes HC-SR04 / robot-arm work.

---

## Goals

Containerize MouseDroidAGI using NVIDIA's L4T PyTorch container to enable **GPU-accelerated PyTorch** on the Jetson Orin Nano, replacing the current CPU-only venv deployment.

---

## Milestones

| # | Milestone | Target | Complexity |
|---|-----------|--------|------------|
| M1 | Dockerfile + docker-compose for L4T PyTorch | Sprint 1 | M |
| M2 | GPU-verified mousedroid container running on Jetson | Sprint 1 | M |
| M3 | Hardware device passthrough (GPIO, UART, Camera) | Sprint 1 | L |
| M4 | Updated deploy scripts + systemd integration | Sprint 1 | S |
| M5 | Container-aware test suite + CI support | Sprint 2 | M |
| M6 | Documentation + architecture updates | Sprint 2 | S |

---

## Epics

### Epic 1: Container Infrastructure (M1, M2)

- **Dockerfile.jetson** based on `nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3`
- **docker-compose.jetson.yml** with runtime: nvidia, device mounts, volumes
- Install mousedroid + deps inside the container
- Verify `torch.cuda.is_available() == True`
- **Complexity**: M | **Dependencies**: Docker installed on Jetson (confirmed)

### Epic 2: Hardware Passthrough (M3)

- Pass `/dev/ttyUSB0` (ESP32 UART), `/dev/video*` (camera), `/dev/gpiochip*` (GPIO)
- Pass `/sys/devices/virtual/thermal/` for health monitoring
- Verify sensor reads work from inside the container
- **Complexity**: L | **Dependencies**: Epic 1 | **Risk**: GPIO access may need `--privileged`

### Epic 3: Deploy & Service Updates (M4)

- Update `deploy_remote.sh` to build + start container instead of venv
- Create `scripts/docker_deploy.sh` for container lifecycle
- Update `mousedroid.service` to manage Docker container via systemd
- **Complexity**: S | **Dependencies**: Epic 1, 2

### Epic 4: Testing & CI (M5)

- Container smoke test: GPU + import + health check
- Integration tests running inside the container
- Docker build step in CI pipeline
- **Complexity**: M | **Dependencies**: Epic 1

### Epic 5: Documentation (M6)

- Architecture ADR for containerization decision
- PRD for the feature
- Updated README deployment section
- **Complexity**: S | **Dependencies**: All epics

---

## Sprint Plan

### Sprint 1 (Epics 1-3) — Container Foundation

| Task | Epic | Est. |
|------|------|------|
| Create `Dockerfile.jetson` | 1 | 2h |
| Create `docker-compose.jetson.yml` | 1 | 1h |
| Verify GPU torch inside container | 1 | 1h |
| Add device passthrough config | 2 | 2h |
| Test hardware access from container | 2 | 2h |
| Create `scripts/docker_deploy.sh` | 3 | 1h |
| Update `mousedroid.service` for Docker | 3 | 1h |

### Sprint 2 (Epics 4-5) — Testing & Docs

| Task | Epic | Est. |
|------|------|------|
| Container smoke test script | 4 | 1h |
| Docker-aware CI step | 4 | 2h |
| Write ADR + PRD | 5 | 1h |
| Update README | 5 | 0.5h |

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| L4T container image not available for R36.4 | Medium | High | Fall back to JetPack 6.1 image or build from NGC catalog |
| GPIO access restricted in container | Medium | Medium | Use `--privileged` flag or specific device cgroups |
| Container overhead impacts 30 Hz loop | Low | High | Benchmark; container overhead is minimal for compute-bound workloads |
| SD card space insufficient for container image | Low | Medium | L4T images are ~8-10 GB; 32 GB free |

---

## Blockers & Dependencies

- [x] Docker installed on Jetson (confirmed: Docker Desktop running)
- [x] NVIDIA container runtime configured (required for `--runtime nvidia`)
- [ ] Verify exact L4T container tag availability: `r36.4.0-pth2.5-py3`
- [ ] Confirm hardware device nodes accessible from container
