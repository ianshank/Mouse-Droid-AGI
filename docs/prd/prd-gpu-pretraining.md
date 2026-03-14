# PRD: GPU Pre-Training Pipeline for MouseDroid

> **Date**: 2026-03-13
> **Author**: Antigravity Agent
> **Status**: Draft — Awaiting Review

---

## User Story

**As a** MouseDroid developer,
**I want** a complete GPU-accelerated pre-training pipeline that runs all phases end-to-end on the Jetson Orin Nano,
**So that** I can produce trained weights (RSSM, BDI, MCTS policy, Constitutional RL) for autonomous navigation without relying on cloud compute.

---

## Background

MouseDroid has 4 sequential training phases, each producing weights consumed by the next:

| Phase | Script | Framework | GPU? | Output |
|-------|--------|-----------|------|--------|
| 0 — Data Gen | `data_generator.py` | PyTorch (mock) | Minimal | `sequences.pt` |
| 0b — Annotations | `collect_annotations.py` | NumPy | No | `bdi_annotations.npz` |
| 2.1 — RSSM | `train_rssm.py` | PyTorch | **Yes** | `rssm/final.pt` |
| 2.2 — Warm-start | `warmstart_policy.py` | PyTorch + NumPy | **Yes** | `mcts/policy_init.npz` |
| 2.3 — BDI | `train_bdi.py` | NumPy SGD | No | `bdi/*.npz` |
| 2.4 — Constitutional RL | `train_constitutional_rl.py` | NumPy + RSSM | **Partial** | `policy.npz`, `value.npz` |

---

## Acceptance Criteria

### AC-1: GPU Device Auto-Detection

**Given** the training scripts are launched on a Jetson with CUDA available,
**When** `--device` is not specified,
**Then** the scripts automatically select `cuda:0` and log the GPU name + memory.

### AC-2: End-to-End Pipeline Script

**Given** a clean environment with no pre-existing weights,
**When** `python -m training.run_pipeline --config config/mock_hardware.yaml` is executed,
**Then** all phases run sequentially, each consuming the outputs of the prior phase, producing final weights in `weights/`.

### AC-3: RSSM GPU Training

**Given** `sequences.pt` exists in the data directory,
**When** `train_rssm.py` runs with `--device cuda`,
**Then** training completes with GPU utilization > 50% and total time < 10x CPU time.

### AC-4: Memory-Safe Batch Sizing

**Given** the Jetson Orin Nano has 8 GB unified memory,
**When** any training script runs,
**Then** peak GPU memory never exceeds 6 GB (leaving headroom for system + display).

### AC-5: Checkpoint Resume

**Given** training was interrupted during training (e.g., between epochs),
**When** the same script is re-launched with `--resume`,
**Then** training resumes from the last saved checkpoint (typically the end of the last completed epoch) without restarting from scratch.

### AC-6: Weight Upload to HuggingFace

**Given** all phases complete successfully,
**When** `--upload` flag is set,
**Then** final weights are uploaded to `ianshank/mousedroid-weights` on HuggingFace Hub.

### AC-7: Docker Integration

**Given** the L4T Docker container is running,
**When** training scripts are executed inside the container,
**Then** GPU access works via `--runtime nvidia` and all phases complete successfully.

---

## Out of Scope

- Cloud-based distributed training (GCP Vertex AI)
- Real hardware data collection (camera, sensors)
- TensorRT export / quantization (separate epic)
- Hyperparameter search / AutoML

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Full pipeline wall-time (mock data) | < 2 hours on Jetson |
| GPU utilization during RSSM training | > 50% |
| Peak memory usage | < 6 GB |
| Test coverage for new code | > 80% |
| Weight file parity (CPU vs GPU) | MSE < 1e-5 |

---

## Open Questions

1. Should BDI training be migrated from numpy SGD to PyTorch for GPU acceleration, or is the numpy path fast enough?
2. What batch size should be the default for 8 GB Jetson unified memory?
3. Should checkpoints use `safetensors` format instead of `.pt` for security?
