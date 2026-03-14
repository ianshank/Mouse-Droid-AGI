# PRD: Sequential Pre-Training Execution Plan

> **Date**: 2026-03-13
> **Author**: Antigravity Agent
> **Status**: Draft — Awaiting Review

---

## User Story

**As a** MouseDroid developer,
**I want** a fully validated, production-quality sequential training execution that runs all 6 phases end-to-end with proper hyperparameters, checkpoint management, and convergence validation,
**So that** the Jetson Orin Nano has a complete set of trained weights (RSSM, BDI, MCTS policy, Constitutional RL) ready for autonomous navigation.

---

## Background

The GPU pre-training pipeline infrastructure is now in place (PR #11 merged), including:

- `run_pipeline.py` orchestrator with phases 0→0b→1→2→3→4
- GPU auto-detection, AMP support, checkpoint resume
- Memory-safe batch sizing for Jetson's 8 GB unified memory

**What's missing**: We have never run a **full**, **tuned** training cycle with validated convergence. The current defaults (100 epochs, 1000 episodes, batch_size=32) may not produce useful navigation weights. This PRD covers the work to run, validate, and iterate on a complete training cycle.

---

## Training Phases (Sequential)

| Phase | Script | Framework | GPU? | Input | Output | Est. Time |
|-------|--------|-----------|------|-------|--------|-----------|
| 0 | `data_generator.py` | PyTorch mock | Minimal | Config | `sequences.pt` | ~15 min |
| 0b | `collect_annotations.py` | NumPy | No | Config | `bdi_annotations.npz` | ~10 min |
| 1 | `train_rssm.py` | PyTorch + AMP | **Yes** | `sequences.pt` | `rssm/final.pt` | ~30-60 min |
| 2 | `warmstart_policy.py` | PyTorch + NumPy | **Yes** | `rssm/final.pt` | `mcts/policy_init.npz` | ~10 min |
| 3 | `train_bdi.py` | NumPy SGD | No | `bdi_annotations.npz` | `bdi/*.npz` (4 files) | ~5 min |
| 4 | `train_constitutional_rl.py` | NumPy + RSSM | Partial | `rssm/final.pt` + policy | `policy.npz`, `value.npz` | ~20-40 min |

**Total estimated wall-time**: ~90-135 minutes on Jetson Orin Nano.

---

## Acceptance Criteria

### AC-1: End-to-End Pipeline Completion

**Given** a clean environment with no pre-existing weights,
**When** `python -m training.run_pipeline --config config/training.yaml` is executed,
**Then** all 6 phases complete without errors, producing a full weight set in `weights/`.

### AC-2: RSSM Convergence

**Given** RSSM training runs for at least 200 epochs with AMP on GPU,
**When** training completes,
**Then** the reconstruction loss is monotonically decreasing and final loss is < 0.05.

### AC-3: BDI Accuracy

**Given** BDI sub-networks are trained on annotated data,
**When** the intention predictor is evaluated on a held-out set (~20%),
**Then** intention classification accuracy is > 60% across 10 classes.

### AC-4: Constitutional RL Safety

**Given** the Constitutional RL phase runs 5000+ training episodes,
**When** the final policy is evaluated,
**Then** the constitutional violation rate is < 5% and mean episode reward is positive.

### AC-5: Training Configuration

**Given** the project needs a dedicated training configuration,
**When** `config/training.yaml` is created,
**Then** it contains tuned hyperparameters for each phase with documentation comments.

### AC-6: Convergence Dashboard

**Given** training phases produce metrics,
**When** training completes,
**Then** a `training/results/training_report.json` summarizes per-phase metrics (loss curves, accuracy, timing).

### AC-7: Checkpoint Integrity

**Given** all phases save checkpoints,
**When** training completes,
**Then** all weight files can be loaded without errors and pass shape validation checks.

---

## Out of Scope

- Real camera/sensor data collection (uses mock hardware)
- Hyperparameter search / AutoML (manual tuning only)
- TensorRT export / quantization (separate epic)
- Cloud-based training (all local to Jetson)
- Real-world navigation evaluation

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Pipeline completes without errors | 100% |
| RSSM final reconstruction loss | < 0.05 |
| BDI intention accuracy | > 60% |
| Constitutional violation rate | < 5% |
| Total wall-time on Jetson | < 3 hours |
| Weight file integrity (all loadable) | 100% |

---

## Open Questions

1. Should we increase training episodes from 1000 to 5000 for better RSSM generalization?
2. What learning rate schedule should RSSM use (constant vs. cosine annealing)?
3. Should BDI training use train/val split for convergence monitoring?
4. How many Constitutional RL episodes are needed for policy convergence?
