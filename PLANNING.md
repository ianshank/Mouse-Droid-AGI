# MouseDroidAGI — Project Plan: Sequential Training Execution

> **Date**: 2026-03-13
> **Author**: Antigravity Agent
> **Status**: Draft — Awaiting Review

---

## Goals

Run a complete, validated training cycle across all 6 phases of the MouseDroid pre-training pipeline on the Jetson Orin Nano, producing deployment-ready weights with verified convergence.

---

## Milestones

| # | Milestone | Target | Complexity |
|---|-----------|--------|------------|
| M1 | Training config + validation module | Sprint 1 | M |
| M2 | Full pipeline execution (all phases) | Sprint 1 | L |
| M3 | Convergence validation + results report | Sprint 1 | M |
| M4 | Test suite for new modules | Sprint 1 | M |
| M5 | Weight upload to HuggingFace | Sprint 1 | S |

---

## Epics

### Epic 1: Training Configuration (M1)

- Create `config/training.yaml` with tuned hyperparameters
- 3000 episodes, 200 RSSM epochs, kl_beta=0.5, batch_size=16
- PPO config: 5000 episodes, 128 rollout steps
- **Complexity**: S | **Dependencies**: None

### Epic 2: Convergence Validation Module (M1, M3)

- Create `training/validate_weights.py` — post-training checks
- Weight file existence, shape validation, loss thresholds
- BDI held-out accuracy evaluation, Constitutional RL violation rate
- Generate `training/results/training_report.json`
- **Complexity**: M | **Dependencies**: None

### Epic 3: Pipeline Execution (M2)

- Run `python -m training.run_pipeline --config config/training.yaml`
- Monitor RSSM loss convergence over 200 epochs
- Verify all 6 phases complete and produce expected weight files
- **Complexity**: L | **Dependencies**: Epic 1

### Epic 4: Test Suite (M4)

- Unit tests for `validate_weights.py`
- Integration test for end-to-end pipeline (with minimal config)
- Regression tests for training metrics thresholds
- **Complexity**: M | **Dependencies**: Epic 2

### Epic 5: Upload & Documentation (M5)

- Run `upload_weights.py` to push final weights to HuggingFace
- Update `CHANGELOG.md` with training results
- **Complexity**: S | **Dependencies**: Epics 3, 4

---

## Sprint Plan

### Sprint 1 — Full Training Execution

| Task | Epic | Est. |
|------|------|------|
| Create `config/training.yaml` | 1 | 0.5h |
| Create `training/validate_weights.py` | 2 | 2h |
| Add training report generation to pipeline | 2 | 1h |
| Run full pipeline end-to-end | 3 | 2-3h (wall-time) |
| Monitor and tune RSSM convergence | 3 | 1h |
| Create `test_validate_weights.py` | 4 | 1h |
| Upload weights + update CHANGELOG | 5 | 0.5h |

**Total estimated effort**: ~8-9 hours (including ~3h wall-time training)

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RSSM doesn't converge with synthetic data | Medium | High | Lower kl_beta, increase epochs or data |
| Jetson thermal throttling (>85°C) | Medium | Medium | Monitor temps, add pauses between phases |
| OOM during Phase 4 (RSSM + Policy) | Low | High | Reduce batch size, offload RSSM to CPU |
| Constitutional RL needs more episodes | Medium | Medium | Increase from 5000 to 10000 if needed |
| Data generation takes too long (3000 eps) | Low | Low | Reduce to 2000 and monitor quality |

---

## Blockers & Dependencies

- [x] GPU pre-training pipeline merged (PR #11)
- [x] `run_pipeline.py` orchestrator implemented
- [x] GPU utilities module (`gpu_utils.py`) implemented
- [ ] Verify RSSM converges on synthetic data
- [ ] Verify Jetson can sustain 200-epoch training without overheating
