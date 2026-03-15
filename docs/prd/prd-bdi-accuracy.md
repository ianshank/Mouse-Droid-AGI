# PRD: BDI Intention Accuracy Improvement

**Epic:** E2 — BDI Accuracy  
**Feature Slug:** `bdi-accuracy`  
**Status:** Draft  
**Date:** 2026-03-14

---

## User Story

> **As** the MouseDroid cognitive core,  
> **I want** the BDI intention predictor to achieve ≥ 60% held-out accuracy,  
> **So that** the robot correctly infers the user's intent and plans appropriate actions without constant fallback to MCTS.

---

## Background

Post-fix held-out accuracy (on `training/data/bdi_annotations.npz`) is **55%** — below the 60% `validate_bdi_accuracy` threshold. BDI training runs `TrainingConfig.epochs = 100`. The BDI pipeline was previously masked by a matmul crash (`concat(belief, desire)` dimension bug, fixed 2026-03-14).

The forward pass chain is:  
`obs(256) → BeliefEncoder → belief(128) → DesireEncoder → desire(64) → IntentionPredictor → probs(10)`

---

## Acceptance Criteria

### AC-1: Extended Training Epochs via Config
**Given** a new `BDITrainingConfig` section is added to `Settings`,  
**When** BDI training is invoked via `train_bdi.py`,  
**Then** it must use `bdi_training.epochs` (default 300) rather than the shared `TrainingConfig.epochs` (default 100).

- `BDITrainingConfig.epochs: int = Field(300, gt=0)` in `schema.py`
- Backward compat: `bdi_training` section is optional; if absent, falls back to `training.epochs`

### AC-2: Class Balance Audit and Optional Rebalancing
**Given** the `bdi_annotations.npz` file contains 25,000 samples across 10 intention classes,  
**When** `collect_annotations.py` is invoked with `--balance-classes`,  
**Then** it must log the per-class sample counts and optionally oversample minority classes to within 20% of the largest class.

- Controlled by `BDITrainingConfig.balance_classes: bool = Field(False)`
- Log class distribution at INFO level regardless of flag

### AC-3: Observation Normalisation (Optional)
**Given** `BDITrainingConfig.normalise_observations: bool = Field(False)`,  
**When** `BeliefEncoder` is constructed with `normalise=True`,  
**Then** it must apply z-score normalisation using statistics computed from the training split before forwarding to the first linear layer.

- Default off for backward compatibility
- Mean/std saved alongside weight `.npz` files as `belief_norm_stats.npz`
- If normalised weights are loaded but `normalise=False`, a `DeprecationWarning` must be logged

### AC-4: Accuracy Validation CI Gate
**Given** the CI pipeline runs post-training validation,  
**When** `validate_bdi_accuracy()` is called in `validate_weights.py`,  
**Then** the phase must fail if accuracy < `bdi_training.accuracy_threshold` (default `0.60`).

- `accuracy_threshold: float = Field(0.60, gt=0, le=1)` in `BDITrainingConfig`
- Threshold is configurable via environment variable `MOUSEDROID_BDI_TRAINING__ACCURACY_THRESHOLD`

### AC-5: Test Coverage ≥ 80%
**Given** the `training/train_bdi.py` and `src/mousedroid/cognitive/bdi_model.py` modules,  
**When** `pytest --cov` is run,  
**Then** both files must show ≥ 80% branch coverage in the test report.

---

## Out of Scope

- Moving BDI training to a PyTorch model (numpy-only is intentional for edge inference)
- Multi-task learning with RSSM
- Online BDI fine-tuning on-device

---

## Success Metrics

| Metric | Current | Target |
|---|---|---|
| Held-out intention accuracy | 55% | ≥ 60% |
| `validate_bdi_accuracy` CI gate | Crashes (fixed) | PASS |
| `train_bdi.py` coverage | 0% | ≥ 80% |
| BDI inference latency (per call) | < 1 ms | < 1 ms (no regression) |

---

## Open Questions

1. Should class balancing use SMOTE or simple oversampling (simpler, more deterministic)?
2. Should `BeliefEncoder` norm stats be embedded in the `.npz` weights file or separate?
3. Is 10 intention classes the right granularity, or should we reduce to 5 to reduce the task difficulty?
