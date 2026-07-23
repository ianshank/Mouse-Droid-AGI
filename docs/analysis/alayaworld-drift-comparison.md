# AlayaWorld Drift Comparison — Corrupted-History Training (F-023)

**Status:** template — filled by WS5 (in-container synthetic run) and later by
the operator real-replay run.
**Harness:** `scripts/compare_drift.py` → `training/drift_reduction.py::train_pair_and_compare`
→ `training/drift_metrics.py::measure_drift`.
**Claims discipline:** results below are internal synthetic-episode metrics on
this repo's RSSM — NOT benchmark scores, NOT an AlayaWorld/iWorld-Bench
comparison or parity claim. A documented negative result is an acceptable
outcome of this comparison.

## Method

Two RSSMs from IDENTICAL seeded initialisations train side-by-side — one on
the standard `train_sequence` objective, one with corrupted-history
augmentation (`train_sequence_corrupted`: a random open-loop prior prefix —
the model's own drifted imagination — followed by a posterior recovery
suffix; per-batch seeded coin at `corruption_prob`). Both arms re-seed the
global RNG identically before every step, so differences can only come from
the objective.

Drift metric: after training, `measure_drift` runs a posterior warmup
(`eval_context_steps` on ground truth) then an open-loop prior rollout
(`eval_horizon`, ground-truth actions), reporting per-step per-channel MSE:

- **`range` (headline)** — the only environment-coupled channel; this robot
  has **no pose channel** (`motor_state = [vx, vy, omega, battery]`), so the
  change request's "pose error" is substituted by range reconstruction drift.
- `motor` (secondary — with ground-truth actions fed, it largely copies the
  action through the GRU), `latent_h`/`latent_z` (divergence vs the
  posterior-inferred trajectory), `motor_corrected` (the evaluation-only
  `DriftCorrectionHead` residual applied — trained-but-not-deployed).

Scope: concrete `RSSM` feasibility vehicle only; the deployed `DualStreamRSSM`
port is explicitly deferred (ADR-015).

## Results — in-container synthetic run (WS5)

> To be filled by WS5 with the seeded run:
> `python scripts/compare_drift.py --episodes 8 --seq-len 48 --steps 60 --memory both --out reports/drift_comparison.json`

| metric | baseline | augmented | delta (baseline − augmented; + = augmented better) |
|---|---|---|---|
| mean range MSE (headline) | TBD | TBD | TBD |
| final range MSE | TBD | TBD | TBD |
| mean motor MSE | TBD | TBD | TBD |
| mean latent_h MSE | TBD | TBD | TBD |
| mean latent_z MSE | TBD | TBD | TBD |
| mean motor_corrected MSE | — | TBD | — |

**Verdict:** TBD (improvement / documented negative result).

### Memory ablation (optional extra — not an R2 requirement)

RSSM-latent ablation at the warmup seam (B=1; distinct from the deployed
DualStream combined latent):

| metric | memory off | memory on |
|---|---|---|
| mean headline MSE | TBD | TBD |

## Results — on-rover real replay (operator follow-up)

Pending: re-run against real LMDB replay data once the rover accumulates
enough records (`NEXT_STEPS.md` item 9). Record command, seed, and numbers
here.
