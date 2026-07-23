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

Run (2026-07-23, container CPU, seed 42):
`python scripts/compare_drift.py --episodes 8 --seq-len 48 --steps 60 --context-steps 8 --horizon 24 --seed 42 --memory both --out reports/drift_comparison.json`
(60 steps/arm; 25 of the augmented arm's batches drew the corrupted objective.)

| metric | baseline | augmented | delta (baseline − augmented; + = augmented better) |
|---|---|---|---|
| mean range MSE (headline) | 0.277645 | 0.273921 | **+0.003724 (~1.3%)** |
| final range MSE | 0.346944 | 0.354418 | −0.007474 |
| mean motor MSE | 0.083096 | 0.084898 | −0.001802 |
| mean latent_h MSE | 0.030423 | 0.030871 | −0.000448 |
| mean latent_z MSE | 1.766881 | 1.774075 | −0.007194 |
| mean motor_corrected MSE | — | 0.206431 | — |

**Verdict: mixed / near-null at this budget — documented honestly, not
overclaimed.** The augmented arm shows a small (~1.3%) improvement on the
mean headline (range) channel but is marginally worse on the final-step
headline and on the secondary channels. At 60 optimisation steps on synthetic
dynamics this is within run-to-run objective noise, so the honest reading is
**inconclusive-leaning-neutral**: the corrupted-history objective neither
demonstrably reduces nor demonstrably worsens drift at this scale. Notably,
the `motor_corrected` channel (residual head applied) is WORSE than the raw
decoded motor at this training budget — the evaluation-only head needs far
more corrupted batches than 25 to learn a useful residual. The decisive
comparison is the operator re-run against real replay data at a real training
budget (below); per the change's validation clause, a documented negative
result remains an acceptable outcome.

### Memory ablation (optional extra — not an R2 requirement)

RSSM-latent ablation at the warmup seam (B=1; distinct from the deployed
DualStream combined latent). With default `sink_warmup_ticks=30` ≥
`context_steps=8` the sink is never captured (the script warns), so this
measures the ring/EMA contribution only:

| metric | memory off | memory on |
|---|---|---|
| mean headline (range) MSE | 1.750252 | 1.751312 |
| mean latent_h MSE | 0.046967 | 0.046964 |

Effectively neutral on an untrained model over a short warmup — expected: the
memory's value proposition is long-horizon anchoring on a trained model, which
this micro-ablation does not exercise.

## Results — on-rover real replay (operator follow-up)

Pending: re-run against real LMDB replay data once the rover accumulates
enough records (`NEXT_STEPS.md` item 9). Record command, seed, and numbers
here.
