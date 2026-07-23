# Spec Delta: world-model-training

## ADDED Requirements

### Requirement: Drift-Reduction Training via Corrupted-History Rollouts

The training pipeline SHALL support generating self-corrupted historical rollouts
and training the model to predict correction residuals toward the ground-truth
trajectory.

Implementation notes (declared interpretations — recorded so this delta does not
overstate coverage):

1. **Scope.** The objective applies to the concrete `RSSM` (the only engine with
   a gradient-enabled `train_sequence`). `RSSM` is the shared-paradigm
   feasibility vehicle; the deployed `DualStreamRSSM` port is explicitly
   deferred (ADR-015).
2. **Corruption source.** `RSSM.train_sequence_corrupted` rolls a random-length
   prefix OPEN-LOOP under the model's own prior (its self-generated drifted
   imagination, detached) and trains the posterior suffix to recover toward the
   ground-truth modalities — a scheduled-sampling-style recovery objective. This
   is the mechanism expected to reduce drift in the trained weights.
3. **Residual predictor.** The literal correction-residual predictor is the
   external `DriftCorrectionHead` (no parameters on the RSSM `state_dict`). It
   is trained jointly and CONSUMED by the drift evaluation (`measure_drift`
   reports drift with and without the residual correction) but is never deployed
   on the rover.
4. **Drift metric.** This robot has no pose channel (`motor_state =
   [vx, vy, omega, battery]`), so "pose error" is substituted by a deterministic
   seeded open-loop drift score: per-modality reconstruction MSE over an
   N-step prior rollout after a posterior warmup, with **range (ultrasonic) as
   the headline channel** (the only environment-coupled replay signal),
   zero-filled channels excluded, `valid_mask` respected, plus latent-space
   divergence versus the posterior-inferred trajectory.

#### Scenario: Model trained with corrupted-history augmentation reduces drift vs baseline

- **GIVEN** two models — one trained with standard rollouts only, one trained with
  corrupted-history augmentation (identical seeded initialization)
- **WHEN** both are evaluated on a held-out long-horizon rollout task
- **THEN** the corrupted-history-augmented model SHALL show measurably lower
  cumulative prediction drift under the agreed metric (above), or a documented
  negative result SHALL be recorded in
  `docs/analysis/alayaworld-drift-comparison.md`

### Requirement: Distillation Feasibility Spike (Non-Binding)

A time-boxed spike SHALL evaluate whether inference-step distillation (reducing
multi-step prediction to fewer steps) is feasible for the current MouseDroid world
model on target edge hardware, without committing to production adoption.

Implementation note: "multi-step prediction" maps to the MCTS planner's
sequential `imagine_step` composition; the spike distils a deterministic
prior-mean k-step teacher into a one-forward jump student
(`scripts/spike_step_distillation.py`, non-production). The spike report
separates primitive-level speedup from the planner-level ceiling (~1.25-1.6×,
because tree expansion requires intermediate states) and marks the Jetson
measurement as pending an operator on-device run
(`docs/runbooks/jetson-alayaworld-spike.md`). No iWorld-Bench-equivalent
evaluation is claimed.

#### Scenario: Spike produces a go/no-go recommendation

- **GIVEN** the spike is completed within its time-box
- **WHEN** results are reviewed
- **THEN** a written recommendation (adopt / defer / reject) with latency and
  accuracy trade-off data SHALL be produced
  (`docs/analysis/alayaworld-distillation-spike.md`; provisional/conditional
  until the Jetson measurement lands)
