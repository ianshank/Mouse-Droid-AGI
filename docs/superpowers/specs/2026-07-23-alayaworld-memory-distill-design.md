# AlayaWorld-Adapted Bounded-Context Memory + Drift-Reduction Training (Design Spec)

**Date:** 2026-07-23
**Status:** Approved design (3-agent adversarial peer review incorporated)
**Branch:** `claude/mouse-droid-alayaworld-adapt-dt07mp`
**Feature:** F-023 · **ADR:** ADR-015 · **OpenSpec archive:** `openspec/changes/mouse-droid-alayaworld-memory-distill/`

---

## 1. Goal

Adapt two ideas from the AlayaWorld technical report (arXiv:2607.18367, as
characterized in the change request — the paper itself is unreachable from the
implementation environment) to the rover's recurrent-latent RSSM world model:

1. **Bounded-context memory** — a persistent "sink" anchor + compressed rolling
   history with a constant-size footprint, incorporated into predictions.
2. **Corrupted-history drift-reduction training** — train on the model's own
   drifted open-loop rollouts so it learns to recover toward ground truth.

Plus a **non-binding, scripts-only distillation feasibility spike** (k-step
imagination → one-forward student) ending in a written go/no-go.

Explicitly NOT adopted: the video diffusion transformer architecture, its 15B
scale, frame-level attention, or any iWorld-Bench-equivalent evaluation claim.
Results in this repo are internal synthetic-episode metrics, not benchmark
scores and not a parity claim.

## 2. Baseline — current state representation and memory scheme (task 4.2)

- The world model's state is the recurrent pair `(h, z)`:
  `h = hidden_dim (256)` for the plain `RSSM`, or the combined
  `hidden_dim + cfc_hidden_dim` for the deployed `DualStreamRSSM`;
  `z = latent_dim (64)`. All dims from `ModelConfig`.
- The interface is exactly `observe_step(observation, prev_action, h, z)` and
  `imagine_step(action, h, z)` (`world_model/protocol.py`) — **history is
  compressed entirely into the recurrent state**. There is no context window,
  ring buffer, sink, or summary anywhere in the world model.
- The orchestrator carries `_h/_z/_prev_action` tick-to-tick; the only related
  deque is `_latent_buffer` (NaN-recovery ring, maxlen
  `latent_recovery_buffer_size=5`) — a recovery cache, not a history.
- The observation set has **no pose channel**: `motor_state = [vx, vy, omega,
  battery]`, plus range (ultrasonic), lidar, vision, audio. Range is the only
  environment-coupled channel present in real replay records.
- `memory/` (working/episodic/semantic) is an orchestrator-layer subsystem; the
  world model never reads it. `WorkingMemory` (FIFO + dot-product attention) is
  the closest bounded-context primitive and this design reuses its attention
  math — with a deliberate cold-start divergence (see D1).

## 3. Scope decisions (locked)

| Decision | Value | Rationale |
|---|---|---|
| Memory placement | **Orchestrator observe seam** — blend into carried `(h, z)` after validation; no world-model signature change | Extending `observe_step/imagine_step` breaks 3 engines (torch/DualStream/ONNX) + both golden suites; off-loop-only could not influence predictions |
| Memory scope | **Engine-agnostic** (`h_dim = hidden_dim + cfc_hidden_dim`) | Operates on the carried tensors, not on engine internals |
| Ablation switch | `world_model_memory` Optional/None on `Settings`, `enabled=False`; factory returns `None`; disabled tick byte-identical | Charter §6 additive/opt-in; invariant #9 |
| Drift-training scope | **Concrete `RSSM` only** (feasibility vehicle); `DualStreamRSSM` port **deferred** | `RSSM` is the sole `train_sequence` engine; matches the on-device capability-gate precedent. Declared honestly in ADR-015 + spec deltas |
| Drift metric | Deterministic seeded per-modality open-loop MSE; **range headline**; zero-fill channels excluded; `valid_mask` threaded; + latent divergence | No pose exists; decoded-motor largely copies ground-truth actions fed to the rollout; range is the only environment-coupled replay channel |
| Residual objective | External evaluation-only `DriftCorrectionHead`, consumed by `measure_drift` | Literal requirement satisfied as a measured artifact; drift reduction in weights comes from the corrupted-prefix recovery objective |
| Distillation spike | `scripts/`-only, deterministic prior-MEAN teacher, γ-discounted k-step return | Non-production; stochastic teacher has an irreducible MSE floor; matches `MCTSPlanner._rollout`'s discounted accumulator |
| Config placement | `WorldModelMemoryConfig` Settings-level; `DriftTrainingConfig` nested `training.drift` | Runtime feature vs training-time surface (`training.replay` precedent) |
| Docs | This spec + plan + ADR-015 + `docs/related-work.md` + OpenSpec archive; **no c4 diagram** (growth precedent); Jetson runbook: yes | Operator-facing spike procedure needs a runbook |

## 4. Architecture & data flow

```text
30 Hz tick (unchanged order):
  read_all → safety.evaluate(obs) → _update_world_model → … → _select_action → …

_update_world_model (memory seam, default-OFF):
  observe_step(obs, prev_action, h, z) ──► (h_raw, z_raw)
  _validate_latent ──► (h_v, z_v, healthy)          # NaN-recovery ring unchanged
  if latent_context and healthy:
      latent_context.observe(h_v, z_v)              # store RAW (pre-blend) state
      h', z' = latent_context.contextualize(h_v, z_v)   # attention over {sink, EMA, ring}; λ-blend
      (isfinite guard: non-finite blend ⇒ identity)
  self._h, self._z = h', z'                          # feeds MCTS/VLA/cognitive + next tick

BoundedContextMemory stores (constant size = recent_size + 2 vectors):
  sink        frozen concat(h,z), captured after sink_warmup_ticks validated ticks;
              re-armed by reset() (OTA swap) and rearm_sink() (mission boundary)
  ring        deque(maxlen=recent_size) of detached hz clones
  long EMA    one vector, folded every `stride` ticks

Training path (offline, RSSM only):
  batch (B,T,…) ──► train_sequence_corrupted:
    prefix k ~ U[0, ⌊max_prefix_frac·T⌋] (private torch.Generator)
    steps 0..k-1  : open-loop prior rollout under no_grad (self-corruption)
    steps k..T-1  : shared _posterior_step helper (recon + free-bits KL), recovery_weight
    residual head : Δ(modality) vs ground truth, SEPARATE loss key, eval-only
  measure_drift: posterior warmup (context_steps) → open-loop rollout (horizon)
                 → per-modality MSE curves (range headline) + latent divergence
  scripts/compare_drift.py: seeded paired training baseline-vs-corrupted → JSON report
```

## 5. Peer-review findings this design closes

Three independent reviewers (adversarial design attack; convention/fact
verification; requirements-fidelity audit). Fixes are folded into §3/§4 and the
implementation contracts:

### CRITICAL resolved
- **NaN poisoning.** `_validate_latent`'s unrecoverable branch returns the NaN
  state unchanged; an unguarded memory would ingest it (EMA/sink poisoned
  forever, permanent NaN blend — strictly worse than today's self-healing).
  **Fix:** `_validate_latent` returns a `healthy` flag; unhealthy ticks skip
  observe+blend entirely; `observe()` drops non-finite inputs defensively;
  `contextualize` isfinite-checks the blend output and falls back to identity.

### MAJORS resolved
- **k=0 equality broken by defaults.** `generator=None` must construct a PRIVATE
  `torch.Generator` (a global-RNG prefix draw would shift every subsequent
  `randn_like`); `recovery_weight` provably inert at k=0; residual loss under a
  separate key. Shared `_posterior_step` helper extracted instead of duplicating
  ~50 lines (golden suites pin behavior, not source text).
- **Sink lifecycle.** A boot-time sink is stale hours later. `reset()` (OTA
  branch) clears everything AND re-arms warmup so a fresh sink is captured under
  the new weights; `rearm_sink()` runs at the mission-completed seam
  (`recapture_on_mission=True`) making the sink a per-mission anchor. Warmup
  counts all validated ticks including emergency ticks (they still run
  `_update_world_model`; per-mission recapture bounds staleness).
- **Cold-start damping.** Mirroring `WorkingMemory.attend`'s zero-vector return
  would damp `h` by (1-λ) every warmup tick. An uncaptured sink / never-folded
  EMA are EXCLUDED from the key set; empty key set ⇒ exact identity.
- **Metric honesty.** Decoded-motor MSE with ground-truth actions mostly copies
  the action through the GRU; replay batches zero-fill lidar/vision. Headline =
  range; zero-fill channels excluded; per-modality reported separately; latent
  divergence added; "no pose" declared.
- **`measure_drift` memory hook.** Applied ONLY during the posterior warmup
  (mirrors the deployment seam); the script warns when
  `sink_warmup_ticks >= context_steps`; measures the RSSM latent, distinct from
  the deployed DualStream combined latent. `--memory both` is an optional
  ablation extra.
- **Spike honesty.** MCTS `plan()` ≈ 500-650 `imagine_step` calls; depth-5
  rollouts ~40%; expansion needs intermediate states ⇒ consumer-level ceiling
  ~1.25-1.6× regardless of primitive speedup. Teacher must be deterministic
  (prior mean) and predict the γ-discounted return `_rollout` actually
  accumulates. Report separates primitive vs consumer levels; Jetson criterion
  pending the operator run.
- **Feedback honesty.** Blended `h` re-enters storage after one `observe_step`;
  the true invariant is "memory stores only posterior-corrected
  (observation-re-anchored) states". A rate-limited `latent_context_blend`
  debug event (‖c − hz‖, blend cosine) makes runaway self-reinforcement
  observable.

### Fact-check corrections applied
- `scripts/` is ruff-lint-only (no mypy/format/C901) and outside the coverage
  denominator — the spike stays there.
- The observe-step budget perf test times `observe_step` directly and does NOT
  cover the orchestrator blend ⇒ dedicated
  `tests/performance/test_latent_context_latency.py`.
- The portfolio-reframe AQA greps a fixed file tuple; new docs are safe, but the
  CLAUDE.md section must avoid the flagged tokens.
- Replay/experience records store raw modalities only (no latents) ⇒ on-device
  replay training data is NOT contaminated by blending.
- `measure_drift` lives in `training/` (the `validation/` package is
  deliberately torch-free).
