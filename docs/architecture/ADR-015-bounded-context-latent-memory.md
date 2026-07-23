# ADR-015 — Bounded-Context Latent Memory + Corrupted-History Drift Training

## Status

Accepted — 2026-07-23 (F-023; design spec
`docs/superpowers/specs/2026-07-23-alayaworld-memory-distill-design.md`;
OpenSpec archive `openspec/changes/mouse-droid-alayaworld-memory-distill/`)

## Context

The world model compresses all history into the recurrent `(h, z)` carried by the
orchestrator; nothing anchors long-horizon context, and training never exposes the
model to its own drifted open-loop rollouts. The AlayaWorld technical report
(arXiv:2607.18367 — cited as characterized in the requesting change; the paper is
unreachable from the implementation environment and was not independently
verified) demonstrates, for a video diffusion world model, (a) a bounded visual
context built from a persistent sink frame + compressed history and (b)
corrupted-history training that reduces long-horizon drift. The transferable
ideas are the memory-management pattern and the training strategy — NOT the video
diffusion architecture, its scale, or its benchmark results. Results produced
under this ADR are internal synthetic-episode metrics; no iWorld-Bench
equivalence or parity is claimed.

## Decision

1. **Memory at the orchestrator observe seam.** A `BoundedContextMemory`
   (`world_model/bounded_context.py`, `LatentContextProtocol` in
   `world_model/protocol.py`) stores a per-mission frozen sink anchor
   (`concat(h,z)` captured after `sink_warmup_ticks`), a recent ring
   (`deque(maxlen=recent_size)`), and one EMA long-summary vector — constant
   `recent_size + 2` footprint. Retrieval is scaled dot-product softmax
   attention; the context is λ-blended into the carried `(h, z)` immediately
   after `_validate_latent`. Default-OFF (`world_model_memory` Optional/None on
   `Settings`); `build_latent_context` returns `None` when absent/disabled and
   the tick path is byte-identical.
2. **NaN contract.** `_validate_latent` surfaces a `healthy` flag; unhealthy
   ticks skip observe+blend entirely (today's self-healing NaN recovery is
   preserved); `observe()` drops non-finite inputs; the blend output is
   isfinite-guarded with fallback-to-identity. Cold-start: an uncaptured sink /
   never-folded EMA are excluded from the key set and an empty key set yields
   the exact identity (deliberate divergence from `WorkingMemory.attend`'s
   zero-vector return, which would damp `h` during warmup).
3. **Sink lifecycle.** `reset()` (called in the OTA weight-swap branch beside
   `_latent_buffer.clear()`, ADR-010) clears all stores AND re-arms warmup so a
   fresh sink is captured under the new weights. `rearm_sink()` re-arms sink
   capture at the mission-completed seam (`recapture_on_mission=True` default) —
   the sink is a per-mission anchor, not a stale boot snapshot.
4. **Corrupted-history drift training on the RSSM feasibility vehicle.**
   `RSSM.train_sequence_corrupted` rolls a random open-loop prior prefix (the
   model's own drifted imagination, `no_grad`, private `torch.Generator` for the
   prefix draw) then trains the posterior suffix to recover, via a shared
   `_posterior_step` helper extracted from `train_sequence` (behavior pinned by
   the golden suites + a forced-k=0 allclose-equality test). The external,
   evaluation-only `DriftCorrectionHead` predicts correction residuals and is
   consumed by `measure_drift` (never deployed; no parameters on the RSSM
   `state_dict`). Drift metric: deterministic seeded per-modality open-loop MSE
   with range as the headline channel, zero-filled channels excluded,
   `valid_mask` threaded, plus latent divergence — this robot has no pose
   channel, so "pose error" is substituted and declared.
5. **Distillation stays a scripts-only spike.** Deterministic prior-mean k-step
   teacher → MLP jump student via the existing
   `KnowledgeDistiller(objective="regression")`; the report separates
   primitive-level speedup from the MCTS consumer-level ceiling (~1.25-1.6× —
   tree expansion requires intermediate states) and the Jetson criterion stays
   pending an operator run. No production adoption under this ADR.

### Alternatives rejected

- **Extend `observe_step`/`imagine_step` with a context argument** — breaks the
  protocol implemented by three engines (torch RSSM, DualStreamRSSM, ONNX
  composite) and both RSSM golden regression suites; the ONNX-traceable path
  cannot carry a Python-side memory object.
- **Off-loop-only memory** (consolidation-style) — a memory that never touches
  the carried `(h, z)` cannot satisfy "sink incorporated into the prediction".
- **Duplicating `train_sequence`'s per-step ops** in the corrupted path — ~50
  lines that must stay allclose forever; the golden tests pin numerical
  behavior, not source text, so a shared private helper is strictly safer.

## Consequences

- The deployed `DualStreamRSSM` does NOT receive the drift-training objective in
  this change; the port is explicitly deferred. Any S2-style drift-improvement
  claim applies to the RSSM feasibility vehicle only.
- The memory is engine-agnostic (operates on carried tensors,
  `h_dim = hidden_dim + cfc_hidden_dim`) and interacts with ADR-010: an OTA swap
  with `reset_state_on_swap=True` also resets the memory; with `False`, the
  operator has accepted stale state and the memory inherits that decision.
- Blended `h` re-enters the memory after one `observe_step`; the stored-state
  invariant is "posterior-corrected (observation-re-anchored) states only". A
  rate-limited `latent_context_blend` debug event exposes ‖c − hz‖ + blend
  cosine for runaway-feedback observability.
- The orchestrator blend is outside the existing observe-step budget test's
  measured path; `tests/performance/test_latent_context_latency.py` owns that
  budget.
