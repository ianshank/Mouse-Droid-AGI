# OpenSpec Change: Adapt AlayaWorld Persistent Memory and Distillation Techniques for the MouseDroid World Model

> **Archival note.** This proposal is imported verbatim-in-substance from the
> requesting OpenSpec workflow. The repo-native artifacts (features.yaml `F-023`,
> `docs/superpowers/specs/2026-07-23-alayaworld-memory-distill-design.md`,
> `docs/architecture/ADR-015-bounded-context-latent-memory.md`) are authoritative;
> deviations discovered during implementation are recorded in `tasks.md` and the
> spec deltas under `specs/`. Corrected paths: the proposal's `src/world_model/…`
> corresponds to `src/mousedroid/world_model/…` in this repository.

- **change_id**: `mouse-droid-alayaworld-memory-distill`
- **project**: MouseDroid (repository `ianshank/Mouse-Droid-AGI`)
- **status**: in progress (implementation on branch `claude/mouse-droid-alayaworld-adapt-dt07mp`)
- **source_paper**: arXiv:2607.18367 — "AlayaWorld: Interactive Long-Horizon World Modeling — Full Technical Report" (as characterized in the change request; not independently verified — arXiv unreachable from the implementation environment)
- **tier**: Certain (product/goal-level match; techniques require adaptation from continuous video to discrete agentic representation)
- **owner**: TBD
- **created**: 2026-07-22

## Why

MouseDroid's goal is an agentic world model for embodied navigation/interaction.
AlayaWorld demonstrates a 15B video diffusion transformer for interactive,
persistent, long-horizon world modeling using a bounded visual context (persistent
sink frame + compressed history + geometry-aligned spatial memory),
corrupted-history training to reduce drift, and autoregressive distillation to cut
inference steps from ~30 to 4 per chunk. MouseDroid does not use continuous video
diffusion; it uses a discrete/lower-dimensional recurrent latent state appropriate
for an edge-deployed (Jetson Orin Nano) agent. The transferable ideas are the
memory-management pattern (bounded context via sink + compressed history) and the
drift-reduction training strategy (train on self-generated corrupted rollouts),
not the video diffusion architecture itself.

## What Changes

- ADD a bounded-context memory manager for the MouseDroid world model: persistent
  "sink" state + compressed rolling history, analogous to AlayaWorld's
  sink-frame/spatial-memory design but adapted to the droid's actual state
  representation (the recurrent `(h, z)` latent — NOT raw video frames).
- ADD a drift-reduction training procedure: augment training data with
  self-generated corrupted-history rollouts and train the model to predict
  correction residuals, adapted from AlayaWorld's approach.
- EVALUATE (spike, not commit) distillation feasibility for reducing inference
  steps/latency on Jetson-class hardware, given MouseDroid's existing model size
  and target FPS/latency budget.
- NO adoption of video diffusion transformer architecture. NO claim of
  iWorld-Bench-equivalent evaluation without an equivalent embodied benchmark.

## Impact

- **Affected specs**: `world-model-memory`, `world-model-training`
- **Affected code**: `src/mousedroid/world_model/bounded_context.py` (new),
  `src/mousedroid/training/drift_reduction.py` + `drift_metrics.py` (new),
  additive changes to `src/mousedroid/world_model/rssm.py`,
  `src/mousedroid/orchestrator/orchestrator.py`, `src/mousedroid/factory.py`,
  `src/mousedroid/config/schema.py`
- **Risk**: Medium — touches the world-model memory path; mitigated by running the
  new memory scheme strictly in parallel/ablation mode (default-OFF, identity when
  disabled) rather than replacing the current scheme.
- **Breaking changes**: None — the change is fully additive; existing YAML loads
  unchanged (backwards-compatibility invariant #9).

## Spec Deltas

See `specs/world-model-memory/spec.md` and `specs/world-model-training/spec.md`
for the ADDED requirements (with implementation-time notes on metric substitution
and the residual-objective interpretation).

## Tasks

See `tasks.md` (with explicit deviation notes for tasks 4.1, 4.7 and 4.8).

## Validation

- Ablation test shows bounded memory manager maintains constant memory footprint
  over long rollouts.
- Drift metric comparison shows measurable improvement (or a documented negative
  result) from corrupted-history training.
- Spike report delivered with explicit go/no-go recommendation before any
  production distillation work begins.
