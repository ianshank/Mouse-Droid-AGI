# AlayaWorld Memory + Drift Adaptation — Implementation Plan (F-023)

**Goal:** Land the bounded-context latent memory, corrupted-history
drift-reduction training, and the distillation feasibility spike as default-OFF,
fully additive workstreams.
**Architecture:** see `docs/superpowers/specs/2026-07-23-alayaworld-memory-distill-design.md` + ADR-015.
**Tech stack:** existing — torch, pydantic v2, structlog, pytest/hypothesis.

## Context — why

The OpenSpec change `mouse-droid-alayaworld-memory-distill` (archived under
`openspec/changes/`) adapts AlayaWorld's memory-management and drift-training
patterns to the RSSM latent world model. The world model currently compresses all
history into the recurrent `(h, z)`; nothing anchors long-horizon context, and
training never exposes the model to its own drifted rollouts.

## Non-negotiable invariants

- Default-OFF everywhere; disabled paths byte-identical (charter §6, invariant #9).
- 30 Hz hot loop stays deterministic + training-free (invariant #10) — the blend
  is pure `no_grad` tensor math; all training is offline.
- `train_sequence` behavior identical (golden suites + k=0 equality pin the
  shared-helper refactor); RSSM `state_dict` keys unchanged.
- Unhealthy (non-finite) ticks never touch the memory (NaN contract).
- No iWorld-Bench-equivalence claims; deviations (no benchmark suite, no Jetson
  in-loop) declared, never silent.

## Workstream / task breakdown

### WS1 — spec artifacts + config skeleton (`docs(spec): … [F-023]`)
- [x] OpenSpec archive (`openspec/project.md`, `changes/<id>/{proposal,tasks,design}.md`, spec deltas)
- [x] Design spec, this plan, ADR-015, `docs/related-work.md`
- [x] `WorldModelMemoryConfig` (Settings-level) + `DriftTrainingConfig` (`training.drift`)
- [x] `features.yaml` F-023; NEXT_STEPS/progress entries
- [x] `tests/regression/test_alayaworld_memory_distill_backwards_compat.py`
**Files:** `src/mousedroid/config/schema.py`, `features.yaml`, docs above.

### WS2 — bounded-context latent memory (`feat(world-model): … [F-023]`)
- [x] `LatentContextProtocol` in `world_model/protocol.py`; `BoundedContextMemory` in `world_model/bounded_context.py`
- [x] `build_latent_context` in `factory.py`; orchestrator wiring (`healthy` flag, observe→blend, OTA `reset()`, mission `rearm_sink()`)
- [x] Tests: unit (boundedness/cold-start/NaN/identity/determinism/lifecycle), factory, orchestrator tick (incl. S1b sink-incorporation + disabled-path trajectory equality), perf, AQA regression
**Files:** `src/mousedroid/world_model/{bounded_context.py,protocol.py,__init__.py}`, `src/mousedroid/factory.py`, `src/mousedroid/orchestrator/orchestrator.py`, `tests/…`.

### WS3 — drift training + metric (`feat(training): … [F-023]`)
- [x] `rssm.py`: `_posterior_step` refactor, `train_sequence_corrupted`, `DriftCorrectionHead`
- [x] `training/drift_metrics.py` (`measure_drift`), `training/drift_reduction.py` (`train_pair_and_compare`)
- [x] `rssm_pretrainer.py` optional `drift=` param; `scripts/compare_drift.py`; analysis-doc template
- [x] Tests: k=0 equality + RNG pins, drift metrics determinism/honesty, pair-compare, script integration; golden suites + on-device suite re-run
**Files:** `src/mousedroid/world_model/rssm.py`, `src/mousedroid/training/{drift_metrics.py,drift_reduction.py,rssm_pretrainer.py}`, `scripts/compare_drift.py`, `docs/analysis/alayaworld-drift-comparison.md`, `tests/…`.

### WS4 — distillation spike (`feat(spike): … [F-023]`)
- [x] `scripts/spike_step_distillation.py` (prior-mean k-step teacher, jump student, `KnowledgeDistiller` regression objective, latency + agreement eval)
- [x] Spike report template (primitive vs consumer ceiling; Jetson pending); `docs/runbooks/jetson-alayaworld-spike.md`
- [x] `tests/integration/test_spike_step_distillation.py`
**Files:** as listed.

### WS5 — results + closure (`docs(feature): … [F-023]`)
- [x] Run `compare_drift.py` + spike script (scaled-down, seeded) in-container; fill both analysis docs (negative result documented if so)
- [x] CLAUDE.md section (AQA-safe wording), NEXT_STEPS/progress/CHANGELOG, F-023 finalization, design-spec review section, openspec `tasks.md` ticks
- [x] Full CI gate; push; draft PR
**Files:** `CLAUDE.md`, `NEXT_STEPS.md`, `progress.md`, `CHANGELOG.md`, `features.yaml`, analysis docs.

## Operator follow-ups (prepared, not executed here)

- Jetson spike run per `docs/runbooks/jetson-alayaworld-spike.md` → paste numbers
  into the spike report → final go/no-go.
- On-rover drift evaluation once real replay data accumulates.
