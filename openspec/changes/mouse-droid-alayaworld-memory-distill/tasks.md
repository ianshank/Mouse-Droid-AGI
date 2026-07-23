# Tasks — mouse-droid-alayaworld-memory-distill

Status ticks are updated as workstreams land on
`claude/mouse-droid-alayaworld-adapt-dt07mp`. Deviations from the original task
wording are recorded inline — they are declared, not silent.

- [x] 4.1 Read full AlayaWorld PDF sections on bounded context design and
      corrupted-history training in detail.
      **Deviation:** arXiv:2607.18367 is unreachable from the implementation
      environment (egress policy). The techniques are adapted from the change
      request's characterization only; `docs/related-work.md` (task 4.10's
      deliverable, which subsumes 4.1's output — one artifact, not two) flags the
      citation as not independently verified.
- [x] 4.2 Document MouseDroid's current world-model state representation and
      existing memory scheme (baseline).
      → "Baseline" section of
      `docs/superpowers/specs/2026-07-23-alayaworld-memory-distill-design.md`.
- [x] 4.3 Design bounded-context memory manager interface adapted to droid state
      representation (recurrent latent `(h, z)`, not video frames).
      → `LatentContextProtocol` (design spec §D1 + ADR-015).
- [x] 4.4 Implement memory manager as an ablation-switchable module (old vs new
      memory scheme). → `src/mousedroid/world_model/bounded_context.py`,
      default-OFF `world_model_memory` config block, identity when disabled.
- [x] 4.5 Implement corrupted-history rollout generator for training data
      augmentation. → `RSSM.train_sequence_corrupted` open-loop prior prefix
      (the corruption source is the model's own imagination, inlined rather than
      a standalone generator).
- [x] 4.6 Implement residual-prediction training objective.
      **Declared interpretation:** drift reduction in the trained weights comes
      from the corrupted-prefix recovery objective (scheduled-sampling style);
      the literal residual predictor is the external, evaluation-only
      `DriftCorrectionHead` consumed by `measure_drift` — trained, measured,
      never deployed. See `specs/world-model-training/spec.md`.
- [x] 4.7 Run comparison: baseline memory/training vs new memory/training on the
      existing navigation benchmark suite.
      **Deviation:** no navigation benchmark suite exists in this repository.
      Substituted: the deterministic seeded synthetic-episode drift harness
      (`scripts/compare_drift.py --synthetic`; MuJoCo episodes behind a flag),
      reporting per-modality open-loop drift with range as the headline channel.
      On-rover evaluation is prepared (runbook), not executed.
- [ ] 4.8 (in-container half DONE; Jetson half PENDING operator) Run
      distillation feasibility spike on target edge hardware (Jetson)
      with latency/accuracy measurement.
      **Deviation:** this environment has no Jetson. The spike script runs
      in-container (CPU) for methodology + provisional numbers; the on-device run
      is prepared via `docs/runbooks/jetson-alayaworld-spike.md` and the spike
      report marks the Jetson criterion as pending until an operator executes it.
- [x] 4.9 Write spike report with go/no-go recommendation.
      → `docs/analysis/alayaworld-distillation-spike.md` (provisional/conditional
      pending the Jetson run).
- [x] 4.10 Update `docs/related-work.md` with AlayaWorld citation, explicitly
      noting the architecture is video-diffusion-specific and only the
      memory/drift-training patterns were adapted. (File is new — the repo had no
      related-work doc.)
