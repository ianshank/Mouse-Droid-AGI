# Related Work

External research this project has deliberately adapted ideas from, with an
explicit record of what was adopted and what was not. Entries here follow the
project's claims discipline (`docs/CHARTER.md` §1): MouseDroid is an edge-AI /
robotics engineering project; adapting an idea from a paper is not a claim of
implementing, matching, or benchmarking against that paper's system.

## AlayaWorld — Interactive Long-Horizon World Modeling

- **Citation:** arXiv:2607.18367, "AlayaWorld: Interactive Long-Horizon World
  Modeling — Full Technical Report". *Cited as characterized in the OpenSpec
  change request `mouse-droid-alayaworld-memory-distill`
  (`openspec/changes/mouse-droid-alayaworld-memory-distill/`); the paper was
  unreachable from the implementation environment at authoring time and has not
  been independently verified.*
- **What it is (per the change request):** a 15B-parameter video diffusion
  transformer for interactive, persistent, long-horizon world modeling. Its
  relevant mechanisms: a bounded visual context (persistent sink frame +
  compressed history + geometry-aligned spatial memory), corrupted-history
  training to reduce autoregressive drift, and autoregressive distillation
  cutting inference from ~30 to 4 steps per chunk.
- **What MouseDroid adapted (F-023, ADR-015):**
  - The **bounded-context memory pattern** — reinterpreted for a recurrent
    latent `(h, z)` state as a per-mission frozen sink anchor + recent ring +
    EMA long-summary, attention-retrieved and λ-blended at the orchestrator
    observe seam (`world_model/bounded_context.py`). Default-OFF,
    ablation-switchable.
  - The **corrupted-history training strategy** — reinterpreted as
    `RSSM.train_sequence_corrupted`: a random open-loop prior prefix (the
    model's own drifted imagination) followed by a posterior recovery suffix,
    plus an evaluation-only residual-correction head.
- **What MouseDroid did NOT adopt:** the video diffusion transformer
  architecture, the 15B scale, frame-level attention/spatial memory over pixels,
  and the paper's evaluation suite. **No iWorld-Bench-equivalent evaluation is
  claimed**; drift results in `docs/analysis/alayaworld-drift-comparison.md` are
  internal synthetic-episode metrics on this repo's RSSM, not benchmark scores
  and not a parity or comparison claim against AlayaWorld.
- **Distillation:** evaluated only as a non-binding feasibility spike
  (`docs/analysis/alayaworld-distillation-spike.md`) mapping "fewer diffusion
  steps" to "k composed `imagine_step` calls → one student forward" for the MCTS
  planner; no production adoption.
