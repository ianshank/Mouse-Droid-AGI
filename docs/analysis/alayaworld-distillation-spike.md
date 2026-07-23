# AlayaWorld Distillation Spike — k-Step Imagination → Jump Student (F-023)

**Status:** SPIKE — non-binding, time-boxed. No production adoption under this
document (ADR-015; CHARTER §5 claims discipline: pillars are promoted only
when a concrete need and a gate exist).
**Harness:** `scripts/spike_step_distillation.py` (scripts-only, non-production).
**Claims discipline:** the AlayaWorld report describes distilling a video
diffusion model from ~30 to 4 steps per chunk. That architecture was NOT
adopted; the honest analog evaluated here is compressing k composed RSSM
`imagine_step` calls into one forward of a compact student. Results are
internal measurements on this repo's RSSM — **no iWorld-Bench-equivalent
evaluation is claimed and no parity with AlayaWorld is implied.**

## Method

- **Teacher** — deterministic prior-MEAN k-step composition (no sampling),
  predicting `(h_k, z_k, γ-discounted k-step return)` with γ from
  `MCTSConfig.gamma`, matching `MCTSPlanner._rollout`'s discounted
  accumulator. (A stochastic teacher has an irreducible MSE floor and does
  not agree with itself across seeds.) Paramless adapter — the distiller
  freeze loop never touches the RSSM.
- **Student** — 2-layer MLP over `[h | z | a_1..a_k]` (width
  `--student-hidden`), distilled with the existing growth-pillar
  `KnowledgeDistiller(objective="regression", hard_labels=None)`.
- **Accuracy** — held-out MSE on `(h_k, z_k, return)` vs teacher, plus
  **action agreement**: argmax over a `n_action_candidates` grid of the
  predicted k-step discounted return, teacher vs student.
- **Latency** — k sequential *stochastic* `imagine_step` calls (the deployed
  primitive) vs one student forward; `latency_stats.summarize` p50/p95/p99.

### Primitive vs consumer level (the honesty split)

The measurable quantity here is PRIMITIVE-level speedup. The deployment
consumer is `MCTSPlanner.plan()`, which makes **~500-650 `imagine_step`
calls** at defaults; depth-5 rollouts are only **~40%** of them, and tree
expansion requires intermediate `(h, z)` states a jump student cannot provide.
**End-to-end planner gain therefore caps at ~1.25-1.6×** regardless of the
primitive numbers below. Full MCTS integration (replacing the rollout leg) is
out of scope for this spike.

## Results — in-container CPU run (WS5)

Run (2026-07-23, container CPU, seed 42, RANDOM-INIT RSSM):
`python scripts/spike_step_distillation.py --k 2,4,8 --distill-steps 200 --trials 200 --seed 42 --out reports/spike_step_distillation.json`
(Latency is an architecture property and transfers approximately; accuracy
numbers understate a trained-checkpoint teacher — see the Jetson procedure.)

| k | action agreement | held-out MSE (hz / return) | primitive p95 (ms) | student p95 (ms) | primitive speedup (p95) |
|---|---|---|---|---|---|
| 2 | 0.422 | 0.00309 / 0.00451 | 0.399 | 0.055 | 7.29× |
| 4 | 0.609 | 0.00122 / 0.01274 | 0.766 | 0.058 | 13.17× |
| 8 | 0.422 | 0.00047 / 0.01418 | 1.883 | 0.060 | 31.54× |

## Results — Jetson Orin Nano (operator run — PENDING)

**The Jetson criterion is UNMET until this section is filled** by an operator
following `docs/runbooks/jetson-alayaworld-spike.md`. Container-CPU numbers
are architecture-relative; absolute latencies and the CPU/GPU balance differ
on device.

| k | action agreement | primitive p95 (ms) | student p95 (ms) | primitive speedup (p95) |
|---|---|---|---|---|
| — | — | — | — | — |

## Go / No-Go recommendation

Decision rubric (all three required for GO):

1. **Primitive**: ≥3× p95 primitive speedup at the chosen k **on Jetson**.
2. **Accuracy**: ≥0.90 action agreement at that k (trained-checkpoint teacher).
3. **Consumer case**: a written plan for the MCTS rollout-leg integration
   whose projected end-to-end gain (bounded by the ~1.25-1.6× ceiling)
   justifies the added model + maintenance — or an alternative consumer
   (e.g. long-horizon imagination for the deliberative tier) with its own
   budget case.

**Recommendation (provisional): DEFER** — pending the Jetson +
trained-checkpoint run. Rationale from the container-CPU numbers:

- **Latency criterion: provisionally strong.** Primitive p95 speedups of
  7.3× (k=2) to 31.5× (k=8) far exceed the ≥3× bar, and the speedup scales
  with k exactly as the one-forward architecture predicts. This is expected
  to transfer approximately to the Jetson (architecture-relative), but the
  absolute on-device numbers remain the gate.
- **Accuracy criterion: NOT met.** Action agreement peaks at 0.609 (k=4) —
  well under the 0.90 bar. Two caveats keep this from being a REJECT: the
  teacher is a random-init RSSM (near-flat reward surfaces make the argmax
  grid nearly a coin toss across similar candidates), and 200 distill steps
  on a 128-wide student is a deliberately small budget. The Jetson run must
  use a trained checkpoint before this criterion is judged.
- **Consumer case: unresolved.** Even with a perfect student, the MCTS
  end-to-end ceiling is ~1.25-1.6×; no alternative consumer (e.g. deliberative
  long-horizon imagination) currently exists on the roadmap with a budget
  case. This alone justifies DEFER over ADOPT regardless of accuracy.

**Final call** (ADOPT / DEFER / REJECT) is made after the operator fills the
Jetson section above; any adoption would be a NEW F-number and a separate
soak-gated decision.
