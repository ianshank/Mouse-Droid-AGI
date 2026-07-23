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

> Filled by WS5:
> `python scripts/spike_step_distillation.py --k 2,4,8 --distill-steps 200 --trials 200 --seed 42 --out reports/spike_step_distillation.json`
> Note: run on a RANDOM-INIT RSSM (methodology + latency are architecture
> properties; accuracy numbers sharpen with a trained checkpoint via
> `--checkpoint`).

| k | action agreement | held-out MSE (hz / return) | primitive p95 (ms) | student p95 (ms) | primitive speedup (p95) |
|---|---|---|---|---|---|
| 2 | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD | TBD |

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

**Recommendation:** TBD — one of:
- **ADOPT** (criteria met on Jetson; open a follow-up F-number for the
  production integration, which remains a separate soak-gated decision),
- **DEFER** (primitive case proven but consumer case not yet justified —
  revisit when a long-horizon imagination consumer lands), or
- **REJECT** (accuracy or Jetson latency fails; document numbers and close).

**Provisional (container-CPU) reading:** TBD by WS5 — expected shape from the
smoke run: large primitive speedups (5-15×) but sub-0.90 agreement at small
distill budgets on a random-init model, i.e. the latency story is promising
and the accuracy story is the open question for the Jetson + trained-checkpoint
run.
