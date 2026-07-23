# Runbook — AlayaWorld Distillation Spike on the Jetson (F-023)

Operator procedure for producing the on-device numbers that
`docs/analysis/alayaworld-distillation-spike.md` needs before its go/no-go
recommendation can be finalised. The spike is **non-binding and
non-production** — nothing here arms motors, touches the 30 Hz loop, or
deploys a model.

## Prerequisites

- The rover's container is running the branch/image that carries
  `scripts/spike_step_distillation.py` (F-023).
- No exclusive-device access is needed (pure compute); it is safe to run
  alongside the orchestrator, but prefer a quiet rover so latency numbers are
  not skewed by a busy CPU/iGPU (the world model owns the shared iGPU — see
  `docs/runbooks/jetson-claude-pilot-deploy.md`).
- Optional but recommended: a trained RSSM checkpoint (accuracy numbers on a
  random-init model understate agreement).

## Procedure

1. Enter the container:

   ```bash
   docker exec -it "${MOUSEDROID_CONTAINER:-mousedroid}" bash
   ```

2. Run the spike (seeded; ~minutes on CPU). With a trained checkpoint if one
   is available under the weights dir:

   ```bash
   python scripts/spike_step_distillation.py \
     --config config/jetson_production.yaml \
     --k 2,4,8 --distill-steps 200 --trials 200 --seed 42 \
     --checkpoint weights/rssm_pretrained.pt \
     --out reports/spike_step_distillation_jetson.json
   ```

   Drop `--checkpoint` if no trained RSSM is present (note the caveat in the
   report), and lower `--distill-steps`/`--trials` if the rover is thermally
   constrained.

3. Copy the printed markdown table (and the JSON under `reports/`) into the
   **Jetson** section of `docs/analysis/alayaworld-distillation-spike.md`.

## Decision rubric (mirror of the analysis doc)

GO requires all three: (1) ≥3× p95 primitive speedup at the chosen k on this
hardware; (2) ≥0.90 action agreement with a trained-checkpoint teacher;
(3) a justified consumer case — remembering the MCTS end-to-end ceiling is
~1.25-1.6× because rollouts are only ~40% of `plan()`'s imagine calls.
Anything less: record DEFER or REJECT with the numbers. Either way the
outcome goes in the analysis doc and `NEXT_STEPS.md` item 9 is updated —
adoption (if any) is a NEW F-number and a separate soak-gated decision.

## Troubleshooting

- `ModuleNotFoundError: mousedroid` — run from the repo root inside the
  container (the script inserts `src/` itself).
- Thermal throttling skews p95/p99: re-run with the rover idle and confirm
  with `tegrastats` that the CPU is not pegged by the orchestrator.
- Grep events: `distiller_init` (objective=regression) confirms the
  growth-pillar distiller path is in use.
