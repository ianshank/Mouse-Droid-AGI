# Proposal — BDI portfolio trainer honesty

- change_id: mouse-droid-bdi-portfolio
- project: mouse-droid
- status: active
- feature_id: F-041
- epic: Cognitive
- owner: ianshank
- created: 2026-09-07
- rev: A

## Why

Belief AE used plain SGD and still plateaued at predict-zero after He init.
`collect_annotations` stored vision as X while `label_intention` reads action,
range, battery, and Law 1/2 flags — the 10-class claim had no observation
signal. This is portfolio quality, not rover-blocking.

## What Changes

- Adam on the BDI numpy trainers.
- Causal `intention_features` aligned with `label_intention`.
- Publish bars: AE MSE < PCA-128 and intention acc > majority. No HF upload
  until both pass. OTA poller remains `ianshank/mousedroid-policy-v2`.

## Impact

Runtime `WEIGHT_INIT_SCALE` and `NeuralBDI` load paths unchanged. Missing
weights still construct `NeuralBDI()` random. BDI stays on the ~1 Hz slow loop.

## Charter

No CHARTER §3 carve-out: training laptop only; no 30 Hz LLM; no new actuation.

## Validation

`bash scripts/validations/F-041.sh`
