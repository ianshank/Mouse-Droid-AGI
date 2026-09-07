# Design — BDI portfolio trainer

## D-1. Adam, He init stays training-local

`AdamOptimizer` in `training/training_utils.py` mutates the same arrays the
forward pass uses. `_init_weights` remains `sqrt(2 / fan_in)` and must not
import `WEIGHT_INIT_SCALE`.

## D-2. Causal X for intention

`intention_feature_vector` packs the scalars `label_intention` reads.
Vision stays in `observations` for the belief AE. `train_intention_predictor(..., features=...)` is the 10-class claim; the 64→10 desire head is load-compat only.

## D-3. Publish bars, not Hub

`passes_bdi_publish_bars` is the gate. `train_bdi` must not call `upload_weights`.
OTA poller default is `ianshank/mousedroid-policy-v2`, not `mousedroid-weights`.
