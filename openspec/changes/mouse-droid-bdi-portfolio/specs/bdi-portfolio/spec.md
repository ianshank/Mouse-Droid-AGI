# BDI portfolio trainer

## Purpose

Make BDI training honest: Adam on the belief AE, causal features for
intention, no Hub publish until bars are beaten.

## Requirements

### Requirement: He init SHALL stay training-local

`WEIGHT_INIT_SCALE` SHALL remain 0.01 at runtime.

### Requirement: intention X SHALL match `label_intention`

The npz `intention_features` matrix SHALL contain the scalars the
heuristic reads. Vision-only X SHALL not be the 10-class claim.

### Requirement: publish SHALL stay blocked until bars pass

`passes_bdi_publish_bars` SHALL require belief MSE below PCA-128 and
intention accuracy above majority class.
