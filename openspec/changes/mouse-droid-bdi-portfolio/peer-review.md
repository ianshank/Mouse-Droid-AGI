# Peer review — BDI portfolio trainer

## Verdict table

| Claim | Verdict |
|---|---|
| Plain SGD was the remaining AE gap after He init | **CONFIRMED** — Adam wired |
| Vision X cannot carry 10-class labels | **CONFIRMED** — causal feature row |
| ReLU AE beating PCA on isotropic Gaussian is not guaranteed | **CONFIRMED** — publish bar is `passes_bdi_publish_bars` on the operator set; unit test pins mean-predictor gap |
| Runtime BDI still random if weights missing | **CONFIRMED** |

## Load-bearing pins

1. `_init_weights` uses `sqrt(2 / fan_in)`.
2. `INTENTION_FEATURE_NAMES` length 10.
3. `WeightUpdatePollConfig.policy_repo_id` default `ianshank/mousedroid-policy-v2`.
