# Tasks — BDI portfolio trainer

Quality gate:

```
python -m ruff check training/train_bdi.py training/collect_annotations.py training/training_utils.py tests/regression/test_f041_aqa.py
bash scripts/validations/F-041.sh
```

**Phase 1 — Optimiser**

- [x] 1.1 Adam on belief AE; He init pinned; runtime `WEIGHT_INIT_SCALE` untouched.

**Phase 2 — Causal labels**

- [x] 2.1 `intention_features` match `label_intention` inputs.
- [x] 2.2 Causal 10→32 ReLU→10 MLP on those features beats majority class.

**Phase 3 — Publish honesty**

- [x] 3.1 `passes_bdi_publish_bars`; no upload from `train_bdi`; OTA repo pin.
