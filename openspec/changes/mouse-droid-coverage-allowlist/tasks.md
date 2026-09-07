# Tasks — Enumerated branch-coverage allowlist

Quality gate:

```
python -m ruff check scripts/check_branch_coverage.py tests/regression/test_f042_aqa.py tests/regression/test_f042_backwards_compat.py tests/unit/scripts/test_check_branch_coverage_base_ref.py
bash scripts/validations/F-042.sh
```

**Phase 1 — Enumerate**

- [x] 1.1 Drop factory/ and orchestrator/_ from `_ALLOWED_DIR_PREFIXES`.
- [x] 1.2 `_ALLOWED_FILES` lists DI factory modules + orchestrator mixins.

**Phase 2 — Keep gating**

- [x] 2.1 `on_device_learning.py`, `mcp_harness.py`, `_replay_batch_helpers.py` fail at 50%.
- [x] 2.2 Hypothetical `factory/new_builder.py` is not exempt.

**Phase 3 — Pins**

- [x] 3.1 Disk classification AQA + remaining-prefix backwards-compat.
- [x] 3.2 `scripts/validations/F-042.sh` + `features.yaml` F-042.
