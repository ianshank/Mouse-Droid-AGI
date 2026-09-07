# Tasks — CI/docs honesty

Quality gate:

```
python -m ruff check tests/regression/test_f038_aqa.py tests/regression/test_f038_backwards_compat.py tests/regression/test_next_steps_reconciled.py tests/functional tests/user_journey tools/doc_hygiene.py
bash scripts/validations/F-038.sh
```

**Phase 1 — Labels**

- [x] 1.1 Relabel ci.yml step, ci.sh echo, Makefile behaviour, skill, module docs.
- [x] 1.2 AQA: parked dirs import `build_autonomous_orchestrator` only.

**Phase 2 — Docs pin**

- [x] 2.1 Strip LANDED from Current Next Steps; leftover-label done F-ids.
- [x] 2.2 CHARTER §5 → root NEXT_STEPS.md; M6 vs LoRA.
- [x] 2.3 Catalog vs smoke-report F-008 named once.

**Phase 3 — Gate**

- [x] 3.1 `doc_hygiene.py --strict` in ci.sh and local-gates.
- [x] 3.2 `scripts/validations/F-038.sh` + `features.yaml` F-038.
