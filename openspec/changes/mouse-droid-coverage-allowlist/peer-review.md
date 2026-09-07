# Peer review — Enumerated branch-coverage allowlist

## Verdict table

| Claim | Verdict |
|---|---|
| Unbounded factory/ prefix exempts future modules | **CONFIRMED** — `new_builder.py` is now gated |
| `orchestrator/_` accidentally matched `__init__.py` | **CONFIRMED** — `__init__.py` is not in `_ALLOWED_FILES` |
| Algorithmic factory is not pure DI | **CONFIRMED** — three files stay on the gate |
| Schema/telemetry/validation prefixes can stay | **CONFIRMED** — still 1-file-to-many split products |

## Load-bearing pins

1. `_ALLOWED_DIR_PREFIXES` has four entries, none under factory or orchestrator.
2. `test_every_factory_module_is_enumerated_or_gated`.
3. `test_evaluate_branch_coverage_gates_algorithmic_factory_modules`.
