# Design — CI/docs honesty

## D-1. Relabel, do not twin

Production already has e2e / integration / smoke through `build_orchestrator`.
Parked files uniquely prove APIs the production class does not have. ADR-016
rejected deleting the parked class. A "fix" that silently points the parked
tiers at production is a reviewed choice, pinned by AST import AQA.

## D-2. Gate the docs, not just edit them

F-016's size budget was pytest-pinned; CI still ran advisory `check_doc`.
`--strict` in `ci.sh` + local-gates makes the budget a real gate. The
LANDED-row rule is a separate pytest pin because `check_doc` is size-only.

## D-3. Two F-008 namespaces

Catalog F-008 is USB-C rover smoke (`features.yaml`, `select_next.py`).
Smoke-report F-008 is a 2026-05-12 telemetry finding. ADR-013. Named once.
