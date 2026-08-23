# Tasks — AutonomousOrchestrator disposition

Quality gate for every task below, run before it is ticked:

```
python -m ruff check src/ tests/ tools/ && python -m ruff format --check src/ tests/ tools/
python -m mypy src/ --strict --ignore-missing-imports
bash scripts/validations/F-031.sh
```

Task ordering is binding: each task lands green before the next starts.

**Phase 1 — Record the decision**

- [x] 1.1 `docs/architecture/ADR-016-autonomous-orchestrator-disposition.md`,
      following `docs/architecture/adr/TEMPLATE.md`.
- [x] 1.2 Row in `docs/architecture/adr-log.md` after ADR-015.
- [x] 1.3 `orchestrator/CLAUDE.md`'s forward reference updated to the real filename.

**Phase 2 — Enforce it**

- [x] 2.1 Extend `tests/regression/test_import_graph_freeze.py` with a third,
      file-level parked-subsystem case for `mousedroid.orchestrator.autonomous`.
- [x] 2.2 Prove the pin can fail: inject a throwaway module-scope import of
      `AutonomousOrchestrator` into `orchestrator.py`, confirm red and that the
      failure names `orchestrator/orchestrator.py`, remove, confirm green and
      `git diff --stat` empty.

**Phase 3 — Catalog**

- [x] 3.1 F-031 entry in `features.yaml`.
- [x] 3.2 `scripts/validations/F-031.sh`.
- [x] 3.3 Register in `openspec/project.md`.

## Explicitly deferred (separate changes, do not fold in)

- Making `AutonomousOrchestrator` production-eligible (off-loop LLM dispatch,
  a real safety monitor) — out of scope by design; would need its own ADR
  superseding ADR-016, not a follow-up to this change.
- `features.yaml`'s `implemented_in` pin — set to `null` while this lands on
  the working branch; pinned to the real merge SHA in a follow-up closeout
  commit once the PR is known, per the established convention (see F-030's
  own closeout, which this change's own review caught as a prior gap).
