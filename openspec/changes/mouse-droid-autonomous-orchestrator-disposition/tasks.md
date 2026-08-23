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

**Phase 4 — Copilot review round on PR #204, all confirmed real**

- [x] 4.1 `_module_scope_imports` had two blind spots, both verified by reading
      its exact AST-walking logic before fixing: (a) it skipped ALL relative
      imports (`node.level == 0` filter) on the assumption "relative imports
      can only reach the importer's own package, which is always allowed" —
      true for `arm/`/`hardware/sensors/` (directory-exempted) but false here,
      since `autonomous.py` shares a directory with the restricted
      `orchestrator.py`; (b) it only recorded bare `node.module` for
      `ImportFrom`, missing `from mousedroid.orchestrator import autonomous`
      (the submodule imported as a name). Fixed by resolving relative imports
      via each file's own containing-package path and additionally recording
      `f"{module}.{alias}"` per alias — additive and safe for the two existing
      cases, whose exemption is directory-based and applied before any
      per-import check runs.
- [x] 4.2 Proved both fixes: injected `from .autonomous import
      AutonomousOrchestrator` into `orchestrator.py` (relative form) and
      separately `from mousedroid.orchestrator import autonomous`
      (parent-import-submodule form); both turned the test red with the
      correct offending file named; both restored byte-identical, confirmed
      green.
- [x] 4.3 The import-scope pin alone doesn't enforce ADR-016's actual Decision
      ("main.py routes exclusively through `build_orchestrator`") — a
      production entrypoint could call `factory.py::build_autonomous_orchestrator`
      (a public function) directly without importing `autonomous.py` at
      module scope anywhere. Added
      `test_no_production_entrypoint_calls_the_autonomous_orchestrator_builder`,
      a direct source-level check on `main.py`. Proved: injected an import of
      `build_autonomous_orchestrator` into `main.py`, confirmed red, restored,
      confirmed green.
- [x] 4.4 `scripts/validations/F-031.sh` ran the whole shared
      `test_import_graph_freeze.py` file, coupling its diagnosis to the
      unrelated pre-existing `arm`/HC-SR04 cases. Narrowed to the two
      F-031-owned test nodes by name.
- [x] 4.5 `features.yaml`'s F-031 `verification:` list claimed the ADR must be
      `Accepted` and the CLAUDE.md reference must resolve, but the script only
      checked file existence — a `Proposed` ADR or a stale reference would
      still pass. Added both checks explicitly; proved each by temporarily
      corrupting the ADR's status line and separately the CLAUDE.md
      reference, confirming red, restoring, confirming green.
- [x] 4.6 `charter-carveout/SKILL.md`'s decision procedure mandated
      `asyncio.to_thread` specifically for off-loop LLM dispatch — verified
      wrong by reading `anthropic_gateway.py` directly (module docstring:
      "Asyncio-only — uses `anthropic.AsyncAnthropic`. No blocking calls";
      the actual call is `await self._client.messages.create(...)`, native
      async I/O, never `to_thread`). Rephrased to describe the hot-loop
      boundary without prescribing one specific dispatch mechanism.
- [x] 4.7 `proposal.md`'s `status: in_progress` used the `features.yaml` enum
      spelling rather than the openspec house format's spaced convention
      (`status: in progress`, confirmed against
      `mouse-droid-doc-reconciliation/proposal.md:5`). Corrected.

**Phase 5 — Closeout**

- [x] 5.1 `features.yaml` flipped to `status: "done"`,
      `implemented_in: "5d40ee45ecb485eaffdec366fd1da238bbc7aa62"` (PR #204's
      merge SHA), matching the established convention — F-030's own closeout,
      which this change's Phase 4 review caught as a prior gap, is the reason
      this step is a dedicated task rather than assumed.
- [x] 5.2 `openspec/project.md` row updated to `implemented`.
- [x] 5.3 `proposal.md`'s `status:` updated to `implemented`.

## Explicitly deferred (separate changes, do not fold in)

- Making `AutonomousOrchestrator` production-eligible (off-loop LLM dispatch,
  a real safety monitor) — out of scope by design; would need its own ADR
  superseding ADR-016, not a follow-up to this change.
