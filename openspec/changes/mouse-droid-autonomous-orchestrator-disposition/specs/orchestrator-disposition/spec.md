# Spec delta — Orchestrator disposition

## ADDED Requirements

### Requirement: AutonomousOrchestrator SHALL NOT be imported at module scope by any active production module

Only `factory.py::build_autonomous_orchestrator`, referencing it via a
function-scoped (lazy) import, SHALL depend on
`mousedroid.orchestrator.autonomous` at runtime import time. No other module
under `src/mousedroid/` — including the production `orchestrator.py` — SHALL
carry a module-scope import of it.

#### Scenario: the import-graph freeze test enforces the disposition

- **GIVEN** the module-scope import graph of every file under `src/mousedroid/`
- **WHEN** `tests/regression/test_import_graph_freeze.py` walks it
- **THEN** no file other than `orchestrator/autonomous.py` itself carries a
  module-scope import of `mousedroid.orchestrator.autonomous`

#### Scenario: a future module-scope import is caught before merge

- **GIVEN** a hypothetical edit adds `from mousedroid.orchestrator.autonomous
  import AutonomousOrchestrator` at the top of `orchestrator.py`
- **WHEN** the regression suite runs
- **THEN** `test_no_active_module_imports_autonomous_orchestrator_at_module_scope`
  fails and names `orchestrator/orchestrator.py` as the offending file

### Requirement: the disposition SHALL be recorded in an accepted ADR

The reasoning for keeping `AutonomousOrchestrator` off the production path —
its LLM call inside the 30 Hz hot loop and absence of safety-monitor
machinery — SHALL be recorded in an ADR, not left as an undocumented
convention.

#### Scenario: the module map points at a real document

- **GIVEN** `src/mousedroid/orchestrator/CLAUDE.md`, the surface contract read
  before touching the orchestrator subsystem
- **WHEN** it references the disposition ADR
- **THEN** that reference resolves to
  `docs/architecture/ADR-016-autonomous-orchestrator-disposition.md`, a real,
  `Accepted` document — not a forward reference to a change that hasn't landed
