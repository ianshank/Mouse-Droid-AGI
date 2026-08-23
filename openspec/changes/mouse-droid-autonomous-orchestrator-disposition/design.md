# Design — AutonomousOrchestrator disposition

## D-1. Why a regression pin, not just an ADR

An ADR alone records intent but not code. `.claude/skills/prove-pin-fails/SKILL.md`'s
own lesson, paid for five review rounds on an earlier PR, is that a pin which
cannot fail is worse than no pin. Recording "AutonomousOrchestrator is parked"
in prose only, with nothing enforcing it, is exactly that shape of debt —
a future refactor could promote it to a module-scope import and nothing would
notice until it shipped.

## D-2. Why a file-level exemption, not directory-level

The two existing cases in `test_import_graph_freeze.py` (`arm/`,
`hardware/sensors/` ultrasonic drivers) exempt an entire *directory*, because
each parked subsystem lives in its own directory and no production code
shares that directory. `autonomous.py` does not have that luxury: it sits in
`orchestrator/`, the same directory as the production `orchestrator.py`. A
directory-level exemption for `orchestrator/` would also license
`orchestrator.py` itself to import `autonomous.py` at module scope — silently
defeating the pin's purpose. The new case exempts exactly
`orchestrator/autonomous.py`, nothing broader.

## D-3. Why the pin is proven by injection, not `prove_pin_fails.sh`

`AutonomousOrchestrator` has always been function-scoped-only — there is no
historical commit where the class was imported at module scope to revert to.
Per the prove-pin-fails skill's own documented exception for catalog-wide
invariants with no bad historical state, this is proven by temporarily
injecting a throwaway module-scope import into `orchestrator.py`, confirming
the new test goes red and names the offending file, removing the injection,
and confirming green and byte-identical (`git diff --stat` empty).

## D-4. Scope boundary — this is a disposition record, not a redesign

Making `AutonomousOrchestrator` production-eligible (moving the LLM call
off-loop, wiring in `SafetyMonitorProtocol`) is explicitly out of scope here.
CHARTER §3 draws that line deliberately; retrofitting it would be a
substantial, separate architectural change needing its own design and review.
If that work is ever undertaken, it supersedes this ADR with a new one — ADRs
are immutable once accepted, per `docs/architecture/adr-log.md`'s own header.
