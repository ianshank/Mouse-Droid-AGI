# ADR-014 — Cyclomatic-Complexity Gate + Enterprise-Hardening Refactor

* **Status:** Accepted
* **Date:** 2026-07-05
* **Owners:** Ian Cruickshank
* **Scope:** `pyproject.toml` (`[tool.ruff.lint]` `C901` + `[tool.ruff.lint.mccabe]`),
  `src/mousedroid/telemetry/metrics.py` (`render_prometheus`),
  `src/mousedroid/telemetry/server.py` (`_handle_mission_post`,
  `_broadcast_loop`), `src/mousedroid/orchestrator/orchestrator.py`
  (`start`, `stop`), `tests/regression/test_render_prometheus_golden.py`,
  the new `tests/unit/{growth,meta,scaling,efficiency,logging}/` suites.

## Context

An "enterprise production standards" refactoring pass was requested against a
codebase that already satisfies most of that bar: single-source Pydantic-v2
config, `structlog` everywhere, `mypy --strict`, zero bare excepts, and a dual
85 % coverage gate (total line + per-changed-file branch). An empirical audit
(three exploration passes + an adversarial peer review that ran `ruff` against
the tree) found the genuine gaps were narrow:

1. **No cyclomatic-complexity gate.** The prompt's "< 15 per function"
   requirement was unenforced. `ruff`'s `C4` rule already in the select set is
   *comprehensions* (flake8-comprehensions), NOT complexity — a common
   confusion. There was no `C901` / McCabe ceiling.
2. **Five functions exceeded a max-complexity of 15**, all pre-existing:
   `telemetry/metrics.py::render_prometheus` (cc 55 — a 624-line method),
   `telemetry/server.py::_handle_mission_post` (20) and `_broadcast_loop` (19),
   and `orchestrator/orchestrator.py::start` (19) + `stop` (18). Three
   procedural-glue `scripts/` files were also over.
3. **A handful of leaf modules had no dedicated unit-test directory**
   (`growth`, `meta`, `scaling`, `efficiency`, `logging`).

Crucially, length ≠ complexity: the peer review's first draft targeted
`Orchestrator.__init__` (cc 4) and `Orchestrator.tick` (cc 9) because they are
*long*, but neither is flagged at a ceiling of 15. Measuring with the pinned
`ruff==0.8.0` corrected the target list before any code moved.

## Decision

1. **Enforce `C901` at `max-complexity = 15`** (the prompt's threshold) across
   `src/`, `tests/`, and `scripts/`. Deliberately 15, not 10: at 10 the `src/`
   backlog balloons from 5 to ~23 functions. The ceiling can tighten later.

2. **Baseline pre-existing offenders via file-level `per-file-ignores`, not
   inline `# noqa`.** File-level keeps the `src/` suppression budget
   (`tests/regression/test_suppression_budget.py`) flat, and each entry is a
   *ratchet backlog* removed in the same change that decomposes its function.
   The `src/` baseline is now **empty** — every offender was decomposed. Only
   the `scripts/**` glob remains baselined (operational glue, already exempt
   from `D`/`T20`).

3. **Decompose the five `src/` offenders by extraction, preserving behaviour:**
   * `render_prometheus` → thirteen `_families_*` helpers returning
     `list[list[str]]`, called in the identical emit order. Chosen over a
     data-driven descriptor table because the 55 emit blocks are heterogeneous
     (8 predicate kinds, 8 render primitives, name transforms, multi-family
     groups) — a table would need a leaky schema and risk an exact-order
     reshape. **Byte-identical exposition is now pinned by a golden
     characterization test** (the prior backwards-compat test only asserted
     `name not in out`, and would not have caught a reshape).
   * `_handle_mission_post` → extract `_parse_mission_request` and
     `_dispatch_mission_command`. `_broadcast_loop` → extract
     `_push_frame_metrics`. `start`/`stop` → extract cloud-subsystem and
     background-task phase helpers.

4. **Explicitly leave `Orchestrator.tick` untouched.** It is within budget
   (cc 9) and its 148 lines are dominated by documented ordering invariants
   (projection-before-swap, latch-clear-before-export, emergency-branch local
   mutation). The 30 Hz / 33 ms budget dwarfs Python call overhead — the risk of
   extraction there is *correctness*, not latency, so it is not worth taking.

5. **Backfill dedicated unit suites** for the five thin modules (all now at
   96–100 % line coverage), including a **characterization** test that documents
   `MAMLAdapter.meta_step`'s current first-order behaviour rather than asserting
   textbook second-order MAML (see Rejected Alternatives).

## Rejected alternatives

* **A shared factory `_build_optional_driver` helper.** The plan scoped this to
  the "uniform" optional-hardware builders. On inspection only `build_microphone`
  and `build_speaker` share the `enabled → mock → real` shape, and they *diverge*
  in logging (microphone logs nothing; speaker emits three events with distinct
  fields). A helper over two shallow, divergent, already-well-tested call sites
  would be a net-negative "leaky kwargs" abstraction. **Not done** — the DRY
  principle does not justify a worse design.
* **Fixing MAML to true second-order.** `meta_step` optimizes deep-copied
  adapted params, so gradients never reach the base model. Making it
  second-order is a real behaviour change — risky, unrequested, and the meta/sim
  surface is soak-frozen. Characterized, not changed.
* **Creating `config/defaults.py` / `development.py` / `production.py` +
  lowering coverage to 70 %** (literal prompt deliverables). These conflict with
  Architecture Invariant #3 (single-source Pydantic + YAML) and would lower the
  existing 85 % gate. Mapped onto the existing `config/*.yaml` overlays instead
  (see `docs/refactor/enterprise-hardening-notes.md`).

## Consequences

* Complexity is now enforced repo-wide, including the safety-critical hot-loop
  file — a new complex function fails CI's existing `ruff check` stages rather
  than landing silently. `build_orchestrator` sits at exactly cc 15; a future
  edit that adds one branch there must budget for a decomposition, not a gate
  workaround.
* No behaviour changed: the Prometheus exposition is byte-identical, and all
  telemetry/orchestrator/mission tests pass unchanged.
* No breaking changes → no migration guide required.
