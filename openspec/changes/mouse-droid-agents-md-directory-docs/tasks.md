# Tasks: `mouse-droid-agents-md-directory-docs` (F-053)

**Revision 3.** Re-scoped after review; see `peer-review.md`. Ordering is binding **within a phase**,
and each phase leaves the tree green — revision 2 declared ordering binding and then scheduled three
tasks that were red until a later phase, so the rule now says what it means.

**Phase 1 ships alone and is worth shipping alone.**

---

## Phase 1 — Make the root `AGENTS.md` load, and keep it out of the wheel

- [x] 1.1 Add `"**/AGENTS.md"` to `pyproject.toml:244`'s wheel `exclude`, plus a test asserting all
  three agent-facing patterns. Today it is `["**/CLAUDE.md", "**/agent.md"]`. First, because it is the
  only task whose omission does harm outside this repository.
- [x] 1.2 Write `tests/regression/test_f053_aqa.py::test_the_root_agents_md_is_imported` — the root
  `CLAUDE.md` must contain the bare `@AGENTS.md` form. A backticked mention must fail, because `@path`
  parsing skips code spans. **Land this test before 1.3**, so it has a genuine failing case; revision 2
  ordered the equivalent proof after the fix, where nothing could fail.
- [x] 1.3 Add `@AGENTS.md` as the first line of the root `CLAUDE.md`. 1.2 goes green.
- [x] 1.4 Fix the staleness this makes live, since both files now load together: root `AGENTS.md`
  points twice at `CLAUDE.md` sections that do not exist ("Test surface mirror", "Live deployment +
  CI-gate contracts") and says "any of the other five" subagents where `.claude/agents/` holds 7 and
  `SKILLS.md:802` says 7.
- [x] 1.5 Confirm the root `CLAUDE.md` stays inside `docs.core_max_lines: 250` (currently 114). This is
  the one doc budget that is actually enforced, by `tools/claude_hooks/docs_trimmer.py` in
  `local-gates`. Do **not** assert a byte size on `AGENTS.md`: `.gitattributes` is absent, so a CRLF
  checkout changes it, and `test-windows` runs `tests/regression`.
- [ ] **Stop here if the rest is declined.** Nothing below is required for this to be correct.

### Phase 1 landed — declared deviations

Phase 1 is complete and green. Four deviations from the task wording, declared rather than silent:

- **1.1 / 1.2 added a shared helper not in the plan.** The pattern roster and the `@path` parsing
  both live in `tests/_claude_md.py`, following the established `tests/_<name>.py` convention
  (`_bash.py`, `_pyproject.py`, `_script_loader.py`, `_jetson_hardware.py`) rather than being
  inlined in the regression file. The roster is imported by the test *and* referenced from the
  `pyproject.toml` comment, which is what makes "added in one place and forgotten in the other"
  detectable. Line cited in 1.1 moved 244 -> 249 because the fix added five comment lines above it.
- **1.2 grew a second test from a prove-pin-fails pass.** The wheel-behaviour test initially passed
  for the wrong reason: removing `"**/AGENTS.md"` from the exclude left it green, because no
  `AGENTS.md` exists under `src/` today, so the built wheel was identical either way. The docstring
  was narrowed to what it actually proves and `test_the_build_test_is_not_vacuous` was added. The
  original test as written would have shipped as decoration.
- **1.3 carries an explanatory HTML comment, not a bare line.** A future editor's most likely
  mistake is backticking the import for "consistency" with the surrounding prose, which silently
  imports nothing. The comment states that and names the pinning test. Nine lines against the
  250-line budget; `CLAUDE.md` is 123 of 250.
- **1.4 left one item deliberately unfixed.** `AGENTS.md` prescribes a `Co-Authored-By` trailer
  naming a different model than the one now writing commits. The import made that line live, so it
  is in scope by the letter of 1.4 — but it is the user's policy surface, not a docs defect, and
  rewriting a user's attribution policy under cover of a staleness sweep is not this change's call.
  Flagged for the user, unchanged. The two phantom section references and the subagent count were
  fixed as written.

Verification: the gate was proven failing first (2 failures). `make gates` green. `make regression`
1966 passed / 31 skipped. Wheel built empirically — 22 agent-facing files tracked under `src/`, zero
in the artifact.

## Phase 2 — Execute WS-8d: one format, every subsystem indexed

- [x] 2.1 Author a `CLAUDE.md` for the **9** subsystems that have only an `agent.md`: `agents`,
  `cognitive`, `comms`, `config`, `experience`, `logging`, `memory`, `safety`, `sensing`. Each carries
  the purpose blockquote at lines 3-4 and a `Key Files` section, matching the shape of the existing 8 so
  there is one format rather than two.
- [x] 2.2 Fan out authoring to `doc-reconciler` subagents, briefs disprove-shaped ("find a statement the
  tree contradicts"). Each brief carries the target package's `__init__.py` docstring and its real
  imports inline — a subagent that skips project instructions loads neither.
- [x] 2.3 Derive every purpose line from the package's actual Protocols and imports. **No line may
  restate a root invariant**: 15 of the 16 `agent.md` carry the same three boilerplate lines already in
  the root `CLAUDE.md`, and one of those is enforced by `ruff` `T20`. Repeating them is pure token cost.
- [x] 2.4 Add `test_f053_aqa.py::test_every_subsystem_with_in_package_docs_has_a_claude_md`, and assert
  the root Surface Map indexes each one — WS-8d's complaint is that 9 subsystems are invisible from the
  root surface, so indexing is half the deliverable.
- [x] 2.5 `peer-reviewer` pass per batch; `config-guardian` pass for thresholds or paths restated from a
  Pydantic schema into prose.
- [x] 2.6 Do **not** create any `AGENTS.md` outside the root. Assert it: an `AGENTS.md` anywhere under
  `src/` or `tests/` fails. D-1/D-3 — it would be unreachable and would duplicate a purpose statement
  that already exists.



### Phase 2 landed — declared deviations

Phase 2 is complete. Deviations from the task wording, declared rather than silent:

- **2.2 / 2.5 were not fanned out to `doc-reconciler` / `peer-reviewer` /
  `config-guardian` subagents.** Cloud Agents are blocked for this account
  (on-demand usage), and the operator asked for `gh` + local clone only. Authoring
  and review were done in-process by the implementing agent: each purpose line and
  Key Files entry was derived from the package `__init__.py` docstring, its
  `@runtime_checkable` Protocols, and the concrete symbols those imports name.
  No threshold or path was restated from a Pydantic schema into prose beyond naming
  the config classes the code already reads (`SafetyConfig`, `ThreeLawsConfig`,
  `ExperienceConfig`, `MemoryConfig`, `LoggingConfig`).
- **2.4 added shared helpers in `tests/_claude_md.py`.** Package discovery and
  Surface Map link detection live beside the Phase 1 import helpers rather than
  being inlined, matching the Phase 1 deviation pattern and the `tests/_<name>.py`
  convention. The roster is discovered via `git ls-files`, not hard-coded.
- **Sensing omits a ring-buffer invariant that would restate root invariant 8.**
  Task 2.3 forbids restating root invariants; `deque(maxlen=N)` is already in the
  root `CLAUDE.md`, so the sensing surface keeps fusion / protocol / human-presence
  rules only.
- **Root Surface Map gains all 9 new links in this phase** (not deferred to 6.1).
  Task 2.4 requires the indexing half of WS-8d with the new files; Phase 6.1's
  "indexes all 17" becomes a no-op confirmation once Phase 2 is green.

## Phase 3 — Gate the `Key Files` lists, by generalising a gate that works

- [ ] 3.1 Generalise `tests/regression/test_doc_reconciliation_aqa.py::test_orchestrator_claude_md_names_only_real_symbols`
  (`:191`) from one file to every nested `CLAUDE.md`: each named symbol resolves to a real `class`/`def`
  in its mapped file, and each named path exists.
- [ ] 3.2 Prove 3.1 fails before any fix lands — it should immediately flag
  `src/mousedroid/llm_gateway/CLAUDE.md:26` (`mock_gateway.py`, which does not exist; the real file is
  `fallback_gateway.py`) and `src/mousedroid/arm/CLAUDE.md:19` (`mock_arm.py`; the real path is
  `src/mousedroid/arm/hardware/mock_arm_driver.py`).
- [ ] 3.3 Fix the `llm_gateway` entry.
- [ ] 3.4 Record `src/mousedroid/arm/CLAUDE.md` as a **declared exemption** with F-008 as the reason:
  `src/mousedroid/arm/**` is denied by the `freeze_gate` PreToolUse hook
  (`.claude/workforce.yaml:23-24`) while `F-008` is `todo`. Enumerate it as a frozenset entry following
  `_GATED_FACTORY_FILES` (`tests/regression/test_f042_aqa.py:14-20`), never a prefix, and add a test
  asserting the exemption lapses when F-008 reaches `done`. Do **not** use
  `MOUSEDROID_WORKFORCE_ALLOW_FROZEN` — a stale filename is not an exceptional edit.
- [ ] 3.5 Extend 3.1 to config values named in prose. It should catch
  `src/mousedroid/llm_gateway/CLAUDE.md:20-21`, which requires `LLMConfig.fallback_backend` to target
  "`mock`, `ollama`" while `src/mousedroid/config/schema/llm.py:157` permits only `none`, `llama_cpp`,
  `openai_compatible` — **neither named value is legal**. Revision 1 found this and revision 2 dropped
  it; the gate is what stops that happening again.

## Phase 4 — Retire the 16 `agent.md`

- [ ] 4.1 Record that this executes WS-8d (`docs/planning/TECH_DEBT_REMEDIATION_PLAN.md:1492-1500`),
  and update that table. Cite WS-8d, not the root-docs row — the root row rules on the root `agent.md`;
  WS-8d rules on the per-directory pair, which is what this change touches.
- [ ] 4.2 Merge each folder-purpose half into the sibling `CLAUDE.md` (new for the 9, existing for the
  5 that have both, plus root).
- [ ] 4.3 **Fix, do not migrate,** `src/mousedroid/llm_gateway/agent.md:6`: "velocity commands via local
  LLM" is false in both halves (`protocol.py:65` returns a `GoalVector`;
  `src/mousedroid/config/schema/llm.py:96` includes the cloud `anthropic` backend).
- [ ] 4.4 Update the **one** reference that actually breaks a delete:
  `tests/regression/test_doc_reconciliation_aqa.py:46` lists `tests/agent.md` in `_SRC_COVERAGE_DOCS`
  and `read_text()`s it at `:130`. The other two commonly cited references are docstring prose —
  `src/mousedroid/skills/loaders.py:7,103` names no file, and
  `tests/regression/test_ci_gate_wiring_aqa.py:825` sits in a docstring stating the roster now comes
  from `git ls-files`.
- [ ] 4.5 Evaluate each persona against the seven existing `.claude/agents/` definitions; promote only
  those that earn one, under the existing contract.
- [ ] 4.6 Remove the 16 `agent.md`; assert zero remain. **This is also where `pyproject.toml`'s
  `"**/agent.md"` exclude becomes dead** — leave it, and say why in the commit: it costs nothing and
  removing it would let a reintroduced `agent.md` ship to PyPI.

## Phase 5 — The generated package map: all 41 folders, with the diagrams

- [ ] 5.1 Write `scripts/generate_package_map.py`: walk `src/mousedroid/*/`, `ast`-parse imports,
  **filter `TYPE_CHECKING` blocks** (three live cases in `world_model` alone), and emit one section per
  package — purpose line, imports, dependents, mermaid subgraph.
- [ ] 5.2 Label it an **import map**, not a dataflow map. In a factory-first DI codebase these diverge
  by design: `factory` has 35 outbound package edges and `config` 33 inbound because invariants 1-2
  require it, and the runtime seams run through injected Protocols `ast` cannot see. State that in the
  generated header so no reader mistakes it for architecture.
- [ ] 5.3 Source each purpose line from the package's `__init__.py` docstring, failing closed on a
  package without one. 40 of 41 have one; `src/mousedroid/telemetry/__init__.py` is **0 bytes**, so
  write it here. Note this makes Phase 5 a `src/` edit subject to `mypy --strict`, ruff docstring rules
  and the coverage floor — it is not a docs-only phase.
- [ ] 5.4 Split the output **per epic** from the start: 41 packages and 221 directed edges estimate to
  ~34 KB, so a single file would be a fork in the deliverable rather than a decision.
- [ ] 5.5 Add a regenerate-and-diff test that **normalises line endings and path separators** before
  comparing, so it holds on `test-windows`. Assert determinism: sorted output, no timestamps, no
  absolute paths.
- [ ] 5.6 Unit-test the generator against a fixture package tree; correctness comes from those tests,
  not from rendering. No renderer is added — none exists anywhere in the toolchain.
- [ ] 5.7 Wire into `scripts/ci.sh` and a `Makefile` target.

## Phase 6 — Wiring and documentation

- [ ] 6.1 Root `CLAUDE.md` Surface Map indexes all 17 in-package `CLAUDE.md`, and
  `docs/claude/surfaces/README.md` gains the package map.
- [ ] 6.2 `CHANGELOG.md` entry: WS-8d executed, one format, `agent.md` retired and why, root
  `AGENTS.md` now loads.
- [ ] 6.3 `NEXT_STEPS.md`: record what this declines — validating the 49 existing `docs/` fences plus
  the 1 in root `README.md`, a `.claude/rules/` migration, a full root `AGENTS.md` audit, and the `arm`
  `Key Files` fix pending F-008.
- [ ] 6.4 Check `docs/architecture/c4-claude-workforce.md`; it diagrams the workforce surfaces and goes
  stale once 17 `CLAUDE.md` are indexed.
- [ ] 6.5 If a skill is written for this procedure, index it in `SKILLS.md` —
  `tests/regression/test_claude_workforce_aqa.py:290` fails an unindexed skill. `.gitignore:10` already
  negates `.claude/skills/`, so no negation task is needed.

## Phase 7 — Validation and closeout

- [ ] 7.1 `make gates`, `make test`, `bash scripts/ci.sh`, `python scripts/validate.py --tier fast` all
  pass.
- [ ] 7.2 `python -m tools.ratchet_budgets --strict` exits 0, budgets unchanged. Phase 5 touches `src/`,
  so this is a real check rather than a formality.
- [ ] 7.3 Write `scripts/validations/F-053.sh`; register F-053 in `features.yaml` with
  `status: in_progress`, `implemented_in: null`, `depends_on: ["F-030"]`.
- [ ] 7.4 Add the regression pair. The backwards-compat half asserts all 17 `CLAUDE.md` still load and
  the 8 existing contracts' invariants are unchanged — the risk of this change is weakening a contract
  while moving prose around it.
- [ ] 7.5 `openspec/project.md` already carries the registry row (landed in `20cdd63`); update its text
  to match revision 3 rather than adding a second row.
