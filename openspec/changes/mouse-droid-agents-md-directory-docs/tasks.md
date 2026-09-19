# Tasks: `mouse-droid-agents-md-directory-docs` (F-053)

Task ordering is binding: each lands green before the next starts. Deviations from task wording are
recorded inline — declared, not silent.

Nothing here is started. Phase 1 is deliberately the wiring, not the content — see `design.md` D-3.

---

## Phase 0 — Prove the premise before acting on it

The whole plan rests on one claim: an `AGENTS.md` in this repository is not read. Establish it by
observation, not by quoting documentation.

- [ ] 0.1 Record the live evidence: the repo has a 403-line root `AGENTS.md` and **zero `@` imports
  in any of the nine `CLAUDE.md` files**, so that file is already dead text today. This is the
  premise; if it is wrong, the rest of the plan is wrong.
- [ ] 0.2 Confirm in a real session which instruction files load — the interactive session prints a
  line naming what it read. Capture it. If Claude is reading `AGENTS.md` despite the root
  `CLAUDE.md`, stop: D-1 through D-3 need rewriting, not implementing.
- [ ] 0.3 Record the Claude Code version in the evidence note. Native `AGENTS.md` reading needs
  v2.1.277+; the import path this plan uses does not, and that difference is why it was chosen.
- [ ] 0.4 Confirm `claudeMdExcludes` is unset in every settings layer this repo controls — an
  exclude glob could suppress a file this plan adds, and it merges across layers.

## Phase 1 — The import seam, and the test that proves it

Before any folder description exists.

- [ ] 1.0 Add `"**/AGENTS.md"` to `pyproject.toml:244`'s wheel `exclude`, and a test asserting all
  three agent-facing patterns are present. Today the list is `["**/CLAUDE.md", "**/agent.md"]`, so
  every `AGENTS.md` this change adds under `src/mousedroid/` would ship to PyPI. This is first because
  it is the one task whose omission does harm outside the repository.
- [ ] 1.1 Add `@AGENTS.md` as the first line of the root `CLAUDE.md`. Bare, not backticked — `@path`
  parsing skips code spans, so `` `@AGENTS.md` `` imports nothing.
- [ ] 1.2 Add `@AGENTS.md` to each of the eight nested `CLAUDE.md` files (`arm`, `growth`,
  `hardware`, `learning`, `llm_gateway`, `orchestrator`, `telemetry`, `world_model`). Relative
  imports resolve against the importing file, so the bare name is right. These eight matter most:
  without the import, a sibling `AGENTS.md` is shadowed even under nested discovery.
- [ ] 1.3 Write `tests/regression/test_f053_aqa.py::test_every_agents_md_is_actually_loaded` — for
  every directory containing an `AGENTS.md`, if it also contains a `CLAUDE.md`, that `CLAUDE.md`
  must import it. Match the **bare** `@AGENTS.md` form only; a backticked mention must fail the
  test, because a backticked mention is not an import.
- [ ] 1.4 Prove 1.3 fails before 1.1/1.2 land, per the `prove-pin-fails` skill. A gate that was
  green before the fix is not a gate.
- [ ] 1.5 Assert no import chain exceeds four hops (the documented maximum) and that no import
  resolves outside the working directory — an external import triggers an approval dialog that would
  silently disable it for anyone who declines.
- [ ] 1.6 Confirm the root `CLAUDE.md` stays inside `docs.core_max_lines: 250`
  (`.claude/workforce.yaml:72`). An import adds one line to the importing file, so the 403-line
  `AGENTS.md` does not count against it — verify rather than assume, because
  `tools/claude_hooks/docs_trimmer.py` is what fails the `local-gates` job.

## Phase 2 — The scope gate, so the set cannot rot

Scope is the nine directories of D-4, not 45 and not 147.

- [ ] 2.1 Pin the in-scope set as an enumerated frozenset: the eight directories holding a
  `CLAUDE.md` (`arm`, `growth`, `hardware`, `learning`, `llm_gateway`, `orchestrator`, `telemetry`,
  `world_model`) plus `tests/`. Following `_GATED_FACTORY_FILES`
  (`tests/regression/test_f042_aqa.py:14-20`) — never a directory prefix, which silently absorbs every
  future package.
- [ ] 2.2 Add `test_f053_aqa.py::test_the_agents_md_set_is_exactly_the_pinned_set` — both directions.
  A missing file fails; an `AGENTS.md` outside the set also fails, so the 147-directory sprawl D-4
  rejects cannot creep back one commit at a time.
- [ ] 2.3 Handle `src/mousedroid/agents/` explicitly: it is scanned by the skill loader
  (`harness_mcp.py:394` globs `*.md` there). Either keep it out of the set or assert its file carries
  no YAML front matter, so a folder description cannot become a registered skill.
- [ ] 2.4 Assert zero `agent.md` files remain anywhere (Phase 5 removes them; this pins it shut).
- [ ] 2.5 Assert the root `AGENTS.md` does not grow. It is 23,125 bytes against the 20,000-byte
  `doc_hygiene.py` budget already, and widening that gate is deferred because files like it would fail.

## Phase 2b — The generated package map: every folder, zero agent tokens

This is what answers "describing the functions of each folder" for all 41 packages.

- [ ] 2b.1 Write `scripts/generate_package_map.py`: walk `src/mousedroid/*/`, `ast`-parse each module's
  imports, and emit `docs/architecture/package-map.md` with one section per package — purpose line,
  what it imports, what imports it, and a mermaid dependency subgraph.
- [ ] 2b.2 Derive the purpose line from the package's `__init__.py` docstring, and fail the generator
  on a package whose `__init__.py` has none. That turns "document the folder" into a docstring the
  linter and the map both read, rather than prose in a third place.
- [ ] 2b.3 Add `test_f053_aqa.py::test_the_package_map_is_current` — regenerate and diff. A stale map
  fails, which is the property 41 hand-written files could never have.
- [ ] 2b.4 Wire the generator into `scripts/ci.sh` and a `Makefile` target so it is reproducible, and
  confirm the output is deterministic (sorted, no timestamps) or the diff test will flap.
- [ ] 2b.5 Confirm `docs/architecture/package-map.md` is inside `doc_hygiene.py`'s 20 KB budget, or
  split it per-epic. 41 sections is the size risk in this change.

## Phase 3 — Author the nine files, by subagent fan-out

Briefs are disprove-shaped. "Verify this is accurate" returns "accurate".

- [ ] 3.1 Write `.claude/skills/agents-md-authoring/SKILL.md` **first**, so the nine files and the
  generator's purpose lines share one procedure: derive purpose from imports and Protocols,
  never from filenames; exactly one diagram of the D-5 shape; point at contracts, never restate them.
- [ ] 3.2 Fan out authoring of the nine files to `doc-reconciler` subagents. Each brief carries the target
  directory's `CLAUDE.md` contract inline — a subagent that skips project instructions does not load
  it, so it cannot be assumed present.
- [ ] 3.3 Each file states: what the folder is for, what enters and leaves, which packages it speaks
  to through Protocols, and a pointer to its `CLAUDE.md` where one exists. No `Key Files` list — that
  is the section that rotted in all 16 `agent.md` files.
- [ ] 3.4 Run a `peer-reviewer` pass per batch with the brief "find one statement the tree
  contradicts, or report clean with the evidence that makes it clean."
- [ ] 3.5 Run `config-guardian` over the batch: no threshold, port, path or dimension restated from
  a Pydantic schema into prose, where it would drift from the schema silently.
- [ ] 3.6 Add `test_f053_aqa.py::test_no_agents_md_restates_an_invariant` — a statement matching a
  numbered invariant in the same directory's `CLAUDE.md` fails. This is D-2's rule made mechanical.

## Phase 4 — Mermaid, gated

- [ ] 4.0 Assert the D-9 line cap: every `AGENTS.md` is 40 lines or fewer, diagram included. This
  gate exists because the measured effect of these files is small or negative and the cost is paid on
  every invocation — see `proposal.md` §7.
- [ ] 4.1 Assert every `AGENTS.md` contains exactly one `mermaid` fence.
- [ ] 4.2 Assert every fence parses and renders. A diagram that does not render is a broken diagram,
  and this repo has no existing mermaid gate — `docs/architecture/` carries ~45 diagrams that nothing
  validates, so this gate is new capability, not a port.
- [ ] 4.3 Assert every node label naming a repo path or dotted symbol resolves on disk, extending the
  rule `tools/validate_skill_commands.py` applies to skills.
- [ ] 4.4 Assert the diagram's shape: a `flowchart`, naming at least one package other than its own —
  a diagram with no edges leaving the folder is describing nothing.
- [ ] 4.5 Decide and record whether the existing `docs/architecture/` diagrams join the parse gate.
  Recommend yes as a follow-up, not here: ~45 unvalidated diagrams is a real gap but a separate one,
  and folding it in would make this change's failure surface someone else's backlog.

## Phase 5 — Migrate the 16 `agent.md`, split rather than deleted

- [ ] 5.0 Record that this executes an existing decision rather than inventing one:
  `docs/planning/TECH_DEBT_REMEDIATION_PLAN.md` already rules `agent.md` **Delete/merge**, having
  flagged it as drift that was never resolved. Update that table in the same commit — a decision
  reversed or duplicated in a second document is how the four-way invariant split arose.
- [ ] 5.1 For each of the 16, split the content: folder-purpose half into that directory's `AGENTS.md`
  where one exists in the nine, otherwise into the package's `__init__.py` docstring, which 2b.2 makes
  the source for the generated map; persona half held for 5.3.
- [ ] 5.1a Fix, do not migrate, `src/mousedroid/llm_gateway/agent.md:6` — "velocity commands via local
  LLM" is false in both halves (`protocol.py:65` returns a `GoalVector`;
  `src/mousedroid/config/schema/llm.py:96` includes the cloud `anthropic` backend). Carrying it forward would launder
  a wrong statement into a new file.
- [ ] 5.2 Update the three in-repo references before removing anything —
  `tests/regression/test_doc_reconciliation_aqa.py:46`,
  `tests/regression/test_ci_gate_wiring_aqa.py:825`, `src/mousedroid/skills/loaders.py:7,103`. A
  blind delete turns a green suite red.
- [ ] 5.3 Evaluate each persona against the seven existing `.claude/agents/` definitions and promote
  only those that earn one. Fourteen near-identical "you are the X architect" stubs do not each
  warrant a subagent. Any promotion obeys the existing contract: `max_lines: 60`, required
  frontmatter, bare tool names (`tests/regression/test_claude_workforce_aqa.py:182-257`).
- [ ] 5.4 Remove the 16 `agent.md` files. 2.4 pins them gone.
- [ ] 5.5 Record in `CHANGELOG.md` that a filename changed, with the reason: `agent.md` is not the
  standard name and was read by nothing.

## Phase 6 — Workforce wiring

- [ ] 6.1 Index the new skill in `SKILLS.md` — required by
  `tests/regression/test_claude_workforce_aqa.py:290`, so an unindexed skill fails CI.
- [ ] 6.2 Run `python tools/validate_skill_commands.py`; every backticked path in the new SKILL.md
  must resolve.
- [ ] 6.3 Add the `AGENTS.md` layer to `docs/claude/surfaces/README.md` so the surface index describes
  the convention that now exists.
- [ ] 6.4 Reconcile the root `AGENTS.md` (403 lines) against the root `CLAUDE.md` (114 lines): with
  1.1 landed both load together, so a contradiction between them becomes a live instruction conflict
  rather than a dormant one. Fix contradictions found; a full audit of that file is out of scope
  (D-8).

## Phase 7 — Documentation reconciliation

- [ ] 7.1 `CHANGELOG.md` entry under `[Unreleased]` for the convention and the rename.
- [ ] 7.2 Root `CLAUDE.md` Surface Map gains the `AGENTS.md` layer, stating the division of labour
  from D-2 in one line so the next reader does not have to infer it.
- [ ] 7.3 `NEXT_STEPS.md`: record the two follow-ups this change declines —
  `docs/architecture/` diagram validation (4.5) and `.claude/rules/` path-scoped migration (D-8).
- [ ] 7.4 Check `docs/architecture/c4-claude-workforce.md` — it diagrams the workforce surfaces and
  will be stale once a per-package `AGENTS.md` layer exists.

## Phase 8 — Validation and closeout

- [ ] 8.1 `make gates` passes.
- [ ] 8.2 `make test` passes (all four pytest steps).
- [ ] 8.3 `python -m tools.ratchet_budgets --strict` exits 0; the three budgets unchanged. This change
  is documentation and tests, so it should need no suppression at all — if it does, that is a signal
  the approach is wrong, not that the budget should move.
- [ ] 8.4 `python scripts/validate.py --tier fast` passes.
- [ ] 8.5 `bash scripts/ci.sh` passes.
- [ ] 8.6 Write `scripts/validations/F-053.sh` and register F-053 in `features.yaml` with
  `status: in_progress`, `implemented_in: null`, `depends_on: ["F-030"]` (F-030 is the doc-reconciliation
  feature this extends).
- [ ] 8.7 Add the regression pair: `tests/regression/test_f053_aqa.py` +
  `test_f053_backwards_compat.py`. The backwards-compat half asserts the nine `CLAUDE.md` files still
  load and their invariants are unchanged — the risk of this change is weakening a contract by moving
  prose around it.
- [ ] 8.8 Register the change in `openspec/project.md`; flip to `implemented` with the trunk SHA only
  after squash-merge.
