# Tasks: `mouse-droid-agents-md-directory-docs` (F-053)

**Revision 2.** Re-scoped after review found three load-bearing claims in revision 1 wrong; see
`peer-review.md`. Task ordering is binding: each lands green before the next starts. Deviations from
task wording are recorded inline — declared, not silent.

**Phase 1 ships alone and is worth shipping alone.** If the rest of this plan is declined, Phase 1
should still land: it is one import line plus one exclude pattern, and it makes 403 lines of existing
instruction load for the first time.

---

## Phase 1 — Make the existing `AGENTS.md` load, and stop the wheel shipping it

- [ ] 1.1 Add `"**/AGENTS.md"` to `pyproject.toml:244`'s wheel `exclude`, plus a test asserting all
  three agent-facing patterns are present. Today the list is `["**/CLAUDE.md", "**/agent.md"]`. First,
  because it is the only task whose omission does harm outside this repository.
- [ ] 1.2 Add `@AGENTS.md` as the first line of the root `CLAUDE.md`. Bare, not backticked — `@path`
  parsing skips code spans, so `` `@AGENTS.md` `` imports nothing.
- [ ] 1.3 Fix the known staleness that this makes live. Phase 1 means root `AGENTS.md` and root
  `CLAUDE.md` load together, so contradictions stop being dormant: `AGENTS.md` points twice at
  `CLAUDE.md` sections that do not exist ("Test surface mirror", "Live deployment + CI-gate
  contracts"), and says "any of the other five" subagents where `.claude/agents/` holds 7 and
  `SKILLS.md` says 7.
- [ ] 1.4 Assert the root `AGENTS.md` does not grow. 23,125 bytes against `doc_hygiene.py`'s 20,000;
  widening that gate is deferred because files like it would fail. Shrinking is fine.
- [ ] 1.5 Confirm the root `CLAUDE.md` stays inside `docs.core_max_lines: 250`
  (`.claude/workforce.yaml:72`; currently 114). An import adds one line to the importing file —
  verify rather than assume, because `tools/claude_hooks/docs_trimmer.py` is what fails `local-gates`.
- [ ] 1.6 Assert no import chain exceeds four hops (the documented maximum) and none resolves outside
  the working directory — an external import triggers an approval dialog that silently disables it for
  anyone who declines.
- [ ] **Stop here if the rest is declined.** Everything above is independent of the nine files and the
  generator.

## Phase 2 — The nine, and the reachability gate that is the point of this plan

- [ ] 2.1 Add `@AGENTS.md` to each of the eight nested `CLAUDE.md` files (`arm`, `growth`, `hardware`,
  `learning`, `llm_gateway`, `orchestrator`, `telemetry`, `world_model`). Relative imports resolve
  against the importing file, so the bare name is correct.
- [ ] 2.2 Write `tests/regression/test_f053_aqa.py::test_every_agents_md_has_a_load_path` — for every
  `AGENTS.md` in the tree, a `CLAUDE.md` in the **same directory** must import it with the bare
  `@AGENTS.md` form. A backticked mention must fail the test, because a backticked mention is not an
  import. This is the gate the whole plan exists for: with a root `CLAUDE.md` present, nested
  `AGENTS.md` auto-discovery is off, so an unimported `AGENTS.md` is unreadable by anything.
- [ ] 2.3 Prove 2.2 fails before 2.1 lands, per the `prove-pin-fails` skill. A gate that was green
  before the fix is not a gate.
- [ ] 2.4 Pin the in-scope set as an enumerated frozenset, following `_GATED_FACTORY_FILES`
  (`tests/regression/test_f042_aqa.py:14-20`) — never a directory prefix, which silently absorbs every
  future package. Assert both directions: a missing file fails, and an `AGENTS.md` outside the set
  fails.
- [ ] 2.5 Assert `src/mousedroid/agents/` is not in the set. The skill loader globs `*.md` there
  (`harness_mcp.py:394`), so an `AGENTS.md` with YAML front matter would register as a skill.
- [ ] 2.6 **Decide `tests/` explicitly, and record which way.** It is high-traffic and has no
  `CLAUDE.md`, so `tests/AGENTS.md` has no load path. Either add a one-line `tests/CLAUDE.md`
  containing `@AGENTS.md` (making it nine) or drop `tests/` (making it eight). Revision 1 had it in the
  set with no importer, which 2.2 would have caught — do not let that stand as an implicit choice.
- [ ] 2.7 Assert zero `agent.md` files remain (Phase 4 removes them; this pins it shut).

## Phase 3 — Author the nine, by subagent fan-out

- [ ] 3.1 Write `.claude/skills/agents-md-authoring/SKILL.md` first, so the nine files and the
  generator's purpose lines share one procedure: derive purpose from imports and Protocols, never from
  filenames; ≤30 lines; point at contracts, never restate them; no diagram (the generator owns those).
- [ ] 3.2 Fan out authoring to `doc-reconciler` subagents. Each brief carries the target directory's
  `CLAUDE.md` contract inline — a subagent that skips project instructions does not load it, so it
  cannot be assumed present.
- [ ] 3.3 Each file states: what the folder is for, what enters and leaves, which packages it speaks to
  through Protocols, and a pointer to its `CLAUDE.md`. **No `Key Files` list** — that is the section
  that rotted in all 16 `agent.md` files.
- [ ] 3.4 Assert the 30-line cap (D-9).
- [ ] 3.5 `peer-reviewer` pass per batch, brief: "find one statement the tree contradicts, or report
  clean with the evidence that makes it clean."
- [ ] 3.6 `config-guardian` pass: no threshold, port, path or dimension restated from a Pydantic schema
  into prose, where it would drift silently.
- [ ] 3.7 The no-restated-invariant rule is a **review item in the skill, not a gate**. Revision 1
  proposed a substring check; paraphrase defeats it, and a gate everyone learns to route around is
  worse than a checklist line. Recorded as a deliberate downgrade.

## Phase 4 — Migrate the 16 `agent.md`, split rather than deleted

- [ ] 4.1 Record that this executes an existing decision:
  `docs/planning/TECH_DEBT_REMEDIATION_PLAN.md` already rules `agent.md` **Delete/merge**, having
  flagged it as unresolved drift. Update that table in the same commit — a decision reversed or
  duplicated in a second document is how the four-way invariant split arose.
- [ ] 4.2 Split each: folder-purpose half into the sibling `AGENTS.md` where one of the nine exists,
  otherwise into the package's `__init__.py` docstring, which Phase 5's generator reads.
- [ ] 4.3 **Fix, do not migrate,** `src/mousedroid/llm_gateway/agent.md:6` — "velocity commands via
  local LLM" is false in both halves (`src/mousedroid/llm_gateway/protocol.py:65` returns a
  `GoalVector`; `src/mousedroid/config/schema/llm.py:96` includes the cloud `anthropic` backend).
  Carrying it forward would launder a wrong statement into a new file.
- [ ] 4.4 Update the three live references before removing anything —
  `tests/regression/test_doc_reconciliation_aqa.py:46`,
  `tests/regression/test_ci_gate_wiring_aqa.py:825`, `src/mousedroid/skills/loaders.py:7,103`. A blind
  delete turns a green suite red.
- [ ] 4.5 Evaluate each persona against the seven existing `.claude/agents/` definitions and promote
  only those that earn one, under the existing contract (`max_lines: 60`, required frontmatter, bare
  tool names — `tests/regression/test_claude_workforce_aqa.py:182-257`).
- [ ] 4.6 Remove the 16 `agent.md`. Task 2.7 pins them gone.

## Phase 5 — The generated package map: all 41 folders, and every diagram

This is what answers "describing the functions of each folder" for every folder, and it is where the
mermaid lives.

- [ ] 5.1 Write `scripts/generate_package_map.py`: walk `src/mousedroid/*/`, `ast`-parse each module's
  imports, and emit `docs/architecture/package-map.md` — one section per package with a purpose line,
  what it imports, what imports it, and a mermaid dependency subgraph **generated from the parsed
  graph**.
- [ ] 5.2 Source each purpose line from the package's `__init__.py` docstring, failing closed on a
  package without one. 40 of 41 already have one; only `src/mousedroid/telemetry` lacks it, so fix that
  docstring here. This turns "document the folder" into a docstring the linter and the map both read,
  rather than prose in a third place.
- [ ] 5.3 Add `test_f053_aqa.py::test_the_package_map_is_current` — regenerate and diff. A stale map
  fails, which is the property 41 hand-written files could never have.
- [ ] 5.4 Unit-test the generator against a fixture package tree, so diagram correctness is proven by
  the generator's tests rather than by rendering. **No renderer is added**: there is no `mermaid`/`mmdc`
  reference in `pyproject.toml`, `.github/workflows/ci.yml`, `scripts/ci.sh` or the `Makefile`, and no
  Python mermaid parser installed, so a render gate means putting a Node toolchain into a Python-only
  CI. D-6 rejects that.
- [ ] 5.5 Assert determinism: sorted output, no timestamps, no absolute paths. Otherwise 5.3 flaps.
- [ ] 5.6 Wire the generator into `scripts/ci.sh` and a `Makefile` target.
- [ ] 5.7 Confirm `docs/architecture/package-map.md` fits `doc_hygiene.py`'s 20 KB budget, or split it
  per epic. 41 sections plus 41 diagrams is the size risk in this change.

## Phase 6 — Workforce and documentation wiring

- [ ] 6.1 Index the new skill in `SKILLS.md` — required by
  `tests/regression/test_claude_workforce_aqa.py:290`, so an unindexed skill fails CI.
- [ ] 6.2 `python tools/validate_skill_commands.py` passes; every backticked path in the new SKILL.md
  resolves.
- [ ] 6.3 Add the `AGENTS.md` layer and the package map to `docs/claude/surfaces/README.md` and the
  root `CLAUDE.md` Surface Map, stating the D-7 division of labour in one line.
- [ ] 6.4 `CHANGELOG.md` entry under `[Unreleased]`: the convention, the `agent.md` removal and why
  (not the standard filename, read by nothing), and that the root `AGENTS.md` now loads.
- [ ] 6.5 `NEXT_STEPS.md`: record the three follow-ups this change declines — validating the ~50
  existing `docs/` diagrams, a `.claude/rules/` path-scoped migration, and a full audit of the root
  `AGENTS.md`.
- [ ] 6.6 Check `docs/architecture/c4-claude-workforce.md`; it diagrams the workforce surfaces and goes
  stale once this layer exists.

## Phase 7 — Validation and closeout

- [ ] 7.1 `make gates` passes.
- [ ] 7.2 `make test` passes (all four pytest steps).
- [ ] 7.3 `python -m tools.ratchet_budgets --strict` exits 0, budgets unchanged. This change is docs,
  tests and one script, so it should need no suppression at all — if it does, that is a signal the
  approach is wrong, not that the budget should move.
- [ ] 7.4 `python scripts/validate.py --tier fast` and `bash scripts/ci.sh` pass.
- [ ] 7.5 Write `scripts/validations/F-053.sh`; register F-053 in `features.yaml` with
  `status: in_progress`, `implemented_in: null`, `depends_on: ["F-030"]`.
- [ ] 7.6 Add the regression pair. The backwards-compat half asserts the nine `CLAUDE.md` files still
  load and their invariants are unchanged — the risk of this change is weakening a contract by moving
  prose around it.
- [ ] 7.7 Register in `openspec/project.md`; flip to `implemented` with the trunk SHA after
  squash-merge.
