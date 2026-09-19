# Design: `mouse-droid-agents-md-directory-docs` (F-053)

**Revision 3.** Revisions 1 and 2 planned a sibling `AGENTS.md` per directory. Review established that
the content it would hold already exists in the `CLAUDE.md` beside it, that one of the target
directories is frozen against writes, and that the reference shape offered as the accuracy exemplar
contained three false statements. Revision 3 changes the deliverable. `peer-review.md` has the record.

---

## D-1 — Pick one format, and it is `CLAUDE.md`

**Decision.** In-package documentation lives in `CLAUDE.md`. `AGENTS.md` exists once, at the root,
imported. This executes `docs/planning/TECH_DEBT_REMEDIATION_PLAN.md` **WS-8d** — *"Pick one format;
index all subsystems"* — which revisions 1–2 never cited.

Three mechanical reasons, not preferences:

1. **`CLAUDE.md` is the only reachable per-directory format.** With a root `CLAUDE.md` present, nested
   `AGENTS.md` discovery is off entirely (D-3). Delivering one file's content elsewhere would take two.
2. **It already holds the content.** All 8 nested `CLAUDE.md` carry a purpose blockquote at lines 3–4
   and a `## Key Files` section. A sibling `AGENTS.md` restating the purpose is duplication by
   construction — the drift factory revision 1 rejected by name, which revisions 1–2 then designed.
3. **One target cannot be written to.** `src/mousedroid/arm/**` is denied by the PreToolUse
   `freeze_gate` hook while `F-008` is `todo` (D-4).

**Rejected: `AGENTS.md` as the per-directory format.** It needs a `CLAUDE.md` importer in every
directory to be read at all, so 9 packages become 18 files; `arm` cannot take either; and the 8
existing `CLAUDE.md` would need their purpose blockquotes stripped to avoid duplication, touching the
43 references D-8 measures.

**Rejected: both formats, disjoint content.** This is what revision 2 proposed. It survives only if
nothing is restated — and §1.3's purpose blockquotes mean the restatement is already there before the
first new file is written.

## D-2 — The one-line change that is the whole present-day win

**Decision.** Root `CLAUDE.md` gains `@AGENTS.md` as its first line, and `pyproject.toml:244` gains
`"**/AGENTS.md"`. This is Phase 1, ships alone, and is worth shipping alone.

The repository has a **403-line root `AGENTS.md`** and **zero `@` imports in any of its nine
`CLAUDE.md` files**. That file is read by nothing. It is maintained, cited by tests and by docstrings
across `src/`, and has never loaded. One line fixes that and simultaneously makes the tool-neutral
standard file this repo's public agent surface — which is the whole standards-compliance goal, achieved
once rather than 41 times.

Confirmed empirically: the session that authored this plan received `CLAUDE.md`'s contents in its
instruction payload and not `AGENTS.md`'s.

## D-3 — Reachability: nested `AGENTS.md` discovery is off, not shadowed

The documented rule: Claude reads `AGENTS.md` "only when you have no `CLAUDE.md` in your working
directory **or above it**", and the nested-subdirectory bullet sits **inside** that "when none count"
block. A root `CLAUDE.md` counts for every directory beneath it.

So with one present, no `AGENTS.md` in this tree is auto-loaded at any depth, and the only load path is
an `@AGENTS.md` from a `CLAUDE.md` in the same directory. Revision 1 called those eight directories
"where `AGENTS.md` would be shadowed"; they are the only places it could load at all. Revision 3 uses
that fact to stop putting files where they cannot be read.

The `claude-md-and-agents-md` setting would change this, but it is **ignored in project and local
settings files** — user or managed scope only — so it cannot be a repository decision. A symlink is
rejected on this repo's own Windows history: the docs warn it needs Administrator or Developer Mode and
that git checks a committed one out as plain text without `core.symlinks`.

## D-4 — The F-008 freeze, handled rather than tripped over

`.claude/workforce.yaml` sets `frozen_paths: [src/mousedroid/arm/**]`, gated on `F-008`, whose
`features.yaml` status is `todo`. `.claude/settings.json` wires `freeze_gate` as a PreToolUse hook on
`Write|Edit|MultiEdit|NotebookEdit`. Every write under that path is denied.

Revisions 1–2 scheduled two write tasks there — add an import to `src/mousedroid/arm/CLAUDE.md`, author
`arm/AGENTS.md` — while citing the freeze notice two sections away. That is the same failure mode this
whole plan is about: knowing a fact and not connecting it to the work.

**Decision.** `arm` needs no new file (it already has a `CLAUDE.md`), so revision 3 has nothing to
write there. Its one known defect — `src/mousedroid/arm/CLAUDE.md:19` names `mock_arm.py` where the
file is `src/mousedroid/arm/hardware/mock_arm_driver.py` — is recorded as a **declared exemption in the
D-5 gate**, with F-008 as the reason. The gate is not weakened to pass; the exemption is enumerated,
and it lapses when the freeze does.

**Rejected: `MOUSEDROID_WORKFORCE_ALLOW_FROZEN=1`.** The override exists for reviewed exceptional
edits. A stale filename in a doc is not exceptional, and using the override to fix documentation would
set the precedent that the freeze yields to convenience.

## D-5 — Gate the `Key Files` lists, by generalising a gate that already works

**Decision.** Extend `tests/regression/test_doc_reconciliation_aqa.py::test_orchestrator_claude_md_names_only_real_symbols`
(`:191`) from one file to **every** nested `CLAUDE.md`.

That test already does exactly the right thing for `orchestrator`: it asserts each named symbol
resolves to a real `class`/`def` in its mapped file, and that two named phantoms are absent. It is the
repository's only content gate on a nested surface. Generalising it is cheaper than inventing a gate and
strictly better than the alternative revisions 1–2 chose, which was to leave the 8 existing `Key Files`
lists ungated while adding a 9th surface beside them.

It catches the two known-stale entries immediately: `src/mousedroid/llm_gateway/CLAUDE.md:26`
(`mock_gateway.py`, does not exist) and `src/mousedroid/arm/CLAUDE.md:19` (`mock_arm.py`, wrong path).
The first is fixed; the second is the D-4 exemption.

**It also catches a third defect nothing else would:**
`src/mousedroid/llm_gateway/CLAUDE.md:20-21` requires `LLMConfig.fallback_backend` to "target local
backends (e.g. `mock`, `ollama`)" — and `src/mousedroid/config/schema/llm.py:157` permits only `none`,
`llama_cpp`, `openai_compatible`. **Neither named value is legal.** Revision 1 recorded this and
revision 2 silently dropped it; the gate makes dropping it impossible.

## D-6 — The generated package map, and what it honestly describes

**Decision.** `scripts/generate_package_map.py` emits `docs/architecture/package-map.md`: one section
per package with a purpose line from its `__init__.py` docstring, its imports, its dependents, and a
mermaid subgraph — all from the `ast`-parsed import graph.

**Two corrections to revision 2's version of this, both from review:**

1. **`TYPE_CHECKING` imports are filtered out.** They are import statements, not dependencies. Three
   live cases sit in the exemplar package alone — `src/mousedroid/world_model/cfc_cell.py`,
   `checkpoint_migration.py`, `dual_stream_rssm_onnx.py` — and without the filter the map would assert
   a `world_model → telemetry` edge that exists only for type checking.
2. **It is labelled an import map, not a dataflow map.** Revision 2 claimed "a diagram cannot be
   wrong, because it is derived from the imports it depicts". The derivation is sound; the *relation*
   is not the one a folder description wants. In a factory-first DI codebase they diverge by design —
   `factory` has 35 outbound package edges and `config` 33 inbound because invariants 1–2 require it,
   and the real runtime seams run through injected Protocols that `ast` cannot see. So the map states
   what imports what, accurately, and says that is what it states.

**Still no renderer.** Nothing in `pyproject.toml`, `.github/workflows/ci.yml`, `scripts/ci.sh` or the
`Makefile` references `mermaid` or `mmdc`, and no Python parser is installed. Correctness comes from
the generator's unit tests against a fixture tree, and one regenerate-and-diff test covers all 41
sections.

**Sized, not guessed:** 41 packages and 221 directed package edges estimate to ~34 KB, so the map
**splits per epic** from the start rather than leaving a fork in the deliverable.

## D-7 — Assertions that hold on `test-windows`

`.gitattributes` is **absent**, so a CRLF checkout changes every file's byte count, and
`.github/workflows/ci.yml` runs `tests/regression` on `test-windows`.

**Decision.** No byte-exact assertion anywhere in this change. Size checks are line counts. The
regenerate-and-diff test normalises line endings and path separators before comparing. Revision 2 had
three byte-exact assertions and rejected symlinks *specifically* on this repo's Windows history without
applying the same lesson to its own tests.

## D-8 — No rename, measured

Renaming the eight nested `CLAUDE.md` to `AGENTS.md` would touch **43 occurrences across 17 files** (40
excluding this bundle): the root Surface Map, `docs/claude/surfaces/README.md`, `CHANGELOG.md`,
`progress.md`, `src/mousedroid/config/schema/hardware.py`, `scripts/validations/F-031.sh`, four openspec
bundles, and a path-specific test. Revision 2 said "~25"; the number in a row it marked verified was
off by 15. `CHANGELOG.md` and the openspec bundles are historical records.

## D-9 — Where the persona content goes

The 16 `agent.md` are persona prompts. Personas belong in `.claude/agents/`, where 7 live under a
validated contract (`max_lines: 60`, required frontmatter, bare tool names —
`tests/regression/test_claude_workforce_aqa.py:182`). Each `agent.md` splits: folder-purpose half into
the sibling `CLAUDE.md` (new for the 9, already present for the 5 that have both); persona half
promoted only where it earns a definition.

**Only one reference actually breaks a delete.** Revision 2 claimed three.
`tests/regression/test_doc_reconciliation_aqa.py:46` puts `tests/agent.md` in `_SRC_COVERAGE_DOCS` and
`read_text()`s it at `:130` — that one breaks. The other two are **docstring prose**:
`src/mousedroid/skills/loaders.py:7,103` describe behaviour that keys off absent front matter and name
no file, and `tests/regression/test_ci_gate_wiring_aqa.py:825` sits in a docstring that says the roster
is now sourced from `git ls-files`.

## D-10 — What this deliberately does not do

- **No new `AGENTS.md` outside the root.** D-1, D-3.
- **No write under `src/mousedroid/arm/**`.** D-4.
- **No rename of the eight contracts.** D-8.
- **No hand-drawn diagrams, no renderer in CI.** D-6.
- **No claim that `tools/doc_hygiene.py` enforces a budget on these files.** It is invoked against
  `NEXT_STEPS.md` only (`Makefile:143`, `ci.yml:421`, `scripts/ci.sh:65`, pinned by
  `test_f038_aqa.py:62-63`). Its 20 KB default is a **reference figure** for judging size, not a gate
  that applies here. Revisions 1–2 wrote "over budget" as though something enforced it.
- **No `.claude/rules/` migration**, and **no audit of the root `AGENTS.md`'s 403 lines** beyond the
  contradictions Phase 1 makes live.
