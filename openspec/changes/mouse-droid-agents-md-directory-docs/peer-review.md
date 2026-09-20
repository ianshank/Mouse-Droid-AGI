# Peer review: `mouse-droid-agents-md-directory-docs` (F-053)

Three revisions in one session. Each was reviewed against the tree; each had load-bearing claims
overturned. This file is the record, because a plan whose premise changed twice is worth less than one
that shows where and why.

---

## Revision 2 → 3: the review changed the deliverable, not just its reasoning

### 1. BLOCKING — two tasks wrote into a frozen path, and the same design cited the freeze

`src/mousedroid/arm/**` is denied by the repository's own PreToolUse hook:
`.claude/workforce.yaml:23-24` sets `frozen_paths: [src/mousedroid/arm/**]` gated on `F-008`, whose
`features.yaml` status is `todo`, and `.claude/settings.json` wires `freeze_gate` on
`Write|Edit|MultiEdit|NotebookEdit`.

Revision 2 scheduled an import edit to `src/mousedroid/arm/CLAUDE.md` and authorship of
`src/mousedroid/arm/AGENTS.md` — **both denied** — while its own D-7 cited that directory's freeze
notice two sections away. The freeze is also in the root `CLAUDE.md` I was given at session start.
Knowing a fact and not connecting it to the work is the exact failure this plan is about.

### 2. BLOCKING — the folder documentation already exists, so the whole deliverable was duplication

All 8 nested `CLAUDE.md` carry a purpose blockquote at lines 3-4 **and** a `## Key Files` section.
`src/mousedroid/world_model/CLAUDE.md:3` already says what revision 2's exemplar proposed to say.

So revisions 1-2 would have shipped **8 duplicated purpose statements** — the "drift factory"
revision 1 rejected by name — while leaving the 8 existing `Key Files` lists ungated, two of which are
already wrong. Revision 3 drops the sibling files entirely and executes
`docs/planning/TECH_DEBT_REMEDIATION_PLAN.md` **WS-8d** instead: *"Pick one format; index all
subsystems."* Revisions 1-2 never cited WS-8d, which is the decision this change was always making.

### 3. BLOCKING — the accuracy exemplar contained three false statements

Revision 2's D-9 reference shape — the thing Phase 3 would copy, in a plan whose premise is that
unverified folder prose rots — named:

- **`SensorBundle`**: zero occurrences in the repository. The real inbound type is
  `ObservationProtocol`.
- **an `Action` type**: no `class Action` in `src/`. `src/mousedroid/world_model/mcts.py::plan` returns a `Tensor`.
- **"no overlay sets `model.cfc_hidden_dim > 0`"**: `config/jetson_dual_stream.yaml:29` sets it to
  `64`. I had stated that correctly in PR #233's body and contradicted it here.

Success criterion 7 ("every backticked repo path resolves") caught none of them — all three are
backticked **symbols**, not paths. The criterion implied a coverage it did not have.

### 4. BLOCKING — byte-exact assertions would fail on `test-windows`

`.gitattributes` is **absent**, so a CRLF checkout changes every file's byte count, and
`.github/workflows/ci.yml` runs `tests/regression` on `test-windows`. Revision 2 asserted a 23,125-byte
figure and "regenerates byte-identically" — while rejecting symlinks *specifically* on this repo's
Windows-breakage history. The lesson was applied to the rejected alternative and not to its own tests.

### 5. MAJOR — "a diagram cannot be wrong" was false as specified

Two reasons. `TYPE_CHECKING` imports are import statements, not dependencies — three live cases sit in
`world_model` alone, and without a filter the map asserts a `world_model → telemetry` edge that exists
only for type checking. And an import graph is not a dataflow graph in a factory-first DI codebase:
`factory` has 35 outbound package edges and `config` 33 inbound *because invariants 1-2 require it*,
while the real seams run through injected Protocols `ast` cannot see. The map is now labelled an
import map, and the filter is task 5.1.

### 6. MAJOR — several "verified" numbers were wrong

| claim | actual |
|---|---|
| rename touches "~25 references" | **43 occurrences across 17 files** (40 excluding this bundle) |
| boilerplate identical in "all five subsystem copies" | **15 of the 16** files |
| "~50 fences across 21 files under `docs/`" | 49 across 20 under `docs/`; the 50th is root `README.md` |
| "three live references" to `agent.md` break a delete | **one**. The other two are docstring prose |
| the root `AGENTS.md` is "over budget" | `tools/doc_hygiene.py` runs against `NEXT_STEPS.md` **only**, so its 20 KB default gates nothing here. The figure is a reference, not a gate |
| constraint (d), the skill loader | `harness_mcp.py` `enabled` defaults `False` and no overlay sets it, so the loader never runs by default. It should not have been one of the four constraints that narrowed scope |

Each was in a table headed "verified". The sign-off condition says any such line must reproduce when a
reader runs the cited command; six did not.

---

## What survived all three revisions

Verified independently at each pass: the 16/9/1 file counts and the exact path list; zero `@` imports
in any `CLAUDE.md`, so the 403-line root `AGENTS.md` is dead text; `pyproject.toml:244` omitting
`**/AGENTS.md`; nested `AGENTS.md` discovery being off while a `CLAUDE.md` sits above; no non-Claude
agent reading this repo; `src/mousedroid/llm_gateway/agent.md:6` false in both halves; 40 of 41
packages carrying an `__init__.py` docstring with `telemetry` at 0 bytes; no mermaid renderer anywhere
in the toolchain; 147 directories under a literal reading.

Two structural judgements also survived and are revision 2's real contribution: stating the benefit as
optionality rather than a present capability gain, and generating diagrams instead of hand-drawing and
render-gating them.

---

## Open weaknesses in revision 3

1. **Revision 3 substitutes twice over.** The request was per-directory `Agent.md`. Revision 3 delivers
   per-directory `CLAUDE.md` plus one root `AGENTS.md` plus a generated map. Both substitutions are
   argued in §2 and §3 and both are mine. A reviewer who wants the literal thing should say so; §7
   records the only mechanically sound route to it (a `CLAUDE.md` + `AGENTS.md` pair per package,
   `arm` excepted, at two files each).
2. **Phase 2 writes 9 new files of exactly the genre §7's evidence argues against.** They replace 9
   unread stubs at roughly the same size, so the token delta is near zero — but "near zero" is an
   estimate, not a measurement, and nobody has drafted one to check.
3. **The D-5 gate may be over-ambitious.** Generalising the orchestrator symbol test across 17 files
   assumes every `Key Files` section is machine-parseable in the same shape. It was written for one
   file. If the shapes differ, the gate becomes per-file special cases, which is how gates rot.
4. **`arm`'s stale entry stays wrong for the life of F-008.** Recorded as a declared exemption rather
   than fixed, which is correct under the freeze and still means a known-false line ships.
5. **Three revisions in one session, each driven by re-reading a source I had already read.** The
   freeze was in my own project instructions. `jetson_dual_stream.yaml:29` was in a PR body I wrote.
   The purpose blockquotes were in files I had cited by line number. The failure was never discovery;
   it was that a claim, once written, stops being re-read.

---

## Sign-off condition

Accepted when a reader can take any row of `proposal.md` §6, run the command or open the cited file,
and get the stated result — **including the backticked symbols, not just the paths.** Revision 2 met
that condition for paths and failed it for symbols, which is how three false statements reached the
document held up as the accuracy exemplar.
