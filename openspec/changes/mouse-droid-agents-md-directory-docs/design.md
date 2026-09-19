# Design: `mouse-droid-agents-md-directory-docs` (F-053)

**Revision 2.** Revision 1 was reviewed against the tree and three of its load-bearing claims were
wrong — including the reason the change existed. `peer-review.md` records what changed and why. Each
section below is a decision with its rejected alternatives.

---

## D-1 — Why `AGENTS.md` at all, stated so a reviewer can decline it

Revision 1 justified this change by asserting that "this repository already runs two non-Claude
reviewers on its pull requests — Devin Review and CodeRabbit both posted on #233 — so content that
lives only in `CLAUDE.md` is invisible to the tools reviewing the code."

**That was misleading, and it was the whole argument.** Verified on both #233 and #235:

| check | actual description |
|---|---|
| Devin Review | `Full review skipped: trial expired and no credits remaining` |
| CodeRabbit | `Review skipped: draft pull request`; on #233 also `manual review required for this OSS repository` |

Neither reviews anything. They post skip statuses. There is no `.coderabbit.yaml`, no Codex or Cursor
configuration, and nothing under `.github/` consumes `AGENTS.md`.

Combine that with D-3 and the honest position is stark: **for Claude Code, content in `CLAUDE.md` and
content in an `AGENTS.md` that `CLAUDE.md` imports are the same thing.** The delta `AGENTS.md` buys is
that *other* tools can read it — and today, on this repository, no other tool does.

**Decision: proceed, on optionality and standards-compliance, not on a capability gain.** `AGENTS.md`
is the Linux Foundation-stewarded filename that 30+ agents read; putting the content there costs one
import line per directory and means the day someone runs Codex, Cursor or a credited Devin against
this repo, the instructions are already in the file those tools look for. That is a real benefit. It
is a *future* benefit, and this section says so rather than dressing it up as a present one.

**A reviewer who does not want to pay for optionality should decline at this section**, and the
plan is arranged (D-2) so that declining the bulk still leaves the one change worth making.

## D-2 — The one line that is the entire present-day win

**Decision.** Root `CLAUDE.md` gains `@AGENTS.md` as its first line. This ships independently of
everything else and is Phase 1 on its own.

The repository has a **403-line root `AGENTS.md`** and **zero `@` imports in any of its nine
`CLAUDE.md` files**. By the loading rules in D-3, that file is read by nothing. It has been
maintained, cited by tests and by `AGENTS.md`-referencing docstrings across the tree, and it has never
been loaded.

One line makes 403 lines of existing, already-written instruction live for the first time. Nothing
else in this plan comes close on value per unit of work, and it is why the phases are ordered so that
a reviewer can take Phase 1 and stop.

**Empirical confirmation, from this session.** The instruction payload for the session that wrote this
plan carried the contents of `CLAUDE.md` and **not** `AGENTS.md`. Revision 1 listed observing that as
Phase 0.2, still to be done; it was already done, in the act of writing.

## D-3 — Reachability, corrected: nested discovery is off, not shadowed

Revision 1 said the eight directories holding a `CLAUDE.md` are "exactly the directories where a
sibling `AGENTS.md` would otherwise be shadowed". **That is backwards.**

The documentation's rule reads: "Claude reads `AGENTS.md` only when you have no `CLAUDE.md` in your
working directory **or above it**", and the nested-subdirectory bullet — "a subdirectory's
`AGENTS.md`, when Claude opens a file there … and that subdirectory has none of the three `CLAUDE.md`
files of its own" — sits **inside** the "when none count" block.

A root `CLAUDE.md` counts, being above every subdirectory. So with one present:

- **nested `AGENTS.md` auto-discovery is off entirely**, everywhere in the tree;
- the only load path for a nested `AGENTS.md` is an explicit `@AGENTS.md` from a `CLAUDE.md` **in
  that same directory**.

**Consequence that reshapes the scope.** Those eight directories are not where `AGENTS.md` is
shadowed — they are the **only** places a nested `AGENTS.md` can load at all. In the other 33
packages it is unreachable, and making it reachable would mean adding a `CLAUDE.md` stub per package
purely to import a sibling: two files where one would do, for content Claude would read either way.

That is the argument against a per-package `AGENTS.md`, and it is mechanical rather than aesthetic.

## D-4 — Scope: 8 + 1, plus a generated map for all 41

| surface | scope | why that scope |
|---|---|---|
| `AGENTS.md` | the **8** directories that hold a `CLAUDE.md` | the only directories where D-3 permits it to load |
| `tests/CLAUDE.md` + `tests/AGENTS.md` | **1** more pair | `tests/` is high-traffic and has no `CLAUDE.md`, so its `AGENTS.md` needs a one-line importer or it is dead |
| `docs/architecture/package-map.md` | **all 41** packages, generated | answers "every folder" at zero agent-token cost, and cannot drift |

**`tests/` was a defect in revision 1.** The nine-file set included `tests/AGENTS.md` while
`tests/CLAUDE.md` does not exist — so under D-3 that file would have been dead on arrival. The
plan that exists to stop unreadable documentation was about to ship some. It is now an explicit pair,
or it is dropped; task 2.6 forces the choice rather than leaving it implied.

**Why generated for the 41.** A hand-written folder description rots, and the 16 `agent.md` files are
the proof — their `Key Files` lists are the part that rotted. A map built from `ast`-parsed imports
states what each package actually depends on and who depends on it, regenerates in CI, and is diffed
so a stale one fails. 40 of the 41 packages already carry an `__init__.py` docstring to source the
purpose line from; only `src/mousedroid/telemetry` lacks one, so the generator's fail-closed rule has
exactly one pre-existing case to fix.

## D-5 — No rename, on measured grounds

The tidiest-looking option is to rename each nested `CLAUDE.md` to `AGENTS.md` and leave a one-line
`CLAUDE.md` importing it: content in the tool-neutral file, no duplication, standard-compliant.

**Rejected, measured.** The eight paths are referenced ~25 times: root `CLAUDE.md`'s Surface Map (8),
`docs/claude/surfaces/README.md` (8), `CHANGELOG.md` (4),
`src/mousedroid/config/schema/hardware.py` (1), `scripts/validations/F-031.sh` (1), three openspec
bundles, and a path-specific test
(`test_doc_reconciliation_aqa.py::test_orchestrator_claude_md_names_only_real_symbols`). Live
references could be updated; `CHANGELOG.md` and the openspec bundles are **historical records** and
rewriting them to match a later rename is the kind of retroactive edit this repository has
consistently refused.

So the eight `CLAUDE.md` contracts stay where they are, under their current names, and `AGENTS.md`
lands beside them holding different content (D-7).

## D-6 — Diagrams belong in the generated map, not hand-written per file

Revision 1 asked for one hand-drawn `flowchart` per `AGENTS.md`, gated by a check that "every fence
parses and renders".

**That gate does not exist and is not cheap to build.** There is no `mermaid` or `mmdc` reference in
`pyproject.toml`, `.github/workflows/ci.yml`, `scripts/ci.sh` or the `Makefile`, and no Python mermaid
parser is installed. Implementing it means adding a Node toolchain to a Python-only CI — a
substantially larger change than a plan task, presented as a one-liner.

**Decision: the generator emits every diagram.** `scripts/generate_package_map.py` builds each
package's dependency subgraph from the parsed import graph and writes the fence. Three properties
follow that hand-drawing cannot match:

1. **A diagram cannot be wrong**, because it is derived from the imports it depicts.
2. **One golden-file test covers all 41** — regenerate and diff — instead of 41 unvalidated fences.
3. **No renderer is needed.** Correctness comes from the generator's own unit tests over a fixture
   package, not from rendering.

The nine hand-written `AGENTS.md` carry no diagram. They are instruction files under a tight line cap
(D-9); a diagram there would be prose the generator already produces, better, elsewhere.

This satisfies "with mermaid" more strongly than revision 1 did: 41 diagrams that are provably
accurate, rather than 9 that nothing checks.

## D-7 — Division of labour: purpose versus contract

| file | answers | changes when |
|---|---|---|
| `AGENTS.md` | *What is this folder for? What flows in and out?* | the architecture changes |
| `CLAUDE.md` | *What must never be weakened here?* — numbered invariants, freeze notices | a contract changes |

The eight nested `CLAUDE.md` contracts are not rewritten. `src/mousedroid/world_model/CLAUDE.md`'s
numbered invariants, `src/mousedroid/telemetry/CLAUDE.md`'s success-path-recording rule and
`src/mousedroid/arm/CLAUDE.md`'s F-008 freeze notice stay authoritative.

**The rule that stops drift:** an invariant is stated in exactly one file. `AGENTS.md` may *point* at
a contract, never restate one. Revision 1 proposed gating this with a substring check (task 3.6);
paraphrase defeats that, so it is now a review item in the skill rather than a gate that would be
worked around — recorded as a deliberate downgrade, not dropped silently.

## D-8 — Where the persona content goes

The 16 `agent.md` files are persona prompts — *"You are the **World Model Architect**"* — which is
neither folder documentation nor the build/test guidance `AGENTS.md` is for. Personas belong in
`.claude/agents/`, where 7 already live under a validated contract (`max_lines: 60`, required
frontmatter, bare tool names; `tests/regression/test_claude_workforce_aqa.py:182-257`).

Each `agent.md` is **split**: folder-purpose half into the sibling `AGENTS.md` where one of the nine
exists, otherwise into the package's `__init__.py` docstring, which D-4's generator reads; persona
half evaluated against the existing seven and promoted only where it earns a definition. Fourteen
near-identical "you are the X architect" stubs do not each warrant a subagent.

**Not deleted blind:** three are referenced by live code and tests
(`tests/regression/test_doc_reconciliation_aqa.py:46`,
`tests/regression/test_ci_gate_wiring_aqa.py:825`, `src/mousedroid/skills/loaders.py:7,103`).

## D-9 — Token budget, and the delete test as a gate

**Decision.** Each of the nine `AGENTS.md` is capped at **30 lines** — lower than revision 1's 40,
because D-6 removed the diagram those lines were reserved for — and every line must pass: *would a
competent developer who had never seen this repo get this wrong?*

Current practice puts these files under ~150 lines and names architecture description as the
anti-pattern; reported measurements have auto-generated files at ~3% *lower* task success for >20%
more inference cost. The 16 existing `agent.md` files are the worked example of the failure: each
repeats "Protocol-based DI", "no hardcoded values", "structlog not print()" — all three already in the
root `CLAUDE.md`, and the third already enforced by `ruff` `T20`.

**The cap is measured, not asserted.** Drafted against `src/mousedroid/world_model` — the hardest
case, carrying two engines and the most behaviour — the shape below comes to **25 lines**, so 30 has
five lines of headroom on the worst package. This is the reference shape Phase 3 authors against:

```markdown
# world_model

Latent dynamics and planning. Encodes a sensor bundle into a latent state, rolls that state
forward without rendering pixels, and hands the planner a scored action.

## What flows through

- **In** — `SensorBundle` from `sensing`, built against dimensions in `config.WorldModelConfig`.
- **Out** — a latent `(z, h)` pair to `mcts`, an `Action` to `orchestrator`, and an
  observe-step histogram to `telemetry` through an injected `ObserveStepLatencySink`.

## Packages this speaks to

`sensing` and `config` upstream; `orchestrator` and `telemetry` downstream. Concrete engines are
constructed only in `factory.world_model` — nothing here imports a concrete type.

## Engines

`engine: torch` is the shipped default and builds a plain `RSSM`, because no overlay sets
`model.cfc_hidden_dim > 0`. `onnx_trt` builds a `CompositeWorldModel` and requires both a locally
exported artifact and that dimension.

## Invariants

Numbered contracts live in `CLAUDE.md` in this directory. They are not restated here.
```

Note what it does not contain: no `Key Files` list (the section that rotted in all 16 `agent.md`
files), no restated invariant, no diagram, and nothing a competent developer would already do.

## D-10 — Three ways this change could do harm, closed in Phase 1

1. **`pyproject.toml:244` gains `"**/AGENTS.md"`.** The exclude list is
   `["**/CLAUDE.md", "**/agent.md"]`, with a comment naming the harm: without it the wheel ships
   internal agent instructions to PyPI. Revision 1 would have added files under `src/mousedroid/` and
   shipped every one. A test asserts all three patterns, so the next agent-facing filename cannot slip
   through either.
2. **`src/mousedroid/agents/` is scanned by the skill loader** (`harness_mcp.py:394` defaults
   `markdown_agent_dirs` there and the loader globs `*.md`). It is not one of the nine, and a test
   pins that it stays out.
3. **The root `AGENTS.md` may not grow.** At 23,125 bytes it is already 1.16× `doc_hygiene.py`'s
   20,000-byte budget, and widening that gate is deferred precisely because files like it would fail.
   This change may shrink it; it may not grow it.

## D-11 — What this deliberately does not do

- **No `AGENTS.md` in the 33 packages without a `CLAUDE.md`.** D-3: unreachable without a second file
  per package. Their folder documentation is the generated map.
- **No rename of the eight contracts.** D-5, measured.
- **No hand-drawn diagrams and no renderer in CI.** D-6.
- **No `.claude/rules/` migration.** Path-scoped rules load alongside `AGENTS.md` and would suit some
  cross-cutting content, but a third surface triples the review burden here. Recorded as follow-up.
- **No toolchain floor.** The import path works on any version; native `AGENTS.md` reading needs
  v2.1.277+ and nothing here requires it.
- **No audit of the root `AGENTS.md`'s 403 lines.** Phase 1 makes it load, which makes its known
  staleness (two references to `CLAUDE.md` sections that do not exist; a subagent count of 8 against
  an actual 7) live rather than dormant. Those specific rows are fixed; a full audit is its own change.
