# Change: `mouse-droid-agents-md-directory-docs`

**F-number:** F-053 (F-052 is reserved by `mouse-droid-branch-hygiene-sweep`; F-009–F-014 and F-033 remain burned holes per ADR-013)
**Status:** proposed
**Tree at authoring:** `992da04` on the default branch `claude/markdown-implementation-plan-aVJ2l`

> There is no `main` in this repository. The default branch is named above.

---

## 1. What was asked, and what the tree already contains

The request: add `Agent.md` files in each directory describing that folder's function, with mermaid
diagrams, using subagents, meeting current coding standards.

Three facts change the shape of that work, and all three were verified before this was written.

### 1.1 It is partly built already — under a filename nothing standard reads

`git ls-files` finds **16 `agent.md` files** — lowercase, singular:

```
agent.md                              src/mousedroid/learning/agent.md
src/mousedroid/agents/agent.md        src/mousedroid/llm_gateway/agent.md
src/mousedroid/cognitive/agent.md     src/mousedroid/logging/agent.md
src/mousedroid/comms/agent.md         src/mousedroid/memory/agent.md
src/mousedroid/config/agent.md        src/mousedroid/orchestrator/agent.md
src/mousedroid/experience/agent.md    src/mousedroid/safety/agent.md
src/mousedroid/hardware/agent.md      src/mousedroid/sensing/agent.md
                                      src/mousedroid/world_model/agent.md
                                      tests/agent.md
```

The standard filename is **`AGENTS.md`** — uppercase, plural. It was formalised as an open
specification in August 2025 (OpenAI, with Google, Cursor and Factory), donated to the Linux
Foundation's Agentic AI Foundation in December 2025, and is read by 30+ agents including Codex,
Copilot, Cursor, Gemini CLI, Aider, Zed, Windsurf and Devin. `agent.md` is not that file. No
agent tooling reads these 16 files, and nothing in-repo requires or validates them beyond three
incidental references (§1.3).

### 1.2 They are persona prompts, not folder documentation

`src/mousedroid/world_model/agent.md` in full is 16 lines opening *"You are the **World Model
Architect** for MouseDroid"*, then Responsibilities and a Key Files list. That is a **role prompt**.
The request — "describing the functions of each folder" — is a different genre, and so is
`AGENTS.md`, which the spec frames as how to build, test and change the code. Every one of the 16
is 13–25 lines. **None contains a mermaid diagram**; the repo's diagram surface is 50 fences across
21 files, all under `docs/` and `README.md`, and there is none anywhere under `src/`.

They are also ~80% duplication, and one of them is **actively wrong**. Three lines are byte-identical
across all five subsystem copies — "Protocol-based DI patterns", "No hardcoded values", "Use structlog
… never print()" — each already stated in the root `CLAUDE.md` and root `AGENTS.md`, and the third
already enforced by `ruff` `T20`. And `src/mousedroid/llm_gateway/agent.md:6` claims *"Natural language
to velocity commands via local LLM"*: `protocol.py:65` returns a `GoalVector`, not velocity commands,
and `src/mousedroid/config/schema/llm.py:96` lists `anthropic` among the backends, so "local" is stale. Both halves
of that line are false today.

### 1.3 Nothing gates them

| reference | what it actually asserts |
|---|---|
| `tests/regression/test_doc_reconciliation_aqa.py:46` | lists `tests/agent.md` in a tracked-docs roster |
| `tests/regression/test_ci_gate_wiring_aqa.py:825` | treats `tests/agent.md` as an agent-facing instructions file |
| `src/mousedroid/skills/loaders.py:7,103` | excludes `src/mousedroid/agents/agent.md` from skill registration |

No gate requires a directory to have an agent-facing file, and none checks that one is accurate.

## 2. The two findings that decide the design

### 2.1 With a root `CLAUDE.md`, nested `AGENTS.md` discovery is off — not shadowed

Claude Code reads `AGENTS.md` natively from v2.1.277, but the documented rule is: it does so "only
when you have no `CLAUDE.md` in your working directory **or above it**". The nested-subdirectory
bullet — "a subdirectory's `AGENTS.md`, when Claude opens a file there … and that subdirectory has
none of the three `CLAUDE.md` files of its own" — sits **inside** that "when none count" block.

A root `CLAUDE.md` counts, being above every subdirectory. This repository has one, plus eight nested.
So today:

- **no `AGENTS.md` anywhere in this tree is read**, including the 403-line root one;
- the only load path for a nested `AGENTS.md` is an explicit `@AGENTS.md` from a `CLAUDE.md` **in that
  same directory**.

Confirmed in the tree: **zero `@` imports in any of the nine `CLAUDE.md` files**. And confirmed
empirically — the session that wrote this plan received `CLAUDE.md`'s contents in its instruction
payload and not `AGENTS.md`'s.

The consequence for scope is mechanical: a nested `AGENTS.md` is reachable **only** in the eight
directories that already hold a `CLAUDE.md`. In the other 33 packages it would need a second file per
package purely to import it.

Only one remedy is committable. The `claude-md-and-agents-md` setting lives under the built-in
`agents-md` plugin's `pluginConfigs` and is **ignored in project and local settings files** — user or
managed only, so it cannot be a repository decision. The symlink alternative is ruled out on this
repo's own grounds: the docs warn a symlink needs Administrator or Developer Mode on Windows and git
checks a committed one out as plain text without `core.symlinks`, and this repo runs a `test-windows`
job that has already caught four rounds of Windows-only breakage.

### 2.2 The benefit is optionality, not a present capability gain

An earlier revision justified this change by claiming the repository "already runs two non-Claude
reviewers on its pull requests, so content that lives only in `CLAUDE.md` is invisible to the tools
reviewing the code." **That was misleading.** Verified on #233 and #235:

| check | actual description |
|---|---|
| Devin Review | `Full review skipped: trial expired and no credits remaining` |
| CodeRabbit | `Review skipped: draft pull request`; on #233 also `manual review required for this OSS repository` |

Neither reviews. There is no `.coderabbit.yaml`, no Codex or Cursor configuration, and nothing under
`.github/` consumes `AGENTS.md`.

Put together with 2.1: **for Claude Code, content in `CLAUDE.md` and content in an `AGENTS.md` that
`CLAUDE.md` imports are the same thing**, and no other tool currently reads this repo. The delta is
that the day someone runs Codex, Cursor or a credited Devin here, the instructions are already in the
file those tools look for.

That is a real benefit and a future one. Stating it that way is what lets a reviewer decline the bulk
of this change on its merits — and the plan is ordered so that declining it still leaves the one piece
worth having.

## 3. Scope

"Each directory" read literally is **147 directories**. Beyond the drift surface that creates, §2.1
makes most of them unreachable. Four further constraints, all verified, push the same way:

| # | constraint |
|---|---|
| a | `pyproject.toml:244` is `exclude = ["**/CLAUDE.md", "**/agent.md"]` — **`AGENTS.md` is absent**, so new files under `src/mousedroid/` would ship to PyPI |
| b | `docs/planning/TECH_DEBT_REMEDIATION_PLAN.md` already rules `agent.md` **Delete/merge** as unresolved drift, and records invariants living in four places |
| c | the root `AGENTS.md` is 23,125 bytes against `doc_hygiene.py`'s 20,000-byte budget, and widening that gate is deferred because files like it would fail |
| d | `src/mousedroid/agents/` is scanned by the skill loader (`harness_mcp.py:394` globs `*.md` there) |

**In scope, in value order:**

1. **One import line** — root `CLAUDE.md` gains `@AGENTS.md`, plus the `pyproject.toml` exclude. This
   makes 403 lines of existing, already-maintained instruction load for the first time. It ships
   alone.
2. **Eight `AGENTS.md`**, in the directories §2.1 permits, each importing-side wired and capped at 30
   lines. Plus a decision on `tests/`: it is high-traffic and has no `CLAUDE.md`, so it needs a
   one-line importer or it is dropped.
3. **One generated `docs/architecture/package-map.md`** covering all 41 packages — purpose, imports,
   dependents and a mermaid subgraph per package, built from the `ast`-parsed import graph.

**Out of scope:** an `AGENTS.md` in the 33 packages without a `CLAUDE.md` (unreachable per §2.1);
renaming the eight contracts (~25 references including `CHANGELOG.md` and three openspec bundles that
are historical records); hand-drawn diagrams and a CI renderer (§5); sub-packages below the first
level.

## 4. Success criteria

1. `tests/regression/test_f053_aqa.py` fails if any `AGENTS.md` in the tree lacks a same-directory
   `CLAUDE.md` importing it with the bare `@AGENTS.md` form. **This is the criterion the plan exists
   for**: an unimported `AGENTS.md` is read by nothing.
2. The root `AGENTS.md` loads, proven by that import being present and by the known contradictions it
   surfaces being fixed.
3. `pyproject.toml`'s wheel exclude covers all three agent-facing patterns, asserted by a test.
4. `docs/architecture/package-map.md` regenerates byte-identically — a stale map fails.
5. Zero `agent.md` files remain; the three live references to them are updated first.
6. Every `AGENTS.md` is ≤30 lines and restates no invariant from its sibling `CLAUDE.md`.
7. Every backticked repo path in a new file resolves on disk.
8. `make gates` and `make test` pass; the three suppression budgets are unchanged.

## 5. Why the diagrams are generated, not written

The request asked for mermaid. Hand-drawing one diagram per file fails two ways here.

**There is no renderer to gate them with.** No `mermaid` or `mmdc` reference in `pyproject.toml`,
`.github/workflows/ci.yml`, `scripts/ci.sh` or the `Makefile`, and no Python mermaid parser installed.
A "does it render" gate means adding a Node toolchain to a Python-only CI — a far larger change than a
plan task.

**And an ungated diagram is a future lie.** The repository already carries ~50 mermaid fences across
21 files under `docs/`, validated by nothing.

So the generator emits every diagram from the parsed import graph. A diagram cannot be wrong, because
it is derived from the imports it depicts; one regenerate-and-diff test covers all 41; and correctness
is proven by the generator's unit tests against a fixture tree rather than by rendering. That is 41
provably accurate diagrams instead of 9 unchecked ones.

## 6. Verified — recorded so the plan stays falsifiable

| claim | evidence |
|---|---|
| Standard filename is `AGENTS.md`, uppercase; Linux Foundation Agentic AI Foundation since Dec 2025; 30+ agents read it | agents.md spec and its stewardship |
| **But no such agent reads this repository today** | Devin: trial expired, no credits. CodeRabbit: skipped. No Codex/Cursor config. No `AGENTS.md` consumer in `.github/` |
| Claude Code reads `AGENTS.md` natively from v2.1.277 | `code.claude.com/docs/en/memory` |
| With a `CLAUDE.md` at or above the cwd, **no** `AGENTS.md` is read — nested discovery included | same page, the three-row table plus the "when none count" block |
| Zero `@` imports in any of the nine `CLAUDE.md` files, so the 403-line root `AGENTS.md` is dead text | grep |
| The both-files setting is not committable | same page: "ignored in project and local settings files" |
| Symlink degrades on Windows | same page, "Share one file with other coding tools" |
| `pyproject.toml:244` omits `**/AGENTS.md` | read |
| Root `AGENTS.md` 23,125 bytes vs `_DEFAULT_MAX_BYTES = 20_000` | `wc -c`, `tools/doc_hygiene.py:28` |
| 40 of 41 packages have an `__init__.py` docstring; only `src/mousedroid/telemetry` lacks one | scripted check |
| No mermaid renderer anywhere in the toolchain; ~50 fences under `docs/` validated by nothing | grep |
| 16 `agent.md`, 9 `CLAUDE.md`, 1 root `AGENTS.md`; none of the 16 has a diagram; `src/` has none | `git ls-files`, grep |
| `src/mousedroid/llm_gateway/agent.md:6` is false in both halves | `protocol.py:65` returns `GoalVector`; `src/mousedroid/config/schema/llm.py:96` includes cloud `anthropic` |
| Renaming the eight contracts touches ~25 references | grep across `--include=*.py --include=*.md --include=*.yaml --include=*.sh` |

## 7. The evidence against, and the condition that would change the recommendation

Current practice says these files should be short and sparse, and names folder description as the
anti-pattern: the test for any line is *would a competent developer who had never seen this repo get
this wrong?* Guidance is to stay under ~150 lines. Reported measurements are worse than neutral —
auto-generated `AGENTS.md` at ~3% **lower** task success for >20% higher inference cost; human-written
~4% better at up to 19% more cost.

The plan answers with structure rather than dismissal: eight files at ≤30 lines, the delete test as a
review gate, and the bulk of the folder documentation moved to a generated doc that costs no agent
tokens at all.

**What would change the recommendation.** If the audience is humans reading the repository rather
than agents acting on it, the whole `AGENTS.md` layer is unnecessary and Phase 5's generated map is
the entire answer — it lives in `docs/architecture/` beside the existing C4 diagrams and costs
nothing on any invocation. Given §2.2, that is a live possibility rather than a rhetorical one, and a
reviewer should feel free to take Phase 1 plus Phase 5 and drop Phases 2-4.

## 8. Authoritative counterparts

- `features.yaml` — F-053
- `scripts/validations/F-053.sh` — the harness-executed proof
- root `CLAUDE.md` + `AGENTS.md` — the import seam
- `scripts/generate_package_map.py` + `docs/architecture/package-map.md`
- `tests/regression/test_f053_aqa.py` / `test_f053_backwards_compat.py`
- `.claude/skills/agents-md-authoring/SKILL.md`
