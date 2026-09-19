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

## 2. The finding that decides the design

**Writing `AGENTS.md` files into this repository, today, produces text Claude Code will not read.**

Claude Code reads `AGENTS.md` natively from v2.1.277. But the default **Project instructions** mode
is `claude-md-or-agents-md`, and the documented behaviour is a table:

| repository has | Claude reads |
|---|---|
| `AGENTS.md`, and no `CLAUDE.md` at or above the working directory | the `AGENTS.md` |
| `AGENTS.md` **and** a `CLAUDE.md` at or above it | **`CLAUDE.md` files only** |
| a `CLAUDE.md` that imports `AGENTS.md` | `CLAUDE.md`, with `AGENTS.md` through the import |

This repository has a root `CLAUDE.md` and **eight** nested ones
(`arm`, `growth`, `hardware`, `learning`, `llm_gateway`, `orchestrator`, `telemetry`,
`world_model`). So it sits in row two: every `AGENTS.md` added would be inert.

Nested discovery has the same shadow. A subdirectory's `AGENTS.md` attaches when Claude opens a
file there **only if that subdirectory has none of the three `CLAUDE.md` files of its own** — so in
those eight directories a sibling `AGENTS.md` is skipped even after the root is fixed.

**Only one remedy is committable to the repository.** The `claude-md-and-agents-md` setting lives
under the built-in `agents-md` plugin's `pluginConfigs`, and Claude Code **ignores it in project and
local settings files** — user or managed settings only. It cannot be a repo decision, so it cannot
be this change's mechanism. What remains is the `@AGENTS.md` import inside `CLAUDE.md`, which the
documentation names as the way to keep one file every tool shares.

The symlink alternative (`ln -s AGENTS.md CLAUDE.md`) is **ruled out on this repository's own
grounds**: the docs warn that on Windows a symlink needs Administrator or Developer Mode and git
checks a committed one out as plain text unless `core.symlinks` is set, leaving that clone with a
one-line `CLAUDE.md` instead of instructions. This repo runs a `test-windows` CI job that is a
promotion candidate, and it has already absorbed four consecutive rounds of Windows-only failures.
A mechanism that degrades silently on that platform is not available here.

**This is the repository's recurring failure mode, arriving as a request.** F-050 existed partly
because a metric had no writer; F-052 opens on a coverage gate that runs in no workflow. Adding
40+ documentation files that the primary agent does not load would be the same defect, self-inflicted
and at scale. So the wiring lands before any content, and a test proves the wiring.

## 3. Scope

"Each directory" read literally is **147 directories** (`src/mousedroid` 70, `tests` 70, `tools` 3,
`scripts` 2, `training` 2, excluding `__pycache__`). That is not a documentation set, it is a drift
surface — and the docs themselves warn that contradictory instructions across nested files make
Claude "pick one arbitrarily".

A reconciliation audit then found four more constraints that push the scope down further, not up:

| # | constraint | evidence |
|---|---|---|
| a | **`AGENTS.md` is not excluded from the wheel.** `pyproject.toml:244` is `exclude = ["**/CLAUDE.md", "**/agent.md"]`, with a comment explaining it stops 22 internal files shipping to PyPI. ~40 new `src/mousedroid/**/AGENTS.md` would ship. | verified |
| b | **The repo already decided to delete these.** `docs/planning/TECH_DEBT_REMEDIATION_PLAN.md` rules on `agent.md`: *"25-line persona stub whose Key Invariants restate `CLAUDE.md` and `AGENTS.md:10-45`. Already flagged as drift … and never resolved."* Action: **Delete/merge**. It also records that invariants now live in **four** places. | verified |
| c | **The root `AGENTS.md` is already over budget** — 23,125 bytes against `doc_hygiene.py`'s `_DEFAULT_MAX_BYTES = 20_000` (1.16×). Widening that gate to cover more docs is already deferred *because* it would land red. | verified |
| d | **`src/mousedroid/agents/` is scanned by the skill loader.** `harness_mcp.py:394` defaults `markdown_agent_dirs` to that directory, and the loader globs `*.md`. An `AGENTS.md` there is picked up and skipped only by the no-front-matter branch. | verified |

**So the scope splits across two surfaces, by what each is for:**

- **`AGENTS.md` — agent instructions, ~9 files.** The eight directories that already hold a
  `CLAUDE.md` (those are the high-traffic subsystems, and the only ones where a sibling `AGENTS.md`
  is otherwise shadowed) plus `tests/`. Tight, capped, diagram included, wheel-excluded.
- **Folder-function documentation for all 41 packages — one generated map, zero agent tokens.**
  `docs/architecture/package-map.md`, **generated from the import graph** rather than hand-written, so
  it cannot drift and costs nothing on any agent invocation.

That second surface is what actually answers "describing the functions of each folder" for every
folder, and it answers it better than 41 hand-written files: generated content cannot go stale, and
`docs/` is where this repository's other 50 mermaid diagrams already live.

**Out of scope:**

- Sub-packages below the first level (`src/mousedroid/hardware/lidar/`, `tests/unit/world_model/`).
  A package inherits its parent's file through nested discovery; a file per leaf directory buys
  nothing and multiplies contradiction risk.
- **A hand-written `AGENTS.md` in each of the 41 packages.** Refused on constraints (a)-(d) above
  plus the token evidence in §7. The need it serves is met by the generated package map instead. If
  the literal per-package version is wanted anyway, §7 records what it costs.
- `__pycache__`, `.git`, and any generated tree.
- Rewriting the eight nested `CLAUDE.md` contracts. Their invariants stay authoritative; this change
  moves *folder-purpose* description into `AGENTS.md` and leaves *contracts* where they are (design
  D-2).
- The root `AGENTS.md` (403 lines) is not rewritten, only reconciled against the root `CLAUDE.md`.

## 4. Success criteria

1. A test proves Claude Code actually loads the new files: root `CLAUDE.md` contains an
   `@AGENTS.md` import, and each of the eight `CLAUDE.md`-bearing directories imports its sibling.
   The test fails if a directory gains an `AGENTS.md` that nothing imports and nothing would read.
2. Exactly one agent-facing folder-purpose file per in-scope package, named `AGENTS.md`. Zero
   `agent.md` files remain.
3. Every `AGENTS.md` mermaid fence parses. A diagram that does not render is a broken diagram.
4. Every backticked repo path inside an `AGENTS.md` resolves on disk — the rule
   `tools/validate_skill_commands.py` already enforces for skills, extended to these files.
5. No statement in an `AGENTS.md` contradicts the `CLAUDE.md` in the same directory or the root.
6. A new package without an `AGENTS.md` fails a gate, so the set cannot silently rot.
7. `make gates` and `make test` pass; the three suppression budgets are unchanged.
8. Nothing claims a diagram or description that the tree does not support — verified by a subagent
   pass whose brief is to disprove, not confirm.

## 5. Why this is not just "write 45 files"

Two properties have to hold or the work is worse than not doing it.

**Accuracy that survives.** A folder description listing files churns on every commit. The 16
existing `agent.md` files already show the failure: each carries a `Key Files` list, and those lists
are unverified by any test. Descriptions here are written against the folder's **Protocol boundary
and dataflow role**, which changes when architecture changes, not when a file is added.

**Diagrams that mean something.** 45 hand-drawn mermaid diagrams with no gate is 45 future lies.
Each diagram is scoped to the folder's own inputs, outputs and Protocol seam — the same thing the
C4 component docs assert at a higher level — and is parse-gated.

## 6. Verified clean / verified true — recorded so the plan stays falsifiable

| claim | evidence |
|---|---|
| Standard filename is `AGENTS.md`, uppercase | agents.md spec; Linux Foundation Agentic AI Foundation stewardship since Dec 2025 |
| Claude Code reads it natively from v2.1.277 | https://code.claude.com/docs/en/memory, AGENTS.md section |
| With a `CLAUDE.md` present, AGENTS.md is **not** read by default | same page, the three-row table |
| A subdir `CLAUDE.md` shadows a sibling `AGENTS.md` | same page, "When Claude Code reads AGENTS.md" |
| The both-files setting is not committable | same page: "Claude Code ignores it in project and local settings files" |
| `@path` imports: relative to the importing file, max 4 hops, backticked paths not imported | same page, "Import additional files" |
| Symlink degrades on Windows | same page, "Share one file with other coding tools" |
| 16 `agent.md`, 9 `CLAUDE.md`, 1 root `AGENTS.md` | `git ls-files` |
| Zero mermaid in any `agent.md`; none under `src/` | grep for fences |
| 147 directories if taken literally; 41 top-level packages | `find` |

## 7. The evidence against doing this, and why the plan still proceeds

The standards search turned up a finding that argues against the request as literally stated, and it
would be dishonest to bury it in a design section.

**Current practice says these files should be short and sparse, and that folder descriptions are the
named anti-pattern.** The prevailing test for any line is: *would a competent developer who had never
seen this repo get this wrong?* If no, delete it — it costs context tokens on every invocation. The
guidance is to stay under ~150 lines, and the explicitly named mistake is "treating AGENTS.md like
documentation with long explanations, architecture philosophy, design rationale, and project
history." "Describing the functions of each folder" is close to that description.

**There is measured evidence of harm.** Reported results: auto-generated `AGENTS.md` files *reduce*
task success by ~3% while raising inference cost over 20%; human-written ones improve success only
marginally (~4%) at up to 19% more cost. Neither figure is a ringing endorsement of adding 45 files.

**No standard or best-practice source backs mermaid in `AGENTS.md`.** The spec is silent on diagrams;
the best-practice writing does not mention them. The argument for including them here is mine, not
cited: a six-line flowchart states a dataflow more cheaply than the paragraph it replaces, so it can
*reduce* tokens rather than add them — but that is reasoning, not evidence, and it is stated as such.

**Why the plan proceeds anyway, and what it changes in response.**

1. **Nested discovery makes the cost local, not global.** A per-package file attaches only when
   Claude opens a file in that package. The per-session tax is the root file alone. That is the
   structural difference between 45 files and one 45-section file, and it is why the D-4 scope rule
   matters beyond tidiness.
2. **The delete test becomes a gate, not advice.** Task 3.3 forbids restating anything a competent
   developer would already do — which is what makes the 16 existing `agent.md` files low-value today:
   "Use structlog for logging, never print()" is already in the root `CLAUDE.md` and in `ruff`'s `T20`
   rule, so restating it per folder is pure cost.
3. **A hard line budget.** 40 lines per file, diagram included — well inside the ~150 the guidance
   suggests, and a quarter of what the existing root `AGENTS.md` spends.
4. **The diagram must earn its tokens.** It replaces prose rather than supplementing it. A file
   carrying both a flowchart and a paragraph describing the same flow fails review.

**What would change my recommendation.** If the intent is documentation for humans reading the
repository, `docs/architecture/` is the better home and already holds the C4 diagrams — no token cost
on any agent invocation. `AGENTS.md` is the right home only if the intent is instructions agents act
on. The plan assumes the latter because the request named `Agent.md`; if it is the former, say so and
the same content lands in `docs/architecture/` with a much simpler plan.

## 7. Authoritative counterparts

- `features.yaml` — F-053
- `scripts/validations/F-053.sh` — the harness-executed proof
- `CLAUDE.md` + `AGENTS.md` (root) — the import seam
- `tests/regression/test_f053_aqa.py` / `test_f053_backwards_compat.py`
- `.claude/skills/agents-md-authoring/SKILL.md` — the repeatable procedure (design D-6)
