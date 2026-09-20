# Change: `mouse-droid-agents-md-directory-docs`

**F-number:** F-053 · **Status:** proposed · **Tree:** `992da04` on the default branch
`claude/markdown-implementation-plan-aVJ2l` (there is no `main`)

**Revision 3.** Revisions 1 and 2 were each reviewed against the tree and each had load-bearing claims
overturned. Revision 3 changes the *deliverable*, not just its justification: the review established
that the folder documentation this change was going to add **already exists**, in the file that already
loads. `peer-review.md` records every correction.

---

## 1. What was asked, and what the tree already contains

The request: add `Agent.md` files in each directory describing that folder's function, with mermaid
diagrams, using subagents, meeting current coding standards.

### 1.1 Partly built already, in two incompatible formats

22 in-package agent docs exist: **16 `agent.md`** (lowercase, singular) and **9 `CLAUDE.md`** (root
plus 8 subsystems), overlapping in 5 subsystems. The repository has already ruled on this, at
`docs/planning/TECH_DEBT_REMEDIATION_PLAN.md` **WS-8d**:

> 22 in-package agent docs in two incompatible formats … 5 directories have **both**; 3 have
> `CLAUDE.md` only; **9** have `agent.md` only. … Root `CLAUDE.md`'s Surface Map indexes only the 8
> `CLAUDE.md` files, so **9 subsystems carry per-directory guidance invisible from the root surface**.
> **Pick one format; index all subsystems.**

**This change executes WS-8d.** That is the decision it is actually making, and revisions 1–2 did not
cite it.

### 1.2 The 16 `agent.md` are persona stubs, largely duplicated, one factually wrong

Each is 13–25 lines, opens *"You are the **World Model Architect**"*, and carries a `Key Files` list.
Three lines are byte-identical in **15 of the 16** — "Protocol-based DI patterns", "No hardcoded
values", "Use structlog … never print()" — all three already in the root `CLAUDE.md`, and the third
already enforced by `ruff` `T20`. None contains a diagram.

`src/mousedroid/llm_gateway/agent.md:6` is false in both halves: it claims "velocity commands via
local LLM", but `src/mousedroid/llm_gateway/protocol.py:65` returns a `GoalVector` and
`src/mousedroid/config/schema/llm.py:96` lists the cloud `anthropic` backend.

### 1.3 The folder documentation the request asks for already exists — in `CLAUDE.md`

**All 8** nested `CLAUDE.md` carry a purpose blockquote at lines 3–4 and a `## Key Files` section. For
example `src/mousedroid/world_model/CLAUDE.md:3` already reads *"Recurrent State Space Model (RSSM)
latent dynamics and Monte Carlo Tree Search (MCTS) trajectory planning for autonomous navigation."*

This is the finding that reshaped the change. Revisions 1–2 planned a sibling `AGENTS.md` per
directory holding exactly that content. It would have shipped **8 duplicated purpose statements** — the
"drift factory" revision 1 rejected by name — while leaving the 8 existing `Key Files` lists ungated.
Two of those are already wrong: `src/mousedroid/llm_gateway/CLAUDE.md:26` names `mock_gateway.py`
(the file is `fallback_gateway.py`) and `src/mousedroid/arm/CLAUDE.md:19` names `mock_arm.py` (it is
`src/mousedroid/arm/hardware/mock_arm_driver.py`).

## 2. Why `AGENTS.md` cannot be the per-directory format here

`AGENTS.md` is the right *standard* — uppercase, plural, Linux Foundation Agentic AI Foundation since
December 2025, read by 30+ agents. It is the wrong *per-directory* vehicle in this repository, for
three mechanical reasons.

**It is unreachable outside 8 directories.** Claude Code reads `AGENTS.md` only "when you have no
`CLAUDE.md` in your working directory **or above it**", and the nested-subdirectory rule sits inside
that same "when none count" block. A root `CLAUDE.md` counts for everything beneath it, so nested
`AGENTS.md` discovery is **off entirely**; the only load path is an `@AGENTS.md` import from a
`CLAUDE.md` in the same directory. Elsewhere it would take two files per package to deliver one file's
content.

**One of the 8 cannot be written to at all.** `src/mousedroid/arm/**` is denied by the repository's own
PreToolUse hook: `.claude/workforce.yaml` `freeze:` sets `frozen_paths: [src/mousedroid/arm/**]` gated
on `F-008`, whose `features.yaml` status is `todo`. Revisions 1–2 scheduled two write tasks inside that
path without noticing, while citing the freeze notice elsewhere in the same design.

**And there is no consumer today.** Verified on #233 and #235: Devin Review reports
`Full review skipped: trial expired and no credits remaining`; CodeRabbit reports `Review skipped`.
No `.coderabbit.yaml`, no Codex or Cursor configuration, nothing under `.github/` consumes
`AGENTS.md`. So for Claude Code, content in `CLAUDE.md` and content in an imported `AGENTS.md` are the
same thing, and no other tool is reading either.

## 3. What this change does instead

**Pick one format: `CLAUDE.md`** — the only reachable one, the one that already holds the purpose
statements, and the one the root Surface Map already indexes. Standards-compliance is handled once, at
the root.

| # | deliverable | scope | why |
|---|---|---|---|
| 1 | `@AGENTS.md` in the root `CLAUDE.md`, plus `"**/AGENTS.md"` in the wheel exclude | 2 lines | the 403-line root `AGENTS.md` is read by **nothing** today — zero `@` imports exist anywhere. One line makes it load, and keeps the tool-neutral file as this repo's public agent surface |
| 2 | A `CLAUDE.md` for the **9** subsystems that have only an `agent.md` | 9 files | closes WS-8d: `agents`, `cognitive`, `comms`, `config`, `experience`, `logging`, `memory`, `safety`, `sensing` currently carry guidance invisible from the root surface |
| 3 | A gate on every nested `CLAUDE.md`'s `Key Files` and named symbols | 1 test | generalises `test_doc_reconciliation_aqa.py::test_orchestrator_claude_md_names_only_real_symbols` (`:191`) from one file to all of them, catching the two known-stale entries and stopping more |
| 4 | `docs/architecture/package-map.md`, generated | all 41 packages | answers "every folder" at zero agent-token cost, with a staleness gate and a mermaid subgraph per package |
| 5 | The 16 `agent.md` removed, content merged | — | WS-8d's "pick one format" |

**No new `AGENTS.md` anywhere but the root.** That is the substantive change from revisions 1–2, and
§1.3 is the reason.

## 4. Success criteria

1. Root `CLAUDE.md` imports `AGENTS.md` with the bare `@AGENTS.md` form, so the root file loads. A
   backticked mention must not satisfy the test — `@path` parsing skips code spans.
2. `pyproject.toml`'s wheel exclude covers all three agent-facing patterns, asserted by a test.
3. Every subsystem under `src/mousedroid/` that carries in-package guidance has a `CLAUDE.md`, and the
   root Surface Map indexes every one. Zero `agent.md` remain.
4. Every symbol and path named in a nested `CLAUDE.md`'s `Key Files` resolves — the two known-stale
   entries fixed, except where the F-008 freeze blocks the write (§5).
5. `docs/architecture/package-map.md` regenerates identically under a normalised comparison that is
   line-ending and path-separator independent, so it holds on `test-windows`.
6. `make gates` and `make test` pass; the three suppression budgets unchanged.

## 5. What this change cannot do, and says so

- **`src/mousedroid/arm/CLAUDE.md:19`'s stale `mock_arm.py` stays wrong.** Fixing it needs
  `MOUSEDROID_WORKFORCE_ALLOW_FROZEN=1`, and F-008 hardware readiness preempts in-flight software
  streams by design. The gate in deliverable 3 records `arm` as a declared exemption with this reason,
  rather than being weakened to pass.
- **No byte-exact size assertion.** `.gitattributes` is absent, so a CRLF checkout changes every
  file's byte count; `test-windows` runs `tests/regression`. Size checks are line counts or
  normalised.
- **The generated map describes the import graph, not dataflow.** In a factory-first DI codebase those
  differ by design: `factory` has 35 outbound package edges and `config` 33 inbound because invariants
  1–2 require it, and the runtime seams run through injected Protocols that `ast` cannot see. The map
  is labelled as an import map, and `TYPE_CHECKING`-only imports are filtered out — three live cases
  sit in `world_model` alone.

## 6. Verified — every row re-checked for revision 3

| claim | evidence |
|---|---|
| WS-8d already rules "pick one format; index all subsystems" | `docs/planning/TECH_DEBT_REMEDIATION_PLAN.md:1492-1500` |
| All 8 nested `CLAUDE.md` carry a purpose blockquote and a `## Key Files` section | per-file check |
| 9 subsystems have `agent.md` and no `CLAUDE.md` | named in §3 |
| `src/mousedroid/arm/**` is frozen; `F-008` status is `todo` | `.claude/workforce.yaml:23-24`, `features.yaml` |
| Zero `@` imports in all nine `CLAUDE.md`; root `AGENTS.md` is 403 lines and read by nothing | per-file grep |
| `pyproject.toml:244` is `exclude = ["**/CLAUDE.md", "**/agent.md"]` — `AGENTS.md` absent | read |
| Nested `AGENTS.md` discovery is off while a `CLAUDE.md` sits above | `code.claude.com/docs/en/memory`, the "when none count" block |
| No non-Claude agent reads this repo | Devin/CodeRabbit skip statuses on #233 and #235 |
| Three boilerplate lines are byte-identical in **15 of 16** `agent.md` | `grep -c` per line |
| `src/mousedroid/llm_gateway/agent.md:6` false in both halves | `protocol.py:65`, `config/schema/llm.py:96` |
| `src/mousedroid/llm_gateway/CLAUDE.md:26` names a file that does not exist | `fallback_gateway.py` is the real one |
| `src/mousedroid/llm_gateway/CLAUDE.md:20-21` names two illegal `fallback_backend` values | `config/schema/llm.py:157` permits `none`, `llama_cpp`, `openai_compatible` |
| 40 of 41 packages have an `__init__.py` docstring; `src/mousedroid/telemetry/__init__.py` is 0 bytes | AST check |
| No mermaid renderer anywhere in the toolchain; 49 fences across 20 files under `docs/`, plus 1 in root `README.md` | grep |
| Renaming the 8 contracts would touch **43 occurrences across 17 files** (40 excluding this bundle) | the cited grep, re-run |
| `tools/doc_hygiene.py` runs against `NEXT_STEPS.md` **only**, so its 20 KB default gates nothing else | `Makefile:143`, `ci.yml:421`, `scripts/ci.sh:65`, pinned by `test_f038_aqa.py:62-63` |
| `.gitattributes` absent | `ls` |

## 7. The evidence against, and the condition that would change the recommendation

Current practice: keep agent instruction files short, delete any line a competent developer would get
right anyway, and do not treat them as architecture documentation. Reported measurements have
auto-generated files at ~3% **lower** task success for >20% more inference cost.

Revision 3 answers this better than its predecessors did, by accident of being corrected: it adds
**no** new per-directory instruction content for the 8 documented subsystems, converts 9 unread stubs
into the read format at roughly their existing size, and puts the bulk — the per-folder description
for all 41 packages — in a generated doc that costs nothing on any invocation.

**What would change the recommendation.** If standards-compliance per directory is the actual goal
rather than folder documentation, the only mechanically sound route is a `CLAUDE.md` + `AGENTS.md` pair
in every package, `arm` excepted, at two files per package. §2 is the argument against; it is a
decision to take deliberately, not a detail.

## 8. Authoritative counterparts

`features.yaml` (F-053) · `scripts/validations/F-053.sh` · root `CLAUDE.md` + `AGENTS.md` ·
`scripts/generate_package_map.py` + `docs/architecture/package-map.md` ·
`tests/regression/test_f053_aqa.py` / `test_f053_backwards_compat.py` ·
`docs/planning/TECH_DEBT_REMEDIATION_PLAN.md` WS-8d
