# Peer review: `mouse-droid-agents-md-directory-docs` (F-053)

Adversarial record for the plan, written before any of it is built. It reviews the findings and the
reasoning, not code.

Method: a standards search (the `AGENTS.md` specification, its stewardship, Claude Code's own
loading behaviour, and current best practice), plus direct recon of the tree at `992da04`. Every
load-bearing claim was re-verified against the repository or against primary documentation before it
entered `proposal.md`.

---

## What the review changed about the plan

### The request is partly already built, under a filename nothing reads

The plan opened as "add per-directory agent files". Recon found **16 `agent.md` files already
present**. They are lowercase and singular, so no agent tooling reads them; they are persona prompts
rather than folder documentation; they are 13–25 lines; and none contains a diagram. So the change is
a **rename, split and gate** of an existing half-built layer, not a greenfield addition. That reframing
is what produced Phase 5 and D-7.

### The plan would have shipped 45 files that Claude Code never loads

This is the finding the design now turns on. Claude Code reads `AGENTS.md` natively, **but only when
no `CLAUDE.md` exists at or above the working directory**. This repository has nine. And a
subdirectory's `AGENTS.md` is skipped when that subdirectory has a `CLAUDE.md` of its own — which
eight of them do.

Then the obvious fix turned out to be unavailable: the `claude-md-and-agents-md` setting is **ignored
in project and local settings files**. It cannot be committed, so it cannot be this repository's
decision. That eliminated an option on documented grounds rather than preference, and left the
`@AGENTS.md` import as the only committable mechanism.

The confirming observation is that this is not hypothetical: the repo already has a **403-line root
`AGENTS.md` and zero `@` imports in any of its nine `CLAUDE.md` files**. That file is dead text
today. The defect the plan guards against is already present, at a scale of one.

### A correction made mid-research, because the first source was wrong

My first fetch — the `anthropics/claude-code` `mods/agents-md` page — states that Claude Code "does
not read `AGENTS.md` natively" and that a plugin adds it. I reported that. The official memory
documentation then contradicted it: native reading landed in v2.1.277. The mod page describes a
plugin, not current engine behaviour.

Recorded because the wrong version would have produced a worse plan — it implied a plugin dependency
that does not exist, and would have hidden the real constraint, which is the default-mode table
rather than a missing capability. Two sources, and the one nearer the product won.

### The standards search argued against the request, and that is in the proposal, not buried here

Current practice: keep these files short (~150 lines), delete any line a competent developer would get
right anyway, and do not treat the file as documentation with architecture description — which is
approximately what "describe the functions of each folder" asks for. Reported measurements are worse
than neutral: auto-generated files ~3% *lower* task success at >20% higher inference cost;
human-written ~4% better at up to 19% more cost.

The plan proceeds because it was asked for, and answers with structure rather than dismissal: nested
discovery localises the cost to the package being read, a 40-line cap per file, and the delete test
promoted from advice to a gate (D-9, task 4.0). `proposal.md` §7 also names the condition under which
I would recommend something else — if the audience is humans rather than agents, `docs/architecture/`
is the right home and costs no tokens at all.

No standard or best-practice source supports mermaid in `AGENTS.md`. The argument that a compact
flowchart is cheaper than the prose it replaces is mine, and is labelled as reasoning rather than
evidence.

---

## Disproved — investigated and kept out of the plan

| claim | why it is wrong |
|---|---|
| "The 16 `agent.md` files are unwired, so deleting them is free" | Three in-repo references exist: `test_doc_reconciliation_aqa.py:46`, `test_ci_gate_wiring_aqa.py:825`, `src/mousedroid/skills/loaders.py:7,103`. A blind delete turns a green suite red. I asserted the unwired version in conversation before checking, and it was wrong. |
| "Add a file to each directory, as asked" | 147 directories, including `__pycache__` and every test subdirectory. Nested discovery means a sub-package already inherits its parent's file, so the extra 102 files add contradiction surface and no information. |
| "Symlink `CLAUDE.md` → `AGENTS.md`, it is one command" | Documented to degrade on Windows — needs Administrator or Developer Mode, and git checks a committed symlink out as plain text without `core.symlinks`, leaving that clone with a one-line `CLAUDE.md`. This repo runs `test-windows` and has absorbed four rounds of Windows-only failures. |
| "Set `claude-md-and-agents-md` in project settings" | Ignored in project and local settings files. |
| "The root `CLAUDE.md` line budget blocks importing a 403-line file" | The import adds one line to the importing file; `docs.core_max_lines: 250` measures `CLAUDE.md` itself. Task 1.6 verifies rather than assumes, because `docs_trimmer` is what fails `local-gates`. |

---

## Open weaknesses in this plan

1. **The premise is documented, not yet observed.** Phase 0 exists because every claim about what
   loads comes from documentation plus the absence of imports in the tree — not from watching a real
   session report its instruction files. If 0.2 shows Claude already reading `AGENTS.md` here, D-1
   through D-3 need rewriting rather than implementing. That task is first for that reason.
2. **The token-cost objection is not fully answered, only bounded.** Nested discovery localises the
   cost and the 40-line cap limits it, but the honest position is that this layer makes agent
   invocations in this repo slightly more expensive for a benefit the cited measurements put at a few
   percent. If that trade is unwanted, the scope to cut is the 22 packages that have neither an
   `agent.md` nor a `CLAUDE.md` today — they are the ones nobody has yet felt the absence of.
3. **45 hand-written descriptions will drift, gates or no gates.** The parse gate catches broken
   diagrams and the path gate catches dead references, but neither catches a description that is
   merely *stale* — accurate prose about last month's architecture. D-1's "write against the Protocol
   boundary, not the file list" reduces the churn rate; it does not eliminate it. The only real
   defence is that a stale folder description is visible to whoever next reads that folder, which is
   weaker than a test.
4. **Task 3.6 may be unimplementable as stated.** "No `AGENTS.md` restates an invariant from the
   sibling `CLAUDE.md`" is easy to state and hard to check — paraphrase defeats substring matching. If
   it cannot be made to work without false positives, it should become a review checklist item in the
   skill rather than a gate that everyone learns to work around. Decide that in Phase 3, and record
   which way it went.
5. **The mermaid render gate is new capability, not a port.** `docs/architecture/` already carries
   ~45 diagrams that nothing validates. Building the gate here and pointing it only at the new files
   is defensible scope, but it does mean the repo's larger diagram surface stays unvalidated, and task
   4.5 records that rather than quietly leaving it.
6. **Scope is a judgement the evidence narrowed but did not settle.** The request said "each
   directory" — 147. The first draft said 45. The audit drove it to 9 plus a generated map. Nine is
   defensible (they are exactly the directories where the import seam is load-bearing) but it is not
   what was asked for, and the generated map is my substitution for the rest. If the literal
   per-package version is wanted, `proposal.md` §7 and D-4 record what it costs; the plan should be
   re-scoped deliberately rather than quietly drifting back up.

### The audit found a defect the plan itself would have introduced

The reconciliation pass over the existing surfaces returned four constraints, all verified, and the
first is the one worth naming plainly: **`pyproject.toml:244` excludes `**/CLAUDE.md` and
`**/agent.md` from the wheel but not `AGENTS.md`**, with a comment explaining that without it 22
internal files ship to PyPI. The plan as first drafted would have added ~40 more files under
`src/mousedroid/` and shipped every one of them. That is now task 1.0, first in the phase, because it
is the only task whose omission does harm outside this repository.

The other three narrowed the scope from 45 files to 9:

- **The repository had already decided.** `docs/planning/TECH_DEBT_REMEDIATION_PLAN.md` rules on
  `agent.md` — persona stub restating `CLAUDE.md` and `AGENTS.md`, "already flagged as drift … and
  never resolved", action **Delete/merge** — and records that invariants now live in **four** places.
  A plan that added a fifth ungated surface would have contradicted a written decision without
  acknowledging it. Phase 5.0 now executes it and updates that table.
- **The root `AGENTS.md` is already 1.16× over budget** (23,125 bytes vs `doc_hygiene.py`'s 20,000),
  and widening that gate is deferred *because* files like it would fail. So this change may shrink
  that file; it may not grow it (task 2.5).
- **`src/mousedroid/agents/` is scanned by the skill loader** (`harness_mcp.py:394` globs `*.md`), so
  an `AGENTS.md` there interacts with skill registration (task 2.3).

It also found that one of the 16 files is **actively wrong**, which changes Phase 5 from "migrate" to
"fix, then migrate": `src/mousedroid/llm_gateway/agent.md:6` says "Natural language to velocity commands via local
LLM", while `protocol.py:65` returns a `GoalVector` and `src/mousedroid/config/schema/llm.py:96` lists the cloud
`anthropic` backend. Both halves false. Moving that line into a new file would have laundered it.

### Pre-existing defects found in passing, recorded not fixed here

None of these is this change's to fix, and all are worth a follow-up because they are the same genre
this change is about:

| finding | evidence |
|---|---|
| `src/mousedroid/llm_gateway/CLAUDE.md:26` names `mock_gateway.py`, which does not exist; the real second backend is `fallback_gateway.py` | verified absent |
| `src/mousedroid/llm_gateway/CLAUDE.md:20-21` says `fallback_backend` must target `mock` or `ollama`; `src/mousedroid/config/schema/llm.py:157` permits only `none`, `llama_cpp`, `openai_compatible` — neither named value is legal | verified |
| `src/mousedroid/arm/CLAUDE.md:19` names `mock_arm.py`; the file is `src/mousedroid/arm/hardware/mock_arm_driver.py` | verified |
| Root `AGENTS.md` points twice at `CLAUDE.md` sections that do not exist ("Test surface mirror", "Live deployment + CI-gate contracts") | verified absent |
| Root `AGENTS.md:139` implies 8 subagents ("any of the other five"); `.claude/agents/` holds 7, and `SKILLS.md:802` says 7 | verified |
| Root `CLAUDE.md:20-21` names 6 test tiers, `SKILLS.md:817` says 7, `tests/` holds 11 directories | verified |

The last two matter to this change indirectly: once task 1.1 lands, root `AGENTS.md` and root
`CLAUDE.md` load **together**, so a contradiction between them stops being dormant. Task 6.4 covers
reconciling the pair; these specific rows are the known content.

---

## Sign-off condition

Accepted when a reader can take any line marked "verified" in `proposal.md` §6, run the command or
open the cited file, and get the stated result — and when Phase 0.2 has confirmed by observation the
loading behaviour that the rest of the plan assumes.
