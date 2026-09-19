# Peer review: `mouse-droid-agents-md-directory-docs` (F-053)

Adversarial record. Revision 1 was written, then reviewed against the tree; **three of its
load-bearing claims were wrong, including the reason the change existed.** This file records what
they were, because a plan whose premise silently changed is worth less than one that shows the
correction.

---

## Revision 1 → 2: what the review overturned

### 1. The justification was misleading, and it was the whole argument (SEVERE)

Revision 1's D-1 read: *"This repository already runs two non-Claude reviewers on its pull requests —
Devin Review and CodeRabbit both posted on #233 — so content that lives only in `CLAUDE.md` is
invisible to the tools reviewing the code."*

Verified on both #233 and #235:

| check | actual description |
|---|---|
| Devin Review | `Full review skipped: trial expired and no credits remaining` |
| CodeRabbit | `Review skipped: draft pull request`; on #233 also `manual review required for this OSS repository` |

They post skip statuses. Neither reviews. There is no `.coderabbit.yaml`, no Codex or Cursor
configuration, and nothing under `.github/` consumes `AGENTS.md`.

I had the evidence in hand when I wrote the claim — I read those exact status descriptions earlier in
the same session while checking CI — and still wrote "both posted on #233" as though posting were
reviewing. The corrected §2.2 states the benefit as optionality, which is what it is.

### 2. The reachability reasoning was backwards (SEVERE)

Revision 1's D-4 said the eight `CLAUDE.md` directories are *"exactly the directories where a sibling
`AGENTS.md` would otherwise be shadowed by their `CLAUDE.md`"*.

The documentation puts the nested-subdirectory bullet **inside** the "when none count" block, and a
root `CLAUDE.md` counts for every subdirectory beneath it. So nested `AGENTS.md` discovery is **off
entirely** while a root `CLAUDE.md` exists — and those eight are not where `AGENTS.md` is shadowed,
they are the **only** places it can load at all.

The scope conclusion survived; the reason for it was inverted. That matters because the wrong reason
implied the other 33 packages were merely lower priority, when in fact an `AGENTS.md` there is
unreachable without a second file per package.

### 3. The plan was about to ship a file nothing could read (SEVERE)

Revision 1's nine-file set included `tests/AGENTS.md`. `tests/CLAUDE.md` does not exist, so under
finding 2 that file had no load path.

**The plan written to stop unreadable documentation was about to produce some.** Task 2.2's gate —
"every `AGENTS.md` must have a same-directory `CLAUDE.md` importing it" — would have caught it, which
is the argument for writing the gate before the content. `tests/` is now an explicit decision (task
2.6) rather than an implied one.

### 4. The mermaid gate was a CI dependency dressed as a test (MAJOR)

Revision 1's task 4.2 asserted "every fence parses and renders". There is no `mermaid` or `mmdc`
reference in `pyproject.toml`, `.github/workflows/ci.yml`, `scripts/ci.sh` or the `Makefile`, and no
Python mermaid parser installed. Implementing it means adding a Node toolchain to a Python-only CI.

The replacement is better than the original intent: the generator emits all 41 diagrams from the
parsed import graph, so a diagram cannot be wrong, one regenerate-and-diff test covers every one, and
no renderer is needed. Revision 1 offered 9 unchecked diagrams; revision 2 offers 41 provable ones.

### 5. The rejected alternative was rejected without measuring it (MAJOR)

Revision 1 rejected renaming the eight nested `CLAUDE.md` to `AGENTS.md` on grounds of drift risk,
which is hand-waving. Measured: ~25 references — root `CLAUDE.md`'s Surface Map (8),
`docs/claude/surfaces/README.md` (8), `CHANGELOG.md` (4),
`src/mousedroid/config/schema/hardware.py`, `scripts/validations/F-031.sh`, three openspec bundles,
and a path-specific test. `CHANGELOG.md` and the openspec bundles are historical records. The
rejection stands, now on a number.

### 6. A gate was proposed that cannot be written (MODERATE)

Revision 1's task 3.6 gated "no `AGENTS.md` restates an invariant from the sibling `CLAUDE.md`".
Paraphrase defeats substring matching, and a gate everyone learns to route around is worse than a
checklist line. Downgraded to a review item in the skill, recorded as a deliberate downgrade rather
than dropped.

---

## What survived review unchanged

| claim | why it held |
|---|---|
| `pyproject.toml:244` omits `**/AGENTS.md`, so the plan would have shipped internal files to PyPI | verified; now task 1.1, first in the plan |
| `docs/planning/TECH_DEBT_REMEDIATION_PLAN.md` already rules `agent.md` **Delete/merge** | verified; Phase 4.1 executes it and updates that table |
| Root `AGENTS.md` is 23,125 bytes against a 20,000-byte budget | verified |
| `src/mousedroid/llm_gateway/agent.md:6` is false in both halves | verified against `protocol.py:65` and `src/mousedroid/config/schema/llm.py:96` |
| 40 of 41 packages carry an `__init__.py` docstring | verified; only `src/mousedroid/telemetry` lacks one, so the generator has exactly one pre-existing case to fix |
| The `claude-md-and-agents-md` setting is not committable | verified in the docs |
| Symlink degrades on Windows | verified in the docs, against this repo's `test-windows` history |

Phase 0 of revision 1 ("observe which instruction files load") is **deleted, satisfied**: the session
that wrote this plan received `CLAUDE.md`'s contents and not `AGENTS.md`'s. The premise was proved by
the act of writing, which is why it took a deliberate pass to notice.

---

## Open weaknesses in revision 2

1. **The change may not be worth making at all, and the plan now says so.** §2.2 establishes the
   present-day delta is zero. Phase 1 (one import line, one exclude pattern) is unambiguously worth it.
   Phases 2-4 buy optionality for a tool nobody here runs yet. A reviewer taking Phase 1 + Phase 5 and
   dropping the middle would be making a defensible call, and §7 says that plainly rather than burying
   it.
2. **The generated map is a substitution, not the thing asked for.** The request was per-directory
   files. Phase 5 answers "what does each folder do" for all 41 at zero token cost and with a staleness
   gate, which I judge better — but it is my judgement replacing an instruction, and it should be
   accepted or rejected as such.
3. ~~**The 30-line cap is asserted, not tested.**~~ **Settled during this review.** Drafted against
   `src/mousedroid/world_model` — the hardest case, two engines and the most behaviour — the reference
   shape in D-9 comes to **25 lines**, so the cap has five lines of headroom on the worst package.
   Recorded because the weakness was real when written and the fix was one draft away, which is the
   argument for drafting rather than estimating.
4. **Phase 5's size risk is unquantified.** 41 sections plus 41 diagrams against a 20 KB budget
   (task 5.7). If it overruns, the split-per-epic fallback changes the shape of the deliverable and
   nobody has estimated which way it goes.
5. **`src/mousedroid/telemetry` has no `__init__.py` docstring**, and it is the largest package (37
   `.py` files) with an existing 30-line `CLAUDE.md`. Writing its purpose line is the one piece of
   Phase 5 that is authorship rather than generation, and it is on the package where getting it wrong
   is most visible.
6. **Three revisions of the same bundle in one session is itself a signal.** Each revision was driven
   by re-reading a source I had already read. The lesson is the one this repository keeps relearning:
   the defect is rarely in the code, it is in the claim about the code, and claims survive review by
   sounding settled.

---

## Sign-off condition

Accepted when a reader can take any row of `proposal.md` §6, run the command or open the cited file,
and get the stated result — **and** when task 2.2's gate has been proven to fail before task 2.1 lands,
because that gate is the only thing standing between this plan and the defect it was written to prevent.
