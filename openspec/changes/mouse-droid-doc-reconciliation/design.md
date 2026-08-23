# Design — Documentation reconciliation across live governance surfaces

## D-1. A new `docs-governance` capability, not a reuse of `dev-governance`

`openspec/changes/mouse-droid-claude-workforce/specs/dev-governance/spec.md` already owns
"Evidence-Backed Claims" and "Truthful Coverage Claims" requirements — thematically close to
this change. It was not reused, because its scope is the workforce tooling's *own* claims:
evidence chains for hardware benchmarks, `.claude/workforce.yaml` as the single config
source, `tools/claude_hooks` coverage. F-030's six drifts are about CI job identity,
orchestrator class names, a cross-file numeric coverage floor, and a cognitive-pillar
wiring narrative — none of which are workforce-tooling claims. Folding them into
`dev-governance` would conflate two different verification surfaces under one capability
name; a new `docs-governance` capability keeps "is the workforce tooling honest about
itself" and "does a live doc match the tree it describes" separately traceable.

## D-2. Catalog pins are discovery-based, not a hardcoded expected-list

`test_every_skill_directory_is_mentioned_in_the_index` globs `.claude/skills/*/` and
`test_every_agent_is_listed_in_the_subagent_skills_table` globs `.claude/agents/*.md` — both
read the real filesystem rather than asserting against a written-out set of expected names.
A hardcoded list would need editing every time a skill or agent ships, and forgetting that
edit is exactly the silent-rot failure mode being closed: 12 of 17 real skill directories
were already missing from `SKILLS.md` before this fix, without any test noticing, because
nothing compared the index against the directory.

## D-3. Both new numeric regexes needed tuning against the live tree before they could be trusted

`test_ci_job_count_docs_match_the_real_workflow`'s `_TOTAL_JOB_COUNT` pattern initially
risked matching "5 jobs run *(advisory)*" — a true statement naming the advisory subset, not
the total — so it is scoped to only the two phrasings this repo actually uses for a TOTAL:
`"(N Jobs)"` and `"N-job"`. `test_no_live_doc_claims_a_stale_src_coverage_floor`'s `85%`
search is line-scoped rather than whole-file, and treats a line containing `claude_hooks` as
self-qualifying, because a whole-file search would have flagged this sprint's own
correctly-written "85% for `tools/claude_hooks/`" text as a violation of the rule it exists
to state. Both false positives were caught by running the new assertion against the tree
immediately rather than trusting it on the strength of the code alone — the worked example
is recorded in `.claude/skills/narrative-correction-sweep/SKILL.md`'s Guardrails section so
the lesson outlives this one pair of tests.

## D-4. `_DOC_GLOBS` becomes `_tracked_docs()` (git ls-files), not a sixth pattern

`tests/regression/test_ci_gate_wiring_aqa.py`'s pre-existing `TestOrphanTierNarrativeAccuracy`
pin (landed under the F-028/F-029 review, not authored by this change) sourced its file list
from `_DOC_GLOBS = ("*.md", "docs/**/*.md", "openspec/**/*.md", ".claude/**/*.md",
"src/**/*.md")`. That roster silently missed 15 git-tracked `.md` files outright, including
`tests/agent.md` — one of this same sprint's own drift-fix targets (item 3 above), so the gap
was not hypothetical. The tempting fix is to add `tests/**/*.md` as a sixth pattern; that
only pushes the next gap out to whichever directory is missed next, which is the identical
shape of bug `TestOrphanTierWiring` (in the same file) already replaced with discovery for CI
tiers. `_tracked_docs()` now sources the list from `git ls-files -- "*.md"`, which needs no
directory enumeration and cannot drift behind a new subsystem doc.

## D-5. The `growth/` correction touches only the clause that was false

`docs/CHARTER.md` §5 grouped `meta/`, `growth/`, and `scaling/` under one "implemented and
unit-tested... but not yet instantiated by `factory.py`" sentence. Only the `growth/` clause
is false — `src/mousedroid/factory.py:4308` calls `build_growth_coordinator`. `meta/` and
`scaling/` were verified separately (grep: no `build_meta_*` or `build_scaling_*` call site
anywhere in `src/mousedroid/factory.py`) and their "not yet instantiated" claim remains
true, so it was left untouched.
The tempting-but-wrong move here is to also soften or hedge the `meta/`/`scaling/` wording
"while we're in the file" — that would introduce a new unverified claim in the act of fixing
an old one, the same failure mode this whole change exists to close. This discipline (fix the
`growth/` clause specifically, leave the still-true `meta/`/`scaling/` wording alone) is
applied identically in all five places this claim turned out to live — see D-6.

## D-6. The `growth/` fix needed a second pass — self-application of the sweep this change introduces

The initial fix touched only `docs/CHARTER.md` §5, the site named in the original scope
statement. Applying `.claude/skills/narrative-correction-sweep/SKILL.md`'s procedure to that
fix's own claim — rather than trusting a single-file edit generalized — found the identical
"`growth/` not yet wired" wording still live in four more places: `docs/CHARTER.md` §1 and §3
(both contradicting the just-corrected §5 *in the same file*), `README.md`'s cognitive-stack
table, and `docs/architecture.md`'s Level 3d component diagram (repeated once more in
`NEXT_STEPS.md` item 0b, which cites the Level 3d comparison). This is the exact failure mode
`.claude/skills/narrative-correction-sweep/SKILL.md` documents from the F-028/F-029 review
round — a claim corrected in the file an engineer remembers, left standing in the ones they
don't — now caught within F-030's own working set rather than in a later round.

`README.md`'s table needed a new third bucket rather than moving `growth/` into either
existing one: the "Wired into the runtime loop" bucket would overclaim (the coordinator is
default-OFF; nothing runs unless an operator sets `Settings.growth`), and the "not yet wired"
bucket is exactly the claim being corrected. A `growth/`-shaped state — factory-instantiated,
metrics-wired, default-OFF pending a decision — already has a precedent in this repository
(M6, on-device incremental learning), so the new bucket names that precedent explicitly
rather than inventing new vocabulary.
