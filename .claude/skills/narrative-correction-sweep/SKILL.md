---
name: narrative-correction-sweep
description: Correct a false claim across every live prose surface, not just the ones you remember writing it in — a whole-file, whitespace-normalized sweep over git-tracked docs, proven against the exact original wording before it counts as fixed.
status: active
---

# Narrative Correction Sweep

A stale claim rarely lives in one file. Fixing the copy you remember and
calling the class "corrected everywhere" is how the same falsehood survives
in the copies you didn't.

## Why this exists

Round 4 of the F-028/F-029 review found `tests/security/` described as "the
only coverage" of the pre-egress injection filter in three places — refuted
(`tests/unit/security/test_injection_filter.py` already carried 11 tests of
unit coverage for that same filter) after an earlier round had already
fixed that exact claim and reported it corrected.
The fix landed in the files that round remembered, not the files that had
it: `proposal.md` and `peer-review.md` still asserted it, and the
peer-review's own verdict table marked it **CONFIRMED** — a governance
record asserting a falsehood as verified, which is worse than an
uncorrected comment, because it is what the next engineer reads as settled.
A line-oriented re-sweep still missed `proposal.md`, because the claim
wrapped across a line break; only a whitespace-normalised, whole-file sweep
found all three.

F-030's own doc-reconciliation pins repeated the lesson from the other
direction: `test_ci_gate_wiring_aqa.py`'s `_DOC_GLOBS` was a five-pattern
directory roster (`*.md`, `docs/**/*.md`, `openspec/**/*.md`,
`.claude/**/*.md`, `src/**/*.md`) that silently missed 15 git-tracked docs —
including `tests/agent.md`, a file this exact sprint corrected a stale claim
in. A sweep is only as complete as its file list; a hand-maintained list of
prefixes drifts the same way the narrative it is meant to catch does.

## The procedure

1. **Source the file list from git, not memory or a glob roster.**
   `git ls-files -- "*.md"` (or the relevant extension) is the actual set of
   tracked prose surfaces — it needs no directory enumeration and cannot
   drift behind a new subsystem doc the way a hardcoded prefix list can.

2. **Normalize whitespace before matching.** A claim that reads as one
   sentence in a rendered doc can be hard-wrapped across two source lines.
   Collapse runs of whitespace (including newlines within a paragraph) before
   searching, or a line-oriented grep will miss exactly the wrapped instance
   that a rendered read would catch.

   ```bash
   python - <<'PY'
   import re, subprocess

   claim = re.compile(r"only coverage", re.IGNORECASE)  # the falsehood, in its own words
   files = subprocess.run(
       ["git", "ls-files", "--", "*.md"], capture_output=True, text=True, check=True
   ).stdout.splitlines()
   for path in files:
       text = open(path, encoding="utf-8").read()
       normalized = re.sub(r"\s+", " ", text)
       if claim.search(normalized):
           print(path)
   PY
   ```

3. **Every hit gets a decision, not a delete.** A verdict table or a
   proposal *should* quote the claim it refutes — stripping the sentence
   entirely erases the record of what was wrong. Either qualify it (name the
   real scope: "unit coverage in `test_injection_filter.py`", not bare
   "coverage") or mark it explicitly refuted. Bare prose is what reads as
   current truth to the next reader; a blacklist of phrases is not the
   fix — a *qualified* claim should still match the search and should still
   pass, because pinning on presence-of-phrase alone reintroduces the
   line-oriented false positive this procedure exists to avoid.

4. **Prove the pin against the exact original wording**, not a paraphrase —
   feed the search pattern the literal sentence that shipped, then confirm
   the fixed files no longer match and the still-wrong ones do. A pattern
   tuned only against invented examples has never been tested against the
   defect it claims to catch.

## Guardrails

- **Tune against your own new text before trusting the pin.** Both numeric
  regressions added in F-030 (`test_ci_job_count_docs_match_the_real_workflow`,
  `test_no_live_doc_claims_a_stale_src_coverage_floor`) produced false
  positives on first write — one matched an unrelated true claim ("5 jobs run
  *(advisory)*"), the other flagged the sprint's own correctly-qualified
  "85% for `tools/claude_hooks/`" text. Both were caught only by running the
  new pin against the tree immediately, before treating it as done.
- A sweep that only searches file **paths** you already suspect is not a
  sweep — it is the same memory-based miss this skill replaces. Always
  derive the file list from `git ls-files`, never from a remembered list.
- Once the sweep is clean, add a permanent regression pin (see
  `tests/regression/test_doc_reconciliation_aqa.py` and
  `TestOrphanTierNarrativeAccuracy` in `test_ci_gate_wiring_aqa.py`) so the
  same claim re-drifting later fails loudly instead of waiting for the next
  manual sweep.

## When to run it

- After correcting any factual claim (a count, a coverage floor, a symbol
  name, "the only X") that could plausibly have been copied into more than
  one doc, proposal, or peer-review verdict.
- Before marking a review finding fixed "everywhere" — that phrase is the
  trigger, not the conclusion.
- As part of closing out any `openspec/changes/*/peer-review.md` verdict
  table entry that quotes a claim being refuted.
