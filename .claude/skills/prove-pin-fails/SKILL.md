---
name: prove-pin-fails
description: Prove a regression pin actually detects the change it claims to protect, by reverting the source and asserting the test goes red. Use before merging any new AQA/backwards-compat test.
status: active
---

# Prove the pin fails

A test that cannot fail is not a pin. It is decoration that reads like safety,
and it is worse than no test, because it retires the question.

`.claude/skills/regression-pair-scaffold/SKILL.md` already states the rule —
"confirm the pair can fail by temporarily reverting the change under test" — but
leaves it a manual snapshot / edit / run / restore dance. That is exactly the
kind of step that gets skipped under time pressure, and skipping it is invisible:
a tautological pin is green, and green looks like done.

## When to run it

- Before merging any new `tests/regression/test_<slug>_aqa.py` or
  `test_<slug>_backwards_compat.py`.
- Before merging any test whose whole purpose is to detect a regression — a
  CI-wiring pin, a schema-default pin, a golden-file pin.
- After tightening an assertion in response to review, to confirm the tightening
  actually bit.

## Run it

```bash
bash scripts/prove_pin_fails.sh \
  --from <git-ref-predating-the-change> \
  --paths "<source files the pin protects>" \
  --tests "<pytest targets>"
```

It snapshots the paths, restores them from the ref, runs the tests and requires
them to **fail**, then restores and requires them to **pass**.

Exit codes: `0` proof succeeded · `1` the pin did not fail, so it is not
load-bearing · `2` invocation error · `3` restore failed and the working tree
needs attention.

## Reading the result

**`PROVE-PIN FAIL: the tests PASSED against the reverted source`** is the finding
worth having. It means the assertions do not depend on the change. The usual
causes, in the order they show up:

- **Asserting a value that was already true.** A description-length check passes
  against the old description too; a default-value check on a sibling field that
  never moved.
- **Matching file text rather than behaviour.** A wiring pin that greps a whole
  file matches the explanatory *comment* added in the same commit, so deleting
  the code keeps it green.
- **Asserting the vacuous half of a scenario.** "Nothing egresses by default"
  is true before and after when the parent block defaults to `None`; the claim
  worth pinning is what happens when someone adds a partial block.

## Cautions

- Refuses to run when the target paths have uncommitted changes — the restore
  would discard them. Commit or stash first.
- The `restore` trap fires on `EXIT`, `INT` and `TERM`, and clears the index as
  well as the working tree: `git checkout <ref> -- <paths>` stages its revert,
  so restoring file contents alone would leave the revert staged and shippable.
- Reverting one file of a multi-file change proves only that file carries the
  behaviour. Pass every path the pin depends on.
