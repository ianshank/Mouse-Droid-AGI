---
name: deploy-repin
description: Create the annotated tags that keep gate-critical pinned SHAs reachable, so a branch cleanup cannot orphan them. Use after pin-reachability-audit reports an unprotected pin, before running archive_stale_branches.sh --push, and after any squash-merge that lands a new features.yaml pin.
status: active
---

# Deploy Repin

Turning "this pin is unprotected" into "this pin is tagged". The remedy half
of a pair whose diagnostic half already exists.

## Why this exists

Three things in this repo touch pin reachability, and until now the third was
missing:

| | What it does | |
|---|---|---|
| `.claude/skills/pin-reachability-audit/SKILL.md` | **Diagnoses**: which pins have zero remote carriers and zero remote tags | had no remedy |
| `scripts/archive_stale_branches.sh` | **Defers**: refuses to delete a branch while it is the last carrier of a pin | protection is a side effect of *not* cleaning up |
| `scripts/repin_tags.sh` | **Fixes**: tags the SHA so the carrier branch stops mattering | this skill |

The audit skill can tell you a pin is one branch deletion away from being
lost. Before `repin_tags.sh` existed, the only response was to keep the stale
branch around forever — which is why `archive_stale_branches.sh` carries a
whole protection subsystem whose entire job is to *not* do its job.

The concrete case that motivated it: `deployments/jetson-image.json`'s own
`notes` field asserted its SHA was "kept reachable by the annotated tag
`deployments/jetson-image-032942b`, NOT by any branch." That tag never
existed — `git ls-remote --tags origin` returned zero tags repo-wide — and the
SHA is not an ancestor of the default branch either. Its real reachability was
a handful of stale feature branches, i.e. exactly the set
`archive_stale_branches.sh` is built to delete. A documented protection that
does not exist is worse than a known gap, because nobody goes looking.

## Run it

Dry run first, always — it is the default and it mutates nothing:

```bash
bash scripts/repin_tags.sh
```

It reports, per pin, either the remote tag already covering it or the tag it
would create. Two families, both derived from the repo rather than hardcoded:
`deployments/jetson-image.json`'s deploy pin, and every `implemented_in` in
`features.yaml`.

To actually create and push the tags:

```bash
bash scripts/repin_tags.sh --push
```

Then confirm the sibling script agrees the pins are safe:

```bash
bash scripts/archive_stale_branches.sh
```

Pins that were previously reported as "NOT reachable from any REMOTE tag --
protecting its carriers" should now read "carriers not needed", and the
branches that were only being kept for them become eligible for archiving.

## `--push` is an operator action

`--push` writes tags to a shared remote. Treat it the way this repo treats
every other irreversible remote write: **do not run it on your own
initiative.** Get explicit confirmation for that specific run, even if a dry
run was already approved. Annotated tags are also not meant to be deleted, so
a wrong tag is permanent litter rather than a mistake you can quietly undo.

Dry-running is always fine and needs no confirmation.

## What it deliberately does not do

- **It does not retire tags.** Re-pinning a feature leaves the old SHA's tag
  in place. That is accepted permanent litter, not an oversight: the tag's
  whole purpose is keeping an object reachable, and the object may still be
  referenced from history that outlives the pin.
- **It does not rename to match a convention.** A SHA already reachable from
  *any* tag the remote publishes is left alone, even under someone else's
  name. The goal is reachability, not tidiness.
- **It does not trust a local tag.** Only tags the remote actually publishes
  count — the same rule `scripts/archive_stale_branches.sh` applies, for the
  same reason: a local-only tag makes a pin look safe in your clone while the
  remote would still lose the commit.

## Reviewing a change to it

The script turns file contents into git arguments, so two properties matter
more than the rest:

1. **Every extracted SHA is re-validated against 40 lowercase hex chars
   before use.** `git cat-file -e` is not a substitute — it happily resolves
   branch names, `HEAD`, and `main~3`, so a malformed deploy pin would tag
   whatever that ref points at *as though it were the pinned commit*. The
   `features.yaml` extraction is structurally constrained by its own sed
   pattern; the JSON one is not.
2. **A pin that does not resolve is a nonzero exit, not a skip.** Reporting
   "0 tags to create" for a lost commit reads as success.

Both are pinned in `tests/unit/scripts/test_repin_tags.py`, including the
case where the pin names a branch that git resolves fine. When changing the
validation, verify the tests still go red without it — the first version of
those tests asserted only a nonzero exit and passed with the format check
deleted, because malformed values also fail the later resolve check.
