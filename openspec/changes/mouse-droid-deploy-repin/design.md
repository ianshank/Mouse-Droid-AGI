# Design — mouse-droid-deploy-repin (F-035)

## D-1 — Why a new script rather than extending `archive_stale_branches.sh`

The sibling's protection subsystem exists to *decline* an action. Folding tag
creation into it would give one script two opposite postures — "delete unless
protected" and "create protection" — behind one `--push` flag, so an operator
approving a cleanup would silently also be approving remote tag writes. Two
scripts, two confirmations.

They stay coupled by contract, not by code: the cross-script test asserts the
sibling's output flips from "protecting its carriers" to "carriers not needed"
after this script runs.

## D-2 — Naming by full SHA, not per-feature

F-028 and F-029 share a pin. Per-feature names would create two tags on one
object and a rename obligation whenever a feature is renamed. SHA-based naming
means one tag per object, no bookkeeping, and the tag name is self-verifying.

Cost: the tag name no longer says *why* the object matters. Mitigated by making
the tags annotated — the message carries the reason.

## D-3 — Format validation is not redundant with `git cat-file -e`

This is the design decision most likely to be read as belt-and-braces, so it is
recorded explicitly with its counter-example.

`git cat-file -e` is a *reachability* check and accepts far more than a SHA:
branch names, `HEAD`, `main~3`. A deploy pin of `"main"` therefore passes it and
would be tagged as though it were the pinned commit — protecting the wrong
object while the real pin stays exposed, and leaving behind a tag whose name
asserts something false. Only the 40-lowercase-hex check catches that.

The `features.yaml` extraction does not need this (its sed pattern matches 40
hex chars or nothing), but the JSON extraction has no format guarantee
whatsoever. Validation is applied uniformly anyway, so a future change to
either extractor cannot quietly remove the guard.

## D-4 — An unresolvable pin exits nonzero

The alternative — skip it and carry on — produces a "0 tags to create" summary
for a repo whose pins are already lost. Silence that reads as success is the
failure mode this whole feature exists to correct, so the script must not
reproduce it.

## D-5 — What the tests initially got wrong

Recorded because the correction is the substantive part.

The first version of the format-rejection tests asserted only
`returncode != 0` and `"ERROR" in stderr`. Both hold when the format check is
deleted, because a malformed value also fails the later `cat-file` resolve
check. Verified by reverting `_is_full_sha` to `return 0`: **5 of 6 security
tests stayed green**, i.e. they were pinning a different check than the one
they named. Only the branch-name case went red.

Tightened to assert the specific format-check message. Re-proven: 5 of 5 now go
red without the check, and green with it, with the script restored
byte-identical in between.
