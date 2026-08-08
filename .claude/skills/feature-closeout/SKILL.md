---
name: feature-closeout
description: Close out a spec-harness feature (F-nnn) — flip status to done, pin implemented_in to a real SHA, run the harness gates, and update the narrative docs
status: active
---

# Feature Closeout

Drive an `F-nnn` feature from "code merged" to "closed in the spec harness".

Use this when a feature's implementation has landed and you are about to mark
it `done` — or when a review asks "is F-nnn actually closed out?".

## Why this exists

The closeout chain has been walked by hand in six consecutive sessions, and
each one deferred the same step: pinning `implemented_in` to a real commit SHA.
A branch name resolves fine while the branch is alive and stops resolving the
moment it is deleted post-merge, which reddens the nightly
`--strict-git` harness job the morning after the merge — far from the change
that caused it.

Step 3 below both fixes and *detects* that class of debt.

## Inputs

- `$ARGUMENTS` — the feature id, e.g. `F-025`. Required.

## Steps

### 1. Confirm the work is actually complete

Read the feature's `verification:` list in `features.yaml`. Every bullet must
map to a test or gate that exists and passes today. A bullet with no
corresponding assertion is not a closeout — it is scope still open. Do not flip
`status` to close a documentation gap.

### 2. Flip the catalog entry

In `features.yaml`, set `status: "done"` and make sure `tier:` matches where the
feature's `validation_command` can actually run. A command that needs the rover
belongs in `hardware`, never `fast` — the fast tier runs on every push and a
hardware-only command turns the whole harness red on CI runners.

### 3. Pin `implemented_in` to a resolvable SHA

`implemented_in` must be a **hex commit SHA**, not a branch name. Audit the
whole catalog, not just the feature you touched:

```bash
python3 - <<'PY'
import re, yaml
doc = yaml.safe_load(open("features.yaml"))
feats = doc["features"] if isinstance(doc, dict) else doc
bad = [
    (f["id"], f.get("implemented_in"))
    for f in feats
    if f.get("status") == "done"
    and not re.fullmatch(r"[0-9a-f]{7,40}", str(f.get("implemented_in") or ""))
]
print("non-SHA implemented_in on done features:", bad or "none")
PY
```

A `null` on a `done` feature is the same debt wearing a different hat: the
harness cannot prove where the feature landed.

For a feature closing out on the current branch, the SHA is not known until the
commit exists. Land the code first, then amend `features.yaml` in a follow-up
commit with the merge SHA — do **not** leave the branch name as a placeholder
and plan to come back.

### 4. Run the harness gates

```bash
python scripts/validate.py --check F-025      # the single feature, any tier
python scripts/validate.py --tier fast        # the whole fast tier
python scripts/validate.py --tier fast,slow --strict-git   # what nightly runs
```

`--check` bypasses the tier filter, so it is the fastest way to prove one
feature's `validation_command` works. `--strict-git` is the one that catches
step 3 — run it locally before pushing rather than discovering it in the
nightly job.

### 5. Update the narrative docs

The catalog is machine state; these are the human record and drift
independently:

- `CHANGELOG.md` — an entry under the unreleased heading describing the
  behaviour change, not the file list.
- `progress.md` — append-only session record.
- `NEXT_STEPS.md` — remove the item the feature closes; a stale "next step"
  that already shipped is worse than no list. Note the doc-hygiene stage in
  `scripts/ci.sh` warns on drift here.
- `CLAUDE.md` — only when the feature adds a *contract* a future agent must not
  break (a config invariant, a wire format, an ordering requirement). Not every
  feature earns a block.

### 6. Verify nothing else moved

```bash
git diff --stat -- config/
```

Must be empty unless the change genuinely edits an overlay. The `config-compat`
workflow validates every changed `config/*.yaml` against the schema of the
commit pinned in `deployments/jetson-image.json` — i.e. the schema the deployed
rover image actually has — so an overlay edit that uses a brand-new field fails
that gate even though it is correct on trunk.

## Definition of done

- `features.yaml` entry is `done` with a hex-SHA `implemented_in`
- `python scripts/validate.py --tier fast` passes
- `python scripts/validate.py --tier fast,slow --strict-git` passes
- `CHANGELOG.md`, `progress.md`, `NEXT_STEPS.md` reflect the change
- `git diff --stat -- config/` is empty, or the overlay edit is deliberate and
  schema-compatible with the pinned image
