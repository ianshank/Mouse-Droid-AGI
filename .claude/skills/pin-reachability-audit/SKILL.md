---
name: pin-reachability-audit
description: Check that every done feature's implemented_in SHA is still reachable from a surviving remote branch or tag, not just resolvable in the local clone. Use before archiving/deleting branches, and after any squash-merge that lands features.yaml pins.
status: active
---

# Pin Reachability Audit

`implemented_in` resolving locally is not the same claim as `implemented_in`
surviving the next branch cleanup. This audits the difference.

## Why this exists

`scripts/validate.py --strict-git` calls `git_rev_ok`
(`src/mousedroid/harness/spec.py::git_rev_ok`), which is a bare
`git rev-parse --verify --quiet <ref>^{commit}` — an **existence** check
against the local object database. A SHA passes it as long as the commit
object is present in *this clone*, regardless of whether any remote ref still
points at it.

That gap is not hypothetical. In one session, PR #201 squash-merged as
`9bd3dc7`, discarding all 11 branch commits. Two `implemented_in` pins
(F-028's `df928d9`, F-029's `5c2f044`) survived only on the source branch —
and GitHub's delete-branch-on-merge then removed it. `git rev-parse` kept
resolving both SHAs in the working clone (the objects were still present
locally) right up until a fresh clone proved otherwise: `git branch -r
--contains <sha>` returned **zero** remote branches for either. The nightly
`--strict-git` run — which clones fresh — was already broken by the time
this was caught by hand, not by any gate.

`tests/regression/test_harness_spec_aqa.py`'s own docstring says this
plainly: `test_done_features_pin_a_hex_sha_not_a_branch_name` is "a FORMAT
check and nothing more. It does not verify that the SHA resolves, that it is
an ancestor of any branch... Resolvability is deliberately left to
`validate.py --strict-git`" — and `--strict-git` itself only checks local
existence. Nothing in the repo checks *remote reachability*. This skill is
that check, run deliberately rather than discovered by accident.

## Run it

For one feature:

```bash
sha=$(python -c "
import yaml
d = yaml.safe_load(open('features.yaml', encoding='utf-8'))
f = next(f for f in d['features'] if f['id'] == 'F-0NN')
print(f['implemented_in'])
")
git fetch origin --quiet
echo "local exists:   $(git cat-file -e "$sha" 2>/dev/null && echo yes || echo NO)"
echo "remote carriers: $(git branch -r --contains "$sha" | wc -l)"
echo "remote tags:     $(git tag --contains "$sha" | while read -r t; do
  git ls-remote --tags origin | grep -qF "refs/tags/$t" && echo "$t"; done | wc -l)"
```

Zero remote carriers **and** zero remote tags means the pin is currently
unresolvable from a fresh clone — fix it now, the same way as a broken pin
(re-pin to a commit that IS reachable, per the squash-SHA convention in
`docs/architecture/` and `HARNESS_SPEC.md`).

For every `done` feature at once, reuse the exact carrier-detection logic
`scripts/archive_stale_branches.sh` already runs (for the opposite purpose —
deciding what *not* to delete):

```bash
git fetch --prune --tags origin --quiet

python - <<'PY'
import subprocess, yaml

d = yaml.safe_load(open("features.yaml", encoding="utf-8"))
remote_tags = set(
    line.split("refs/tags/")[-1].rstrip("^{}")
    for line in subprocess.run(
        ["git", "ls-remote", "--tags", "origin"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
)
for f in d["features"]:
    sha = f.get("implemented_in")
    if f["status"] != "done" or not sha:
        continue
    carriers = subprocess.run(
        ["git", "branch", "-r", "--contains", sha], capture_output=True, text=True
    ).stdout.strip()
    tagged = bool(
        set(
            subprocess.run(
                ["git", "tag", "--contains", sha], capture_output=True, text=True
            ).stdout.split()
        )
        & remote_tags
    )
    if not carriers and not tagged:
        print(f"UNRESOLVABLE: {f['id']} pins {sha[:12]} with no surviving remote carrier")
PY
```

## When to run it

- **Before** running `scripts/archive_stale_branches.sh --push` — its own
  pin-carrier protection covers `features.yaml` + the deploy manifest, but
  only for branches it is *about to delete*; it does not retroactively audit
  branches already gone.
- **After** any squash-merge that closes out a feature, before the source
  branch is deleted (by you or by delete-branch-on-merge) — catch the gap
  before it opens, not after.
- Periodically against `origin` as a standing health check — a branch can be
  deleted by someone else, or by a repo setting, outside any workflow this
  repo's own tooling sees.

## Guardrails

- Read-only. `git fetch` and `git ls-remote` only; never deletes or rewrites
  anything.
- A SHA with zero remote carriers is not necessarily wrong *today* if the
  commit is still reachable via a remote tag — check tags before treating a
  bare carrier count of zero as broken.
- Re-pinning the fix is a separate, deliberate step (see `feature-closeout`
  for the mechanics of picking a squash-merge SHA over a branch-point SHA).
