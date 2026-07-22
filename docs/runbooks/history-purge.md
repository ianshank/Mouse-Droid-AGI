# Runbook — Git history purge (large-blob removal)

Permanently remove `training/data/bdi_annotations.npz` (~25.8 MB) and
`docs/3D_printing_files/` (~6.5 MB STL/FCStd) from **all** git history, shrinking a full clone
from **~28 MB** (of which ~25.8 MB is the single `.npz` blob) to **~2 MB**.

Automated by [`scripts/purge_history.sh`](../../scripts/purge_history.sh).

> ## ⚠️ This is destructive and irreversible
> Rewriting history changes **every commit SHA**. Consequences:
> - All existing clones, open PRs, and forks become stale and must re-clone / be recreated.
> - The default branch must be **force-pushed** — branch protection has to be lifted for the push.
> - There is no undo once collaborators fetch the rewritten history.
>
> Do this deliberately, ideally right after the content-reframe PR merges, and tell collaborators
> to re-clone.

## Why this can't run from the Claude Code container

The remote is a policy proxy scoped to the feature branch (no `main`; the default branch is
`claude/markdown-implementation-plan-aVJ2l`), and the container has no `HF_TOKEN` / `gh` / real
GitHub token to preserve the artifacts. So the purge is an operator step you run on a machine with
your credentials and push rights. The reframe PR only *stops tracking* the files (they remain
recoverable in history until this purge).

## Preconditions — preserve the blobs first (mandatory)

The purge deletes the only copies left in history. Before running it:

1. **`bdi_annotations.npz` → Hugging Face dataset.** It's a *generated* artifact, so it can also be
   regenerated (`bash scripts/fetch_data.sh`), but mirror it for a fast path:
   ```bash
   # extract the blob from pre-purge history (works from any full clone)
   git show 032942b50ab71abf285282a4de7333193f208c38:training/data/bdi_annotations.npz > bdi_annotations.npz
   huggingface-cli upload ianshank/mouse-droid-bdi-annotations bdi_annotations.npz \
     bdi_annotations.npz --repo-type dataset
   ```
2. **STL/FCStd → GitHub Release `hardware-v6`.** Extract each from history (or a pre-purge
   checkout) and attach them to the release:
   ```bash
   gh release create hardware-v6 --title "MSE-6 CAD v6" --notes "Chassis STL + FreeCAD sources"
   gh release upload hardware-v6 docs/3D_printing_files/*.stl docs/3D_printing_files/*.FCStd
   ```

The in-repo pointers (`training/data/README.md`, `docs/3D_printing_files/README.md`) already
reference these destinations.

## Run it

```bash
pip install git-filter-repo          # not bundled with git

# DRY RUN (default): fresh clone -> purge -> re-pin -> verify config-compat; NO push
bash scripts/purge_history.sh

# For real (force-pushes the default branch + tags):
bash scripts/purge_history.sh --push
```

The script operates on a throwaway fresh clone under a temp dir — never your working repo.

## What the script does, and the one subtlety that matters

1. **Fresh clone** of `origin` (all refs).
2. **`git filter-repo --path … --invert-paths`** drops the two paths from every commit. (filter-repo
   refuses to run on a non-fresh repo and **removes the `origin` remote** by design — the script
   re-adds it before pushing.)
3. **Re-pin `deployments/jetson-image.json`** — the load-bearing step. The `config-compat` CI gate
   (`scripts/check_config_compat.py:worktree_at_sha`) does `git worktree add --detach <sha>` on the
   pinned SHA and loads `config/schema.py` **at that SHA** to validate the config YAML. The rewrite
   makes the old SHA unreachable, so the gate would die on every config-touching PR. The script
   maps the deployed SHA through filter-repo's `.git/filter-repo/commit-map` to its **rewritten
   image** and pins that — **not `HEAD`.** Pinning HEAD would silently change *which* schema the
   gate enforces (`schema.py` churns heavily across history; the deployed SHA was chosen
   schema-equivalent to the running image). The purge doesn't touch `schema.py`, so the commit-map
   image preserves the intended schema. The script aborts if the SHA mapped to a pruned/empty
   commit.
4. **Verify** by actually running `check_config_compat.py` against the re-pinned SHA (proves the
   schema *loads*, not merely that the SHA is reachable).
5. **Force-push** the default branch + tags (only with `--push`).

## Verify

```bash
git count-objects -vH        # size-pack should be ~2 MB, not ~28 MB
python scripts/check_config_compat.py --platform jetson --changed-files config/jetson_production.yaml
```

## Recovery

Until collaborators fetch the rewritten history, the pre-purge history still exists in their
clones and in GitHub's reflog for a while. If you must undo before propagation, force-push the old
default-branch SHA back. After collaborators re-clone, recovery is no longer possible — which is
the point.

## Optional: also purge stale large text blobs

History also carries many ~210 KB revisions of `config/schema.py` and `CHANGELOG.md`. They are not
targeted here (the task scoped the purge to the CAD/data blobs). If you later want them gone too,
add `--path config/schema.py --blob-callback …` filtering — but that rewrites far more and is a
separate decision.
