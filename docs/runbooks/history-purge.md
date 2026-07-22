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

The remote is a policy proxy scoped to the working feature branch (this fork has no `main`; the
default branch is discovered dynamically by the purge script), and the container has no
`HF_TOKEN` / `gh` / real GitHub token to preserve the artifacts. So the purge is an operator step
you run on a machine with your credentials and push rights. The reframe PR only *stops tracking*
the files (they remain recoverable in history until this purge).

## Preconditions — preserve the blobs first (mandatory)

The purge deletes the only copies left in history. Before running it:

1. **`bdi_annotations.npz` → Hugging Face dataset.** It's a *generated* artifact, so it can also be
   regenerated (`bash scripts/fetch_data.sh`), but mirror it for a fast path. Discover the newest
   commit that still carries the blob rather than pinning a SHA that history may rewrite:
   ```bash
   npz="training/data/bdi_annotations.npz"
   c="$(git rev-list -n1 --all -- "$npz")"   # newest commit still holding it
   git show "$c:$npz" > bdi_annotations.npz
   huggingface-cli upload ianshank/mouse-droid-bdi-annotations bdi_annotations.npz \
     bdi_annotations.npz --repo-type dataset
   ```
2. **STL/FCStd → GitHub Release `hardware-v6`.** Restore the CAD files from the last commit that
   still had them (they were `git rm`-ed at the tip in Phase A), then attach them:
   ```bash
   c="$(git rev-list -n1 --all -- docs/3D_printing_files/mse6_v6_complete.stl)"
   git checkout "$c" -- docs/3D_printing_files/   # restores STL + FCStd into the worktree
   gh release create hardware-v6 --title "MSE-6 CAD v6" --notes "Chassis STL + FreeCAD sources"
   gh release upload hardware-v6 docs/3D_printing_files/*.stl docs/3D_printing_files/*.FCStd
   ```

The in-repo pointers (`training/data/README.md`, `docs/3D_printing_files/README.md`) already
reference these destinations.

## Run it

```bash
pip install git-filter-repo          # not bundled with git

# DRY RUN (default): mirror clone -> purge -> re-pin -> verify config-compat; NO push
bash scripts/purge_history.sh

# For real (force-pushes EVERY rewritten branch + tags):
bash scripts/purge_history.sh --push
```

The script operates on a throwaway bare **mirror** clone under a temp dir — never your working repo.

## What the script does, and the subtleties that matter

1. **Bare mirror clone** of `origin` (`git clone --mirror` — ALL refs, so every branch is rewritten,
   not just the default). The default branch is resolved from the *target* remote (`ORIGIN_URL`), so
   an `ORIGIN_URL` override re-pins the right repo.
2. **`git filter-repo --path … --path-glob … --invert-paths`** drops the data blob + CAD binaries
   from every ref (globbing the CAD *binaries* keeps the pointer `README.md`). filter-repo refuses to
   run on a non-fresh repo and **removes the `origin` remote** by design — the script re-adds it
   before pushing.
3. **Re-pin `deployments/jetson-image.json`** — the load-bearing step. The `config-compat` CI gate
   (`scripts/check_config_compat.py:worktree_at_sha`) does `git worktree add --detach <sha>` on the
   pinned SHA and loads `config/schema.py` **at that SHA** to validate the config YAML. The rewrite
   makes the old SHA unreachable, so the gate would die on every config-touching PR. The script maps
   the deployed SHA through filter-repo's `commit-map` to its **rewritten image** and pins that in a
   worktree commit on the default branch — **not `HEAD`.** Pinning HEAD would silently change *which*
   schema the gate enforces (`schema.py` churns heavily across history; the deployed SHA was chosen
   schema-equivalent to the running image). The purge doesn't touch `schema.py`, so the commit-map
   image preserves the intended schema. The script aborts if the SHA mapped to a pruned/empty commit.
4. **Verify** by actually running `check_config_compat.py` against the re-pinned SHA (proves the
   schema *loads*, not merely that the SHA is reachable).
5. **Force-push every rewritten branch + tags** (`git push --force --all` + `--tags`, only with
   `--push`). All branches are published because a blob left reachable from any un-rewritten branch
   would defeat the purge; `--all` / `--tags` update refs without deleting any.

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
