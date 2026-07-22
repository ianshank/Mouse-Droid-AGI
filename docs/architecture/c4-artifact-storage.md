# C4 Component — Large-Artifact Storage (pointer READMEs · regeneration-first fetch · history purge)

> How MouseDroid keeps large binaries **out of git** without losing them. The
> repo previously tracked a 25.8 MB generated `bdi_annotations.npz` and ~6.5 MB
> of CAD (STL/FreeCAD) blobs — they bloated every clone and lived in history
> forever. This layer replaces them with in-tree **pointer READMEs**, a
> **regeneration-first** fetch helper, and an operator-run **history purge** that
> rewrites all refs safely. Additive and reversible up to the purge: the reframe
> PR only *stops tracking* the files; they stay recoverable in history until the
> (separate, opt-in) purge runs.
>
> Companion to `docs/runbooks/history-purge.md` (operator workflow),
> `docs/architecture/c4-overview.md` (Levels 1–2), and the reframe contracts in
> `CLAUDE.md` / `tests/regression/test_portfolio_reframe_aqa.py`.

## Level 3 — Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Ignore rules | `.gitignore`, `.dockerignore` | Keep `*.npz` / `*.stl` / `*.FCStd` and the CAD dir out of the index AND the Docker build context (pointer READMEs are negated back in). |
| Data pointer | `training/data/README.md` | Documents that `bdi_annotations.npz` is generated + how to fetch/regenerate it. |
| CAD pointer | `docs/3D_printing_files/README.md` | Points to the `hardware-v6` GitHub Release for the STL/FreeCAD files. |
| Fetch helper | `scripts/fetch_data.sh` | Regenerates the `.npz` via the training pipeline (default) or pulls the HF-dataset mirror (`--from-hf`). |
| History purge | `scripts/purge_history.sh` | Operator-run `git filter-repo` rewrite that permanently drops the blobs from all history. |
| Contract lock | `tests/regression/test_portfolio_reframe_aqa.py` | Pins "blobs untracked + pointers present", the fetch/purge script contracts, and the deploy-SHA format. |

## Data flow — get the artefact (no history rewrite)

```
operator ── bash scripts/fetch_data.sh ──▶ python -m training.run_pipeline --phases 0
                                              │  (writes cfg.training.data_dir/bdi_annotations.npz)
              bash scripts/fetch_data.sh --from-hf ──▶ HF dataset  ianshank/mouse-droid-bdi-annotations
```

- **Regeneration is authoritative**; the HF mirror is an opt-in fast path.
- `CONFIG` is passed to Python via `sys.argv` — never interpolated into source — and a
  load/import error propagates (no silent wrong-path fallback).
- `DATA_DIR` governs only the `--from-hf` download dir; the regenerate path resolves the real
  output dir from `cfg.training.data_dir`.

## Data flow — purge (destructive, operator-run, opt-in)

```
git clone --mirror ORIGIN  ──▶  git filter-repo --invert-paths       (drops the data blob +
    (ALL refs)                    --path bdi_annotations.npz            CAD *binaries* by glob,
                                  --path-glob '.../*.stl' '.../*.FCStd' keeping the pointer README)
                                        │
                                        ▼
   commit-map re-pin of deployments/jetson-image.json  ──▶  config-compat verify (worktree)
        (rewritten image of the deployed SHA, NOT HEAD)          │
                                                                 ▼
                                      git push --force --all  +  --tags   (every rewritten ref)
```

## Non-negotiable contracts

- **Mirror + all-refs push.** A blob left reachable from any un-rewritten branch defeats the
  purge, so the script uses `git clone --mirror` and `git push --force --all`/`--tags`.
- **Commit-map re-pin, not HEAD.** The `config-compat` CI gate worktrees the pinned SHA and loads
  `config/schema.py` *at that SHA*; the deployed SHA was chosen schema-equivalent to the running
  image, and `schema.py` churns across history, so the rewritten image of the deployed SHA is
  pinned — never HEAD. The purge aborts if the SHA maps to a pruned commit.
- **CAD binaries by glob.** `--path-glob` targets the STL/FreeCAD files, never the whole
  `docs/3D_printing_files/` dir, so the pointer `README.md` survives.
- **Dry-run by default.** `purge_history.sh` clones + purges + verifies but pushes nothing until
  `--push`; the default branch is resolved from the TARGET remote (`ORIGIN_URL`).
- **Preserve before purge.** Uploading the `.npz` to HF and the CAD to a Release is a mandatory
  operator precondition (the purge deletes the only in-history copies).

## Blast radius

The purge rewrites every commit SHA: all clones/PRs/forks become stale and must re-clone, and it
requires branch-protection lifted for the force-push. It is therefore a deliberate, operator-run
maintenance step — not part of any CI job.
