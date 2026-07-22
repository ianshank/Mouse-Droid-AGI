# Training data

Large training artefacts are **not committed** to this repository — they bloat clone size and are
reproducible from source. They are gitignored (see `.gitignore`: `training/data/*.npz`, `*.pt`).

## `bdi_annotations.npz`

A **generated** BDI intention-annotation array (~26 MB) consumed by the GPU pretraining pipeline
(`training/run_pipeline.py` → `run_phase_0b_annotations` → `training/collect_annotations.py`).

Get it with the helper — regeneration is the authoritative path; the Hugging Face mirror is an
optional fast path:

```bash
# Regenerate from source (default)
bash scripts/fetch_data.sh

# Or pull the published mirror (dataset: ianshank/mouse-droid-bdi-annotations)
bash scripts/fetch_data.sh --from-hf
```

See `bash scripts/fetch_data.sh --help` for the environment overrides (`CONFIG`, `HF_DATASET`,
`DATA_DIR`).

> This file previously lived in git history and was purged to shrink the clone — see
> [`docs/runbooks/history-purge.md`](../../docs/runbooks/history-purge.md).
