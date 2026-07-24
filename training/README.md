# Offline Training

GPU-host training pipelines that run **outside** the 30 Hz runtime loop. Entry point: `run_pipeline.py`
(phases 0 → 4). See [`../docs/training.md`](../docs/training.md) and
[`../docs/architecture/c4-rssm-sim-pretraining.md`](../docs/architecture/c4-rssm-sim-pretraining.md).

`data/` holds generated artifacts (gitignored — regenerate via `scripts/fetch_data.sh`).
