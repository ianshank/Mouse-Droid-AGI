# Training

Offline, GPU-host training that runs **outside** the 30 Hz reactive loop. The pipeline phases and scripts are
summarised in the root README "Training" table.

```bash
python training/run_pipeline.py                          # full pipeline (phase 0 → 4)
python training/run_pipeline.py --phases 0,1             # data generation + RSSM baseline
python training/run_pipeline.py --resume training/results/rssm_epoch_10.pt
python training/upload_weights.py --repo ianshank/mousedroid-weights
```

## Dual-Stream CfC/GRU RSSM (experimental)

A liquid-network hybrid: a GRU slow-planning stream (256-dim) + a CfC fast-reflex stream (64-dim) fused into a
320-dim hidden state. **Disabled by default** (`cfc_hidden_dim=0`); activate via
`MOUSEDROID_MODEL__CFC_HIDDEN_DIM=64`.

```bash
pip install -e ".[cfc]"
MOUSEDROID_MODEL__CFC_HIDDEN_DIM=64 \
    python -m training.train_dual_stream_rssm --config config/local_training.yaml \
    --data training/data/sequences.pt --device cuda --validate-only
```

Weights are auto-pulled at startup when `cognitive.enabled=true`. Architecture:
[architecture/c4-rssm-sim-pretraining.md](architecture/c4-rssm-sim-pretraining.md).
