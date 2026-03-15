# CI/CD Pipeline: MCTS Latency & BDI Accuracy Gates

> Part of ADR-007 — append these stages to `.github/workflows/ci.yml`

## Stage Additions

### 1. `pytest-benchmark` Performance Gate

```yaml
# .github/workflows/ci.yml — add after 'unit-tests' job
benchmark:
  name: MCTS Benchmark Gate
  runs-on: ubuntu-latest
  needs: [unit-tests]
  env:
    MOUSEDROID_MCTS__N_SIMULATIONS_BASE: "50"
    MOUSEDROID_MCTS__ROLLOUT_DEPTH: "5"
    MOUSEDROID_MCTS__N_ACTION_CANDIDATES: "9"
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.10" }
    - run: pip install -e ".[dev]"
    - name: Run benchmark
      run: |
        pytest tests/benchmarks/test_mcts_benchmark.py \
          --benchmark-json=benchmark-results.json \
          --benchmark-max-time=60
    - name: Enforce p50 <= 50ms
      run: |
        python scripts/check_benchmark.py \
          --input benchmark-results.json \
          --metric p50_ms \
          --threshold 50.0
    - uses: actions/upload-artifact@v4
      with:
        name: benchmark-results
        path: benchmark-results.json
```

### 2. BDI Accuracy Post-Train Gate

```yaml
validate-training:
  name: BDI Accuracy Gate
  runs-on: ubuntu-latest
  needs: [unit-tests]
  env:
    MOUSEDROID_BDI_TRAINING__ACCURACY_THRESHOLD: "0.60"
    MOUSEDROID_BDI_TRAINING__EPOCHS: "300"
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.10" }
    - run: pip install -e ".[dev]"
    - name: Validate weights
      run: |
        python -m training.validate_weights \
          --weights-dir weights/ \
          --annotations training/data/bdi_annotations.npz \
          --output training/results/training_report.json
    - name: Check BDI accuracy
      run: |
        python scripts/check_report.py \
          --report training/results/training_report.json \
          --phase bdi_accuracy \
          --must-pass
    - uses: actions/upload-artifact@v4
      with:
        name: training-report
        path: training/results/training_report.json
```

## How to Debug Failures Locally

```bash
# Run benchmark locally
MOUSEDROID_MCTS__N_SIMULATIONS_BASE=50 \
  pytest tests/benchmarks/test_mcts_benchmark.py -v --benchmark-only

# Run BDI accuracy gate locally
PYTHONPATH=src python -m training.validate_weights \
  --weights-dir weights/ \
  --annotations training/data/bdi_annotations.npz

# Check report output
cat training/results/training_report.json | python -m json.tool
```

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `MOUSEDROID_MCTS__N_SIMULATIONS_BASE` | `50` | Simulation count per `plan()` |
| `MOUSEDROID_MCTS__ROLLOUT_DEPTH` | `5` | Rollout depth per simulation |
| `MOUSEDROID_MCTS__SIMULATION_BUDGET_MS` | `0.0` | Time budget (0 = disabled) |
| `MOUSEDROID_BDI_TRAINING__ACCURACY_THRESHOLD` | `0.60` | Min BDI accuracy to pass |
| `MOUSEDROID_BDI_TRAINING__EPOCHS` | `300` | Training epochs |
| `MOUSEDROID_BDI_TRAINING__BALANCE_CLASSES` | `false` | Enable class balancing |

All variables use the `MOUSEDROID_` prefix and `__` nested delimiter (configured in `Settings.model_config`).
