# MLflow Local UI — Operator Runbook

The MLflow experiment logger writes to a local file backend at
`<repo>/mlruns/` (or whatever `cfg.observability.experiment_logger.tracking_uri`
points to). This runbook covers viewing the data.

## Prerequisites

Install the local-viewer extras (different from the rover-side `[mlflow]`
extras — the viewer needs the full `mlflow` package for the UI server):

```bash
pip install "mlflow>=2.22,<3"
```

## Viewing runs

From the repo root:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

This binds `127.0.0.1:5000` by default. Override:

```bash
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5050
```

Open the URL it prints. The default experiment is `mousedroid`; runs are
named after the pipeline (parent) and each phase (child, nested via the
`mlflow.parentRunId` tag).

## Common pitfalls

* **No runs visible** — verify the working directory has an `mlruns/`
  subdirectory. The factory pins the path at startup; if the rover ran from
  a different CWD, the runs live elsewhere. Check
  `cfg.observability.experiment_logger.tracking_uri`.
* **"RUNNING" status stuck** — a training process crashed before `end_run`
  fired. Use the UI's "Delete run" or `mlflow.delete_run(run_id)`. The
  experiment logger calls `set_terminated(status="FAILED")` from the
  pipeline orchestrator's `finally` block, so this should be rare.
* **Concurrent writers** — the file backend is NOT thread/process safe for
  concurrent writers. Run one training process at a time per tracking URI.
* **Disk filling up** — old runs accumulate. Periodically clean with
  `mlflow gc --backend-store-uri file:./mlruns` or rotate the directory.

## Enabling on the rover

In `config/<overlay>.yaml`:

```yaml
observability:
  experiment_logger:
    backend: mlflow
    tracking_uri: file:/opt/mousedroid/mlruns
    experiment_name: mousedroid-jetson
    log_step_every_n: 10  # every 10th update_step (long runs)
```

Or via env:

```bash
MOUSEDROID_OBSERVABILITY__EXPERIMENT_LOGGER__BACKEND=mlflow
```

Required extras on the rover: `pip install "mousedroid[mlflow]"` —
installs only `mlflow-skinny` (write-only client, no Flask/SQLAlchemy).
