# MLflow Local UI — Operator Runbook

The MLflow experiment logger writes to a local SQLite database at
`<repo>/mlflow.db` by default (or whatever
`cfg.observability.experiment_logger.tracking_uri` points to — see
"Enabling on the rover" below for the legacy file-backend alternative).
This runbook covers viewing the data.

## Prerequisites

Install the local-viewer extras (different from the rover-side `[mlflow]`
extras — the viewer needs the full `mlflow` package for the UI server):

```bash
pip install "mlflow>=2.22,<3"
```

## Viewing runs

From the repo root, against the default sqlite backend:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

This binds `127.0.0.1:5000` by default. Override:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5050
```

If `tracking_uri` was explicitly overridden to a `file:` URI (the legacy
backend — see "Enabling on the rover"), point the UI at that path instead:
`mlflow ui --backend-store-uri file:./mlruns`.

> **⚠️ Network exposure:** `--host 0.0.0.0` binds every interface, exposing
> experiment metadata (params, metrics, artifact paths) to anyone on the
> reachable network — the MLflow UI has no authentication. Prefer the default
> loopback (`127.0.0.1`); only use `0.0.0.0` on a trusted/private network, and
> tunnel over SSH (`ssh -L 5000:127.0.0.1:5000 <host>`) for remote access.

Open the URL it prints. The default experiment is `mousedroid`; runs are
named after the pipeline (parent) and each phase (child, nested via the
`mlflow.parentRunId` tag).

## Common pitfalls

* **No runs visible** — verify the working directory has an `mlflow.db`
  file (or, for a `file:` override, an `mlruns/` subdirectory). The factory
  pins `file:` URIs to an absolute path at startup (sqlite URIs pass
  through unresolved — a relative `sqlite:///mlflow.db` is relative to
  whatever CWD the process was launched from); if the rover ran from a
  different CWD, the database lives elsewhere. Check
  `cfg.observability.experiment_logger.tracking_uri`.
* **"RUNNING" status stuck** — a training process crashed before `end_run`
  fired. Use the UI's "Delete run" or `mlflow.delete_run(run_id)`. The
  experiment logger calls `set_terminated(status="FAILED")` from the
  pipeline orchestrator's `finally` block, so this should be rare.
* **Concurrent writers** — SQLite handles concurrent readers/writers far
  better than the legacy file backend (which is NOT thread/process safe at
  all), but still serializes writes at the database-file level. Run one
  training process at a time per tracking URI for best results; a second
  concurrent writer will retry-and-block rather than corrupt data.
* **Disk filling up** — old runs accumulate. Periodically clean with
  `mlflow gc --backend-store-uri sqlite:///mlflow.db` or archive/rotate the
  database file (a `file:` override has the same pitfall — rotate the
  directory, or run `mlflow gc --backend-store-uri file:./mlruns`).

## Enabling on the rover

In `config/<overlay>.yaml` (`tracking_uri` is optional — this is already
the schema default, shown explicitly here for clarity):

```yaml
observability:
  experiment_logger:
    backend: mlflow
    tracking_uri: sqlite:////opt/mousedroid/mlflow.db  # note: 4 slashes for an absolute path
    experiment_name: mousedroid-jetson
    log_step_every_n: 10  # every 10th update_step (long runs)
```

Or via env:

```bash
MOUSEDROID_OBSERVABILITY__EXPERIMENT_LOGGER__BACKEND=mlflow
```

Required extras on the rover: `pip install "mousedroid[mlflow]"` — installs
`mlflow-skinny` (write-only client, no Flask/UI server) plus `sqlalchemy`
and `alembic`, which mlflow's sqlite tracking store needs even with the
skinny client.
