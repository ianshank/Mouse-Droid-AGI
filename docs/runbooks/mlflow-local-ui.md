# MLflow Local UI — Operator Runbook

The MLflow experiment logger writes to a local SQLite database at
`mlflow.db`, relative to the process's working directory, by default (or
whatever `cfg.observability.experiment_logger.tracking_uri` points to —
see "Enabling on the rover" below for the legacy file-backend alternative
and "Common pitfalls" for the CWD caveat). This runbook covers viewing the
data.

## Prerequisites

Install the local-viewer extras (different from the rover-side `[mlflow]`
extras — the viewer needs the full `mlflow` package for the UI server).
Match the rover's own `mlflow-skinny` upper bound (`pyproject.toml`'s
`[mlflow]` extra) rather than pinning the viewer independently: the sqlite
backend is schema-versioned via Alembic migrations, so a viewer *older*
than the client that wrote the database can fail to open it.

```bash
pip install "mlflow>=2.22,<4"
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
  pins **both** local schemes — `file:` and relative `sqlite:///` — to an
  absolute path at startup, and logs the outcome as
  `experiment_logger_tracking_uri_resolved`. Grep that event first: it
  carries the effective `resolved_uri` and is emitted *before* the store is
  opened, so it survives an init failure (the store's own errors — "unable
  to open database file", "file is not a database" — name no path). Note
  pinning happens against the *launching* process's CWD, so a rover started
  from a different directory still gets a different database; the cure is an
  absolute `tracking_uri`, not the pin. Check
  `cfg.observability.experiment_logger.tracking_uri`.
* **Config refuses to load with a `tracking_uri` error** — two values are
  rejected at schema validation because they silently discard every run: an
  empty or whitespace-only URI (mlflow would fall back to its own ambient
  default, or to `MLFLOW_TRACKING_URI`), and `sqlite://` with **two**
  slashes — SQLAlchemy's in-memory database, where every run is dropped at
  process exit with no error anywhere. Use `sqlite:///mlflow.db` (three
  slashes = a real file); the explicit in-memory spelling
  `sqlite:///:memory:` is still allowed.
* **"RUNNING" status stuck** — a training process crashed before `end_run`
  fired. Use the UI's "Delete run" or `mlflow.delete_run(run_id)`. The
  experiment logger calls `set_terminated(status="FAILED")` from the
  pipeline orchestrator's `finally` block, so this should be rare.
* **Concurrent writers** — SQLite handles concurrent readers/writers far
  better than the legacy file backend (which is NOT thread/process safe at
  all), but still serializes writes at the database-file level. Run one
  training process at a time per tracking URI for best results; a second
  concurrent writer will retry-and-block rather than corrupt data.
* **Disk filling up** — under the sqlite default, runs accumulate in **two**
  places, and this catches people out: run *metadata* goes into `mlflow.db`,
  but *artifacts* (the resolved-settings snapshot, per-phase checkpoints)
  still land under `mlruns/`. mlflow's SQLAlchemy store defaults its artifact
  root to `./mlruns` regardless of where the database lives, and the logger
  does not override it — so archiving or rotating `mlflow.db` alone orphans
  the artifact tree rather than reclaiming it. `mlflow gc
  --backend-store-uri sqlite:///mlflow.db` prunes both, for runs already
  marked deleted. Both paths are covered by `.gitignore` and `.dockerignore`,
  so this is a disk concern, not an accidental-commit one.

## Enabling on the rover

In `config/<overlay>.yaml`. Setting `tracking_uri` to an **absolute** path is
recommended even though the relative default works: the default is resolved
against whatever directory the training process was launched from, so
launching from two different directories quietly produces two separate
databases. An absolute URI makes the **database** location independent of how
the process was started.

It does not do the same for artifacts. mlflow's SQLAlchemy store derives its
artifact root separately, defaulting to `./mlruns` relative to the process
CWD, and an absolute `tracking_uri` does not move it — so a rover launched
from two directories gets one shared database and two artifact trees. Until
the artifact root is configurable here, pin the working directory (the
container already does: `WORKDIR /opt/mousedroid`) rather than relying on the
URI alone.

```yaml
observability:
  experiment_logger:
    backend: mlflow
    # POSIX absolute path: 4 slashes total (scheme's 3 + the path's leading /).
    # On Windows use 3 and let the drive letter follow: sqlite:///C:/mousedroid/mlflow.db
    tracking_uri: sqlite:////opt/mousedroid/mlflow.db
    experiment_name: mousedroid-jetson
    log_step_every_n: 10  # every 10th update_step (long runs)
```

To confirm which database a run actually went to, grep the startup logs for
`experiment_logger_tracking_uri_resolved` — it reports both the configured
URI and the absolute one it resolved to.

### Upgrading from the file backend

The default changed from `file:./mlruns` to `sqlite:///mlflow.db`. If you
enable mlflow via config file you are unaffected (no shipped overlay sets
`backend: mlflow`), but if you opt in via the environment variable below,
the new default takes effect on upgrade and **runs recorded before the
upgrade stay in `mlruns/` and stop appearing in the UI**. They are not lost.
Either keep using them by pinning the old backend explicitly:

```yaml
observability:
  experiment_logger:
    backend: mlflow
    tracking_uri: file:./mlruns   # legacy directory store, still fully supported
```

...or view them side by side by pointing a second UI at the old directory:
`mlflow ui --backend-store-uri file:./mlruns --port 5001`. MLflow provides no
in-place file-store→SQLite migration, so there is no one-shot conversion step.

Or via env:

```bash
MOUSEDROID_OBSERVABILITY__EXPERIMENT_LOGGER__BACKEND=mlflow
```

Required extras on the rover: `pip install "mousedroid[mlflow]"` — installs
`mlflow-skinny` (write-only client, no Flask/UI server) plus `sqlalchemy`
and `alembic`, which mlflow's sqlite tracking store needs even with the
skinny client.
