# Proposal — MLflow sqlite tracking default

- change_id: mouse-droid-mlflow-sqlite-tracking
- project: mouse-droid
- status: in progress
- feature_id: F-034
- epic: Training Observability
- owner: ianshank
- created: 2026-08-27
- basis_commit: fd84395
- rev: A

## Why

`ExperimentLoggerConfig.tracking_uri` (`config/schema/telemetry.py`) defaults
to the legacy `file:./mlruns` URI. mlflow 3.x hard-rejects the plain
file-store backend without an explicit `MLFLOW_ALLOW_FILE_STORE=true`
opt-in — already set unconditionally in `mlflow_logger.py` since PR #198
(2026-08-20), three days after dependabot PR #145 widened the
`mlflow-skinny` version bound and a manual test run against mlflow 3.15.1
diagnosed 29 failed / 17 errored tests (of ~74) and posted the finding to
`NEXT_STEPS.md` item 0.

Verified empirically before writing this bundle, not trusted from that
diagnosis: installed mlflow 3.15.2, ran the full 88-test mlflow-touching
surface against the *old* `file:./mlruns` default with the current tree's
already-present `MLFLOW_ALLOW_FILE_STORE=true` workaround — all 88 passed.
The acute hard-rejection blocker `NEXT_STEPS.md` item 0 diagnosed was
already resolved by that env var; nobody updated the doc to say so. This
bundle is therefore reframed from "unblock a broken path" to "complete the
fix mlflow's own docs recommend" — sqlite is mlflow's documented production
backend, and moving the default there removes the codebase's dependency on
an env-var workaround for a rejection future mlflow versions could tighten
further (e.g. removing the escape hatch entirely).

## What Changes

- `config/schema/telemetry.py`: `ExperimentLoggerConfig.tracking_uri`'s
  default → `sqlite:///mlflow.db`. Both its own docstring and
  `backend`'s neighbouring docstring (which cited the old default) updated.
- `pyproject.toml`: the `[mlflow]` extra gains `sqlalchemy>=2.0,<3` and
  `alembic>=1.13,<2` — confirmed by direct test that mlflow-skinny alone
  raises `UnsupportedModelRegistryStoreURIException` for any `sqlite:///`
  URI without them, and that a real `MlflowClient` run (create_experiment /
  create_run / log_metric / set_terminated) succeeds once they're present.
- `.github/workflows/ci.yml`: new advisory `mlflow-extras` job, mirroring
  `onnx-world-model-extras`'s exact shape — the first job to ever install
  the `[mlflow]` extra in any CI path. Tracked in
  `.github/advisory_stages.yaml` (180-day green-run-count window, same
  convention as its sibling).
- `docs/architecture/c4-experiment-logger.md`: the tracking-store diagram
  label and the operator-usage YAML sample updated to the new default.
- `docs/runbooks/mlflow-local-ui.md`: viewing-runs commands, "Common
  pitfalls" (concurrent-writer/disk-cleanup guidance), and "Enabling on the
  rover" all updated for the sqlite default; the closing "Required extras"
  paragraph's now-incorrect "no ... SQLAlchemy" claim corrected.
- `CLAUDE.md` + `docs/claude/surfaces/ci-gates.md`: job-count bumped 16 → 17,
  advisory-job-count bumped 5 → 6, `mlflow-extras` added to both docs'
  stage lists (both were already-accurate counts before this change, so
  both needed updating in the same commit to stay accurate after it).
- `NEXT_STEPS.md`: item 0 removed (resolved) rather than reworded — its
  "Open dependabot PR #145" framing was independently stale too (#145
  merged 2026-08-17, three days before the diagnosis commit still called
  it "open").
- `_resolve_tracking_uri` (`factory.py`) is unchanged: it already passed
  non-`file:` URIs (including `sqlite:`) through unchanged, so the new
  default needed no code change there — only new test coverage proving
  the previously-untested-but-unaffected behaviour for the new default.

## Impact

No behaviour change for any shipped config: `grep`-confirmed zero
`config/*.yaml` overlays reference `observability:`/`experiment_logger:`
at all, so every shipped deployment resolves `Settings.observability` to
`None` and gets `NoOpExperimentLogger` regardless of what
`ExperimentLoggerConfig.tracking_uri` defaults to — a stronger safety
margin than F-029's twin-overlay case, which had one shipped overlay to
account for explicitly.

**Residual risk, named rather than hidden:** an operator who sets
`backend: mlflow` with no explicit `tracking_uri` override, on a host that
happens to have `mlflow-skinny` pinned in the 2.x range specifically (not
this repo's own bound, which already covers 2.x-4.x), would see a real
tracking-backend change. This exact combination was already broken under
mlflow 3.x before this change (per the diagnosed-but-then-independently-
resolved blocker), so the tradeoff is asymmetric in this fix's favour — but
it is a genuine, if narrow, behaviour change worth naming rather than
presenting this as purely inert.

No CHARTER.md §3 carve-out needed: this is training-side (offline,
GPU-host) instrumentation, entirely outside the 30 Hz control loop, gated
behind the pre-existing `backend` opt-in which stays `"none"` by default.

## Spec Deltas

`openspec/changes/mouse-droid-mlflow-sqlite-tracking/specs/mlflow-experiment-tracking/spec.md`

## Tasks

See `openspec/changes/mouse-droid-mlflow-sqlite-tracking/tasks.md`.

## Validation

`bash scripts/validations/F-034.sh`
