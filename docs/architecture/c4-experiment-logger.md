# C4 Component — Training Experiment Logger (MLflow / NoOp)

> Opt-in experiment tracking for the GPU pre-training pipeline. Training metrics,
> params, and artifacts flow to an MLflow backend when enabled; otherwise a
> byte-identical `NoOpExperimentLogger` is wired in so the orchestrator/trainer
> code path is unconditional. Protocol-DI: the orchestrator and trainers depend
> only on `ExperimentLoggerProtocol`, never on a concrete logger.

## Component Diagram

```mermaid
C4Component
title Training Experiment Logger — Component Diagram

Container_Boundary(train, "Training pipeline (offline, GPU host)") {

    Component(factory, "build_experiment_logger", "factory.py", "Resolves NoOp vs MLflow from cfg.observability; NEVER-None; degrades to NoOp on ImportError OR construction failure; pins relative file:/sqlite:/// URIs absolute via _resolve_tracking_uri and logs the redacted result")

    Component_Boundary(obs, "training/observability/") {
        Component(proto, "ExperimentLoggerProtocol", "@runtime_checkable", "start_run/log_params/log_metric/log_artifact/end_run + start_phase/log_phase_metric/log_phase_artifact/end_phase; all total (never raise on backend failure)")
        Component(noop, "NoOpExperimentLogger", "default", "Byte-identical no-op; phase run_id = noop-phase-<phase>")
        Component(mlflow, "MlflowExperimentLogger", "MlflowClient", "Parent + nested child runs (mlflow.parentRunId tag); finite-float coercion; status normalisation")
        Component(ctx, "PhaseContext", "frozen dataclass", "Opaque (run_id, phase) handle routed back to per-phase calls")
    }

    Component(orch, "PipelineOrchestrator", "asyncio", "start_run -> per-phase start_phase/end_phase -> end_run; gates artifacts on cfg.log_artifacts")
    Component(trainer, "OfflineRLTrainer (CQL/IQL)", "torch", "Per-step loss metrics; throttles on step % log_step_every_n == 0")
}

ComponentDb(store, "MLflow tracking store", "sqlite:///mlflow.db (or remote)", "Runs, metrics, params, artifacts")
Component(cfg, "ObservabilityConfig.experiment_logger", "Pydantic", "backend/tracking_uri/experiment_name/run_name/log_step_every_n/log_artifacts; tracking_uri validator rejects blank and sqlite:// (silent-discard values)")

Rel(cfg, factory, "resolves")
Rel(factory, noop, "default / backend=none / mlflow extra missing / init failed")
Rel(factory, mlflow, "backend=mlflow + extras present + constructs OK")
Rel(noop, proto, "implements")
Rel(mlflow, proto, "implements")
Rel(orch, proto, "depends on (DI)")
Rel(trainer, proto, "depends on (DI)")
Rel(orch, ctx, "start_phase -> ctx -> end_phase")
Rel(mlflow, store, "writes runs/metrics/artifacts")
```

## Key contracts

- **Backwards-compat (invariant #6).** `Settings.observability` defaults `None`;
  `ExperimentLoggerConfig.backend` defaults `"none"`. Pre-feature YAML loads
  unchanged and resolves to `NoOpExperimentLogger` — pinned by
  `tests/regression/test_observability_backwards_compat.py` and, for the
  F-034 `tracking_uri` default flip specifically (including the env-var
  opt-in that materialises it),
  `tests/regression/test_f034_mlflow_sqlite_backwards_compat.py`.
- **Credentials never reach a log event.** `tracking_uri` is a plain `str`,
  not a `SecretStr`, and a remote store may legitimately be spelled
  `http://user:password@host:5000`. There is no redaction processor in the
  structlog chain, so every site logging a tracking URI masks the userinfo
  component first via `mousedroid.logging.redaction.redact_uri_credentials`.
  Scheme, host, port and path survive; the secret does not. Pinned by
  `tests/integration/test_experiment_logger_redaction.py` (which asserts over
  the whole event stream, so a new log site cannot silently reintroduce the
  leak) and `tests/unit/logging/test_redaction.py`.
- **Protocol-DI (invariant #1).** The orchestrator + trainers import only
  `ExperimentLoggerProtocol`; concrete loggers are imported solely inside
  `build_experiment_logger`. The factory returns a NEVER-None protocol type, so
  callers drop the `logger is not None` guard.
- **No hardcoded values (invariant #2).** `tracking_uri`, `experiment_name`,
  `run_name`, `log_step_every_n`, and `log_artifacts` all come from
  `ObservabilityConfig.experiment_logger`. Each is consumed by its owning
  component: the orchestrator wires `run_name` + `log_artifacts`;
  `log_step_every_n` is the per-step throttle of `OfflineRLTrainer` (CQL/IQL),
  honoured wherever a trainer is constructed with it. (Threading
  `log_step_every_n` from config into the orchestrator's offline-RL phases is
  follow-up — those phases are stubs today.)
- **Total methods.** Every protocol method MUST NOT raise on backend failure
  (network drop, malformed input, NaN) — it emits a structured warning and
  returns, mirroring the LLM gateways' "never raises on backend failure" contract.
- **Degrade, never crash.** `build_experiment_logger` falls back to NoOp on a
  missing `[mlflow]` extra (`ImportError`) AND on construction failure (bad
  `tracking_uri` / store / permissions) — distinct warnings
  (`experiment_logger_mlflow_extras_missing` / `experiment_logger_mlflow_init_failed`).
- **Throttle safety.** `log_step_every_n` is `gt=0` in schema; the trainer also
  raises on `< 1` so the `step % n` throttle can never `ZeroDivisionError`. The
  step counter always increments so step indices stay monotonic.
- **Outside the hot loop.** This is offline training instrumentation; nothing here
  touches the 30 Hz reactive control loop.

## Operator usage

```yaml
# Opt in (YAML overlay) — defaults are OFF
observability:
  experiment_logger:
    backend: mlflow                  # "none" (default) | "mlflow"
    tracking_uri: sqlite:///mlflow.db  # default; mlflow's own recommended local backend
    experiment_name: mousedroid
    run_name: my-pipeline      # optional; falls back to "pipeline"
    log_step_every_n: 10       # throttle per-step writes on long runs
    log_artifacts: true        # settings snapshot + per-phase checkpoints
```

The CLI entry point resolves the logger from this config, so a YAML/env opt-in
takes effect with no code change:

```bash
python -m mousedroid.training.pipeline_orchestrator --config <training.yaml>
# async_main -> build_experiment_logger(settings) -> PipelineOrchestrator(experiment_logger=...)
```

**Operator triage** — structlog events to grep when runs do not appear, in
the order worth checking:
`experiment_logger_tracking_uri_resolved` **first** — the only event naming
the effective store path (`configured_uri` + `resolved_uri`), and emitted
*before* construction so it survives an init failure; then
`mlflow_logger_initialised` (backend up),
`experiment_logger_mlflow_extras_missing` /
`experiment_logger_mlflow_init_failed` (degraded to NoOp — check the
`[mlflow]` extra and `tracking_uri`; both now carry `configured_uri` too),
and `mlflow_logger_*_failed` (per-call backend warnings; the run is never
crashed). Every URI in these events has its credentials redacted
(`mousedroid.logging.redaction`), so a `user:password@host` remote store
shows as `***@host` — the host and path stay readable, the secret does not
reach the log. Operator runbook (local UI, pitfalls, SSH-tunnel guidance):
`docs/runbooks/mlflow-local-ui.md`.
