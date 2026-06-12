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

    Component(factory, "build_experiment_logger", "factory.py", "Resolves NoOp vs MLflow from cfg.observability; NEVER-None; degrades to NoOp on ImportError OR construction failure")

    Component_Boundary(obs, "training/observability/") {
        Component(proto, "ExperimentLoggerProtocol", "@runtime_checkable", "start_run/log_params/log_metric/log_artifact/end_run + start_phase/log_phase_metric/log_phase_artifact/end_phase; all total (never raise on backend failure)")
        Component(noop, "NoOpExperimentLogger", "default", "Byte-identical no-op; phase run_id = noop-phase-<phase>")
        Component(mlflow, "MlflowExperimentLogger", "MlflowClient", "Parent + nested child runs (mlflow.parentRunId tag); finite-float coercion; status normalisation")
        Component(ctx, "PhaseContext", "frozen dataclass", "Opaque (run_id, phase) handle routed back to per-phase calls")
    }

    Component(orch, "PipelineOrchestrator", "asyncio", "start_run -> per-phase start_phase/end_phase -> end_run; gates artifacts on cfg.log_artifacts")
    Component(trainer, "OfflineRLTrainer (CQL/IQL)", "torch", "Per-step loss metrics; throttles on step % log_step_every_n == 0")
}

ComponentDb(store, "MLflow tracking store", "file:./mlruns (or remote)", "Runs, metrics, params, artifacts")
Component(cfg, "ObservabilityConfig.experiment_logger", "Pydantic", "backend/tracking_uri/experiment_name/run_name/log_step_every_n/log_artifacts")

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

- **Backwards-compat (invariant #9).** `Settings.observability` defaults `None`;
  `ExperimentLoggerConfig.backend` defaults `"none"`. Pre-feature YAML loads
  unchanged and resolves to `NoOpExperimentLogger` — pinned by
  `tests/regression/test_observability_backwards_compat.py`.
- **Protocol-DI (invariant #1/#2).** The orchestrator + trainers import only
  `ExperimentLoggerProtocol`; concrete loggers are imported solely inside
  `build_experiment_logger`. The factory returns a NEVER-None protocol type, so
  callers drop the `logger is not None` guard.
- **No hardcoded values (invariant #3).** `tracking_uri`, `experiment_name`,
  `run_name`, `log_step_every_n`, and `log_artifacts` all come from
  `ObservabilityConfig.experiment_logger`. Every config field is actually
  consumed (no inert knobs).
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
    backend: mlflow            # "none" (default) | "mlflow"
    tracking_uri: file:./mlruns
    experiment_name: mousedroid
    run_name: my-pipeline      # optional; falls back to "pipeline"
    log_step_every_n: 10       # throttle per-step writes on long runs
    log_artifacts: true        # settings snapshot + per-phase checkpoints
```

Operator runbook (local UI, pitfalls, SSH-tunnel guidance):
`docs/runbooks/mlflow-local-ui.md`.
