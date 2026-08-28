# Spec delta — MLflow sqlite tracking default

## MODIFIED Requirements

### Requirement: The experiment logger's default tracking store SHALL be a local SQLite database, not the legacy file store

`ExperimentLoggerConfig.tracking_uri` SHALL default to
`sqlite:///mlflow.db`. The `[mlflow]` optional-dependency extra SHALL
include `sqlalchemy` and `alembic` in addition to `mlflow-skinny`, since
mlflow's sqlite tracking store requires both even with the skinny client.

#### Scenario: default config resolves to the sqlite backend

- **GIVEN** an `ExperimentLoggerConfig` constructed with no explicit
  `tracking_uri`
- **WHEN** its `tracking_uri` field is read
- **THEN** it equals `sqlite:///mlflow.db`

#### Scenario: the mlflow extra carries its sqlite-store dependencies

- **GIVEN** `pyproject.toml`'s `[project.optional-dependencies]` table
- **WHEN** the `mlflow` extra's requirement list is parsed
- **THEN** it names `mlflow-skinny`, `sqlalchemy`, and `alembic`

#### Scenario: an operator's explicit file: override still works unchanged

- **GIVEN** an `ExperimentLoggerConfig` with `tracking_uri` explicitly set
  to `file:./mlruns`
- **WHEN** `build_experiment_logger` resolves it via `_resolve_tracking_uri`
- **THEN** the relative path is pinned to an absolute `file:` path exactly
  as it was before this default changed — the legacy backend remains a
  fully-supported, unbroken opt-out

### Requirement: No shipped configuration overlay's effective experiment-logger state SHALL change

Every `config/*.yaml` overlay in the repository SHALL either declare no
`observability:` block at all, or (should a future overlay add one) set
`tracking_uri` explicitly rather than relying on the schema default.

#### Scenario: every shipped overlay is unaffected today

- **GIVEN** every file under `config/*.yaml`
- **WHEN** each is parsed and checked for an `observability:` key
- **THEN** none is present in any shipped overlay — every shipped
  deployment resolves `Settings.observability` to `None` and gets
  `NoOpExperimentLogger`, independent of what `tracking_uri` defaults to
