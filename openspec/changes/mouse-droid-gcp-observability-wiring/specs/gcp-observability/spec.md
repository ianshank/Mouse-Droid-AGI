# Spec delta — GCP observability wiring

## ADDED Requirements

### Requirement: Cloud Logging, Monitoring, and Firestore sinks SHALL be constructed only when explicitly enabled and their required collaborator is available

`factory.py` SHALL provide `build_cloud_logging_sink`, `build_cloud_metrics_exporter`,
and `build_cloud_firestore_sync`, each returning `None` rather than raising
when `cfg.gcp` is `None`, when the corresponding `gcp.<x>.enabled` flag is
`False`, when a required collaborator (`metrics_registry` /
`episodic`) is `None`, or when the underlying `google-cloud-*` package is not
installed.

#### Scenario: disabled by default

- **GIVEN** `Settings.gcp` is `None` (the default)
- **WHEN** `build_cloud_logging_sink`, `build_cloud_metrics_exporter`, and
  `build_cloud_firestore_sync` are each called
- **THEN** all three return `None`

#### Scenario: enabled GCP block, flag still off

- **GIVEN** a `GCPConfig` with `logging.enabled=False`,
  `monitoring.enabled=False`, `firestore.enabled=False` (the schema default
  for each)
- **WHEN** the three builders are called
- **THEN** all three return `None`

#### Scenario: required collaborator missing

- **GIVEN** `gcp.firestore.enabled=True` and `memory.enabled=False` (the
  schema default), a legal but unvalidated combination
- **WHEN** `build_cloud_firestore_sync(cfg, episodic=None)` is called
- **THEN** it returns `None` rather than constructing `CloudFirestoreSync`
  with a `None` `episodic` argument
- **AND** the same holds for `build_cloud_metrics_exporter(cfg,
  metrics_registry=None)` when `gcp.monitoring.enabled=True` and
  `metrics.enabled=False`

#### Scenario: SDK not installed

- **GIVEN** `gcp.logging.enabled=True` and the `google-cloud-logging`
  package is not importable
- **WHEN** `build_cloud_logging_sink` is called
- **THEN** it returns `None` and logs a warning naming the `[gcp]` extra,
  rather than raising `ImportError`

### Requirement: The orchestrator SHALL start and stop the metrics exporter and Firestore sync only when wired, in LIFO order

`MouseDroidOrchestrator` SHALL accept `cloud_metrics_exporter` and
`cloud_firestore_sync` as optional constructor parameters. When present,
`start()` SHALL start them (after `cloud_sink`/`cloud_experience_exporter`,
before the OTA weight-update pollers) and `stop()` SHALL stop them (before
`cloud_experience_exporter`/`cloud_sink`, after the OTA weight-update
pollers) — `cloud_metrics_exporter` via its `stop()` method,
`cloud_firestore_sync` via its `close()` method.

#### Scenario: neither wired

- **GIVEN** an orchestrator constructed without `cloud_metrics_exporter` or
  `cloud_firestore_sync`
- **WHEN** `start()` then `stop()` are called
- **THEN** neither call touches either collaborator (both stay `None`)

#### Scenario: both wired

- **GIVEN** an orchestrator constructed with fake `cloud_metrics_exporter`
  and `cloud_firestore_sync` collaborators
- **WHEN** `start()` is called
- **THEN** both collaborators' `start()` methods are invoked
- **WHEN** `stop()` is subsequently called
- **THEN** `cloud_metrics_exporter.stop()` and `cloud_firestore_sync.close()`
  are each invoked exactly once, with `cloud_firestore_sync` before
  `cloud_metrics_exporter` (LIFO relative to their start order)

### Requirement: `CloudLoggingSink`'s lifecycle SHALL be driven by `main.py`, independent of the orchestrator, and SHALL NOT block startup, shutdown, or health checks on a Cloud Logging failure

`main.py::cli_entry()` SHALL build at most one `CloudLoggingSink` instance
per process and thread it into both `configure_logging()` and whichever of
`_run()`/`_health_check()` is invoked. Both functions SHALL wrap the sink's
`start()` and `close()` calls in `try/except Exception`, logging and
continuing on failure, and SHALL invoke `close()` on every exit path
(including a `_health_check()` failure that calls `sys.exit(1)`).

#### Scenario: default path touches nothing

- **GIVEN** `cfg.gcp` is `None` (the default), so `build_cloud_logging_sink`
  returns `None`
- **WHEN** `_run()` or `_health_check()` is called with
  `cloud_logging_sink=None`
- **THEN** the function completes exactly as it did before this change,
  performing no cloud-logging-related calls

#### Scenario: a Cloud Logging start failure does not block the run

- **GIVEN** a `cloud_logging_sink` whose `start()` raises
- **WHEN** `_run()` is called
- **THEN** the orchestrator still starts and runs normally
- **AND** the failure is logged, not raised

#### Scenario: the sink is closed even when the guarded body exits early

- **GIVEN** a `cloud_logging_sink` and a `_health_check()` call whose
  orchestrator health check reports a non-`"ok"` status (triggering
  `sys.exit(1)`)
- **WHEN** `_health_check()` runs to completion
- **THEN** `cloud_logging_sink.close()` was called before the process exits

#### Scenario: `configure_logging` actually forwards events, not just identity

- **GIVEN** `configure_logging(cfg.logging, cloud_logging_sink=sink)` where
  `sink` is a Protocol-conforming fake
- **WHEN** a log event is emitted via `get_logger(...).info(...)`
- **THEN** the fake's `__call__` recorded that event — not merely that the
  same object identity was threaded through
