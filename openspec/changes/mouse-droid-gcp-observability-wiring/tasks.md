# Tasks — GCP observability wiring

Quality gate for every task below, run before it is ticked:

```
python -m ruff check src/ tests/ tools/ && python -m ruff format --check src/ tests/ tools/
python -m mypy src/ --strict --ignore-missing-imports
bash scripts/validations/F-032.sh
```

Task ordering is binding: each task lands green before the next starts.

**Phase 1 — Core wiring**

- [x] 1.1 `cloud/protocol.py`: new `CloudFirestoreSyncProtocol`
      (`start`/`sync_once`/`close`). `CloudLoggingSinkProtocol` widened to
      also declare `start`/`close`.
- [x] 1.2 `factory.py`: top-level `TYPE_CHECKING` imports extended
      (`CloudFirestoreSyncProtocol`, `CloudLoggingSinkProtocol`,
      `CloudMetricsExporterProtocol`, `EpisodicReplay`). Three new builders
      — `build_cloud_logging_sink`, `build_cloud_metrics_exporter`,
      `build_cloud_firestore_sync` — each following the existing
      `build_cloud_telemetry_sink` idiom plus the null-collaborator guard
      (design.md D-2) for the latter two.
- [x] 1.3 `build_orchestrator`: constructs the two new builders after
      `cloud_experience_exporter`, threading `memory_tier.episodic` (or
      `None`) into `build_cloud_firestore_sync`; passes both into
      `MouseDroidOrchestrator(...)`.
- [x] 1.4 `orchestrator.py`: `TYPE_CHECKING` import extended
      (`CloudFirestoreSyncProtocol`, `CloudMetricsExporterProtocol`).
      Constructor gains `cloud_metrics_exporter`/`cloud_firestore_sync`
      params + docstring entries + `self._cloud_*` assignments.
      `_start_cloud_subsystems`/`_stop_cloud_subsystems` updated with LIFO
      ordering (design.md D-6) — `cloud_metrics_exporter.stop()`, not
      `.close()`.
- [x] 1.5 `main.py`: `TYPE_CHECKING` import of `CloudLoggingSinkProtocol`.
      `cli_entry()` builds the sink once via `build_cloud_logging_sink` and
      threads it into `configure_logging()` and both `_run()`/
      `_health_check()`. Both functions gain a `cloud_logging_sink`
      keyword param (default `None`) with `start()`/`close()` independently
      wrapped in `try/except Exception`; `_health_check()` gains a
      `try`/`finally` it previously had none of (design.md D-4), so
      `close()` runs even through its `sys.exit(1)` branch.
- [x] 1.6 Full quality gate green on the four touched files plus a whole-
      `src/` `mypy --strict` and `ruff check` sweep (catches any other
      caller affected by the widened `CloudLoggingSinkProtocol` or the new
      orchestrator params) — clean, zero unrelated regressions.

**Phase 2 — Test suite**

- [x] 2.1 Unit-tier tests for the three new factory builders in
      `tests/unit/factory/test_factory_cloud_observability.py` (14 cases):
      disabled-by-default, `.enabled=True` construction, the two
      null-collaborator-guard cases, and a simulated-`ImportError` case per
      builder (`monkeypatch.setitem(sys.modules, "mousedroid.cloud.<x>",
      None)`). Integration-tier factory-composition coverage lands in 2.6a
      below, added during the review round.
- [x] 2.2 Orchestrator-tier tests in `tests/unit/orchestrator/
      test_cloud_subsystems_wiring.py` (5 cases, `AsyncMock` fakes):
      `_start_cloud_subsystems`/`_stop_cloud_subsystems` correctly no-op
      when unwired, correctly call `start`/`stop`/`close` (matching each
      protocol's real teardown method name) when wired, and assert both
      start- and stop-order (LIFO).
- [x] 2.3 `main.py` tests in `tests/unit/test_main.py` (11 cases,
      `_run`/`_health_check` exercised as `async def` tests; `cli_entry()`
      exercised as a synchronous test per 2.4 below):
      - default path (`cloud_logging_sink=None`) completes unchanged
      - a fake sink whose `start()` raises does not block `_run()`/
        `_health_check()`
      - the sink's `close()` is called on every exit path, including
        `_health_check()`'s `sys.exit(1)` branch and a downstream
        `build_orchestrator()` failure after `start()` already succeeded
- [x] 2.4 `test_cli_entry_threads_one_sink_instance_into_configure_logging_and_run`
      (`tests/unit/test_main.py`) — a synchronous test driving `cli_entry()`
      directly, proving the one sink instance it builds reaches both
      `configure_logging()` (asserting a live forwarded event, not just
      identity — `configure_logging`'s own processor-chain-insertion
      correctness is separately covered by `tests/unit/logging/
      test_setup.py`) and `_run()`'s start/close lifecycle.
- [x] 2.5 Protocol-conformance tests for `CloudFirestoreSyncProtocol`
      (`tests/unit/cloud/test_firestore_sync.py`) and the widened
      `CloudLoggingSinkProtocol` (`tests/unit/cloud/test_logging_sink.py`)
      following `tests/regression/test_f025_aqa.py`'s pattern — `isinstance`
      as a smoke check plus an explicit per-member callable/arity check —
      not the bare-`isinstance()`-only anti-pattern already flagged by
      `test-tier-mirror/SKILL.md` in `test_pubsub_sink.py`/
      `test_experience_exporter.py`.
- [x] 2.6 Regression/AQA tests: Protocol-shape and signature pins in
      `tests/regression/test_f032_cloud_wiring_aqa.py` (4 cases) and
      `tests/regression/test_f032_cloud_wiring_backwards_compat.py`
      (2 cases); the config-gating, null-collaborator, and twin-overlay
      behavioral coverage this task originally described actually lives in
      the extended `tests/regression/test_gcp_egress_defaults_aqa.py`/
      `test_gcp_egress_defaults_backwards_compat.py` (F-029's files,
      extended rather than duplicated) plus 2.1/2.2 above — reworded here
      to match reality, per the review round's finding.
- [x] 2.6a *(added during review)* Integration-tier test in
      `tests/integration/test_factory_integration.py` —
      `build_orchestrator()` end-to-end with GCP enabled, proving
      `cloud_metrics_exporter`/`cloud_firestore_sync` reach the resulting
      orchestrator through the real factory composition path, not just
      their own builders (2.1) or the orchestrator constructor called
      directly with fakes (2.2). Closes a gap the two precedent builders
      (`build_cloud_telemetry_sink`/`build_cloud_experience_exporter`) still
      have, left open for them as pre-existing and out of this bundle's
      scope.

**Phase 3 — Catalog**

- [x] 3.1 F-032 entry in `features.yaml` (`status: "in_progress"`,
      `implemented_in: null` until the dedicated closeout PR).
- [x] 3.2 `scripts/validations/F-032.sh`.
- [x] 3.3 Register in `openspec/project.md`.

## Explicitly deferred (separate change, do not fold in)

- `[gcp]` extras installed in zero CI jobs (same-shaped gap as F-034 fixes
  for `[mlflow]`) — future F-number, named in the plan this bundle comes
  from.
- Flipping `features.yaml` to `status: "done"` with a real `implemented_in`
  SHA — a dedicated closeout PR after merge, per the F-030/F-031 convention.
- **Hardening `CloudLoggingSink`'s I/O model** (non-blocking `log_struct()`
  dispatch — it currently bypasses the SDK's `BackgroundThreadTransport`,
  making it a blocking call reachable from the 30 Hz hot loop once
  `gcp.logging.enabled` is `true` — plus content filtering on forwarded
  `event_dict` values). Found by this bundle's own review round; deliberately
  not fixed here since `CloudLoggingSink` is pre-existing code outside this
  bundle's file scope and a redesign needs its own design/test cycle.
  Operator decision: ship F-032 as scoped, document the risk (recorded in
  `features.yaml`'s F-032 `notes:` and `CHANGELOG.md`), fix separately —
  not yet assigned an F-number.
