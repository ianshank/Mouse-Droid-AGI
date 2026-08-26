# Proposal — GCP observability wiring

- change_id: mouse-droid-gcp-observability-wiring
- project: mouse-droid
- status: in progress
- feature_id: F-032
- epic: GCP Digital Twin
- owner: ianshank
- created: 2026-08-23
- basis_commit: 5e9de98
- rev: A

## Why

`CloudLoggingSink`, `CloudMetricsExporter`, and `CloudFirestoreSync`
(`src/mousedroid/cloud/{logging_sink,monitoring_exporter,firestore_sync}.py`)
were all three fully implemented and unit-tested, but none was ever
constructed by `factory.py` or threaded through `orchestrator.py`/`main.py`.
The same gap already existed for Pub/Sub telemetry and GCS experience export
before `build_cloud_telemetry_sink`/`build_cloud_experience_exporter` closed
it — this bundle closes it for the remaining three GCP data channels using
the identical mechanism.

An operator who configured `gcp.logging.enabled` / `gcp.monitoring.enabled` /
`gcp.firestore.enabled` today got nothing: the flags existed and validated,
but no code path ever read them to build anything. Wiring them turns a
silently-inert config surface into a working one.

## What Changes

- `cloud/protocol.py`: new `CloudFirestoreSyncProtocol` (`start`/`sync_once`/
  `close`, matching `CloudFirestoreSync`'s shape). `CloudLoggingSinkProtocol`
  widened from a bare `__call__` to also declare `start`/`close` — `main.py`
  drives that lifecycle directly, so the protocol needs to describe it.
- `factory.py`: three new builders — `build_cloud_logging_sink`,
  `build_cloud_metrics_exporter`, `build_cloud_firestore_sync` — following
  the existing `build_cloud_telemetry_sink`/`build_cloud_experience_exporter`
  idiom (`cfg.gcp is None` guard → `.enabled` guard → function-scoped
  `try/except ImportError` → construct → log → return), plus a fifth step
  the two precedent builders never needed: an explicit null-collaborator
  guard. `CloudMetricsExporter.__init__`'s `registry` and
  `CloudFirestoreSync.__init__`'s `episodic` are both non-Optional, while
  `build_metrics_registry`/`build_memory_tier` can each independently return
  `None` — an unvalidated but legal config (e.g.
  `gcp.firestore.enabled=true` with the default `memory.enabled=false`)
  would otherwise crash `build_orchestrator()` outright.
- `orchestrator.py`: two new constructor params, `cloud_metrics_exporter`
  and `cloud_firestore_sync`, wired into the existing
  `_start_cloud_subsystems`/`_stop_cloud_subsystems` helpers in LIFO order.
  `cloud_metrics_exporter` exposes `.stop()`, not `.close()` — the one
  asymmetry among the four cloud collaborators.
- `main.py`: `CloudLoggingSink`'s lifecycle stays outside the orchestrator
  by design — `configure_logging()` runs synchronously in `cli_entry()`
  before `build_orchestrator()` is ever called, so folding it into the
  orchestrator would create two disconnected instances. `cli_entry()` builds
  it once and threads the same instance into `configure_logging()` (whose
  `cloud_logging_sink` parameter and processor-chain insertion already
  existed and were already tested) and into new keyword parameters on
  `_run()`/`_health_check()`. Both functions wrap `.start()`/`.close()` in
  `try/except Exception` independently — `_health_check()` previously had
  no `try`/`finally` at all, so it gained one specifically to guarantee the
  close-on-exit path runs even through its existing `sys.exit(1)` branch;
  `_run()`'s own `try/finally` was similarly widened during review (see
  `design.md` D-4) to cover `build_orchestrator()`/the `isinstance` check/
  `orch_obj.start()` too, not just `orch_obj.run()`, so a downstream failure
  after `cloud_logging_sink.start()` already succeeded still closes it.
- `orchestrator/CLAUDE.md`: one stale line-number reference
  (`factory.py::build_orchestrator` `:4016` → `:4020`) corrected —
  an incidental touch caused by the new builders shifting `factory.py`'s
  line numbers.

## Impact

No behaviour change for any shipped config that leaves the three flags at
their default (`False`) — every new builder returns `None` immediately,
matching pre-change behaviour exactly. `config/gcp_digital_twin.yaml` is the
only shipped overlay setting any of these `True` (`logging`/`monitoring`,
not `firestore`); it now actually gets the egress it already asked for
instead of silently getting nothing.

No CHARTER.md §3 carve-out needed — see
`openspec/changes/mouse-droid-gcp-observability-wiring/design.md` D-1 for the
full three-question walkthrough (no motion, no LLM/training work, every new
path gated behind a default-`False` schema field). `features.yaml`'s F-032
`notes:` field records the same reasoning per
`.claude/skills/charter-carveout/SKILL.md`.

**Known, accepted risk, not fixed in this PR:** a six-agent adversarial review
round (security-scanner, test-engineer, peer-reviewer, config-guardian,
doc-reconciler, openspec-author) found that `CloudLoggingSink.__call__`
(pre-existing code, not touched by this bundle) makes a blocking synchronous
network call reachable from the 30 Hz hot loop once `gcp.logging.enabled` is
`true`, and forwards `event_dict` content with only a primitive-type filter.
Both dormant by default. Operator decision: ship as scoped, document the risk
(`features.yaml`'s F-032 `notes:`, `CHANGELOG.md`), fix `CloudLoggingSink`
separately — see `tasks.md`'s "Explicitly deferred" section.

## Spec Deltas

`openspec/changes/mouse-droid-gcp-observability-wiring/specs/gcp-observability/spec.md`

## Tasks

See `openspec/changes/mouse-droid-gcp-observability-wiring/tasks.md`.

## Validation

`bash scripts/validations/F-032.sh`
