# Design — GCP observability wiring

## D-1. Why no CHARTER.md §3 carve-out is needed

Applying `.claude/skills/charter-carveout/SKILL.md`'s three-question procedure
directly to this change:

1. **Motion?** No. All three components are read-side data egress (forwarding
   already-computed log events, metric samples, and episodic-memory records
   off-device). None constructs or issues a motor command.
2. **LLM inference or training inside the hot loop?** No. None of
   `CloudLoggingSink`, `CloudMetricsExporter`, or `CloudFirestoreSync`
   perform inference or training — they serialize and forward data that was
   already computed elsewhere. This is categorically different from the risk
   CHARTER §3 names (unbounded-latency LLM/training compute), which is why
   the two precedent builders (`build_cloud_telemetry_sink`,
   `build_cloud_experience_exporter` — themselves the same shape of
   off-device forwarding) never needed one either.
3. **Runtime behaviour changed by editing source, not config?** No. Every
   new code path is gated behind `gcp.logging.enabled` /
   `gcp.monitoring.enabled` / `gcp.firestore.enabled`, each defaulting
   `False` (`src/mousedroid/config/schema/gcp_cloud.py`). F-029
   (`mouse-droid-cloud-egress-default-off`) is the compensating control this
   depends on — its own `design.md` D-5 names this relationship explicitly:
   "This is the compensating control for the operator decision to wire
   `CloudLoggingSink` / `CloudMetricsExporter` / `CloudFirestoreSync`
   **without** a ratified CHARTER §3 carve-out." F-029 already merged
   (`9bd3dc7`), so the compensating control is in place before this bundle
   lands.

All three questions resolve "no" — proceeding without a carve-out, per the
skill's own "no carve-out needed" path, which requires this reasoning be
written down rather than silently assumed. Mirrored in `features.yaml`'s
F-032 `notes:` field.

## D-2. Why the null-collaborator guard is a genuinely new idiom step

`build_cloud_telemetry_sink` and `build_cloud_experience_exporter` — the two
precedent builders — take no required collaborator that can independently be
`None`; `metrics_registry` is an *optional* keyword on the former (used only
if present). `CloudMetricsExporter.__init__(self, cfg, registry:
MetricsRegistry)` and `CloudFirestoreSync.__init__(self, cfg, episodic:
EpisodicReplay)` both declare their second parameter non-Optional — the
concrete classes assume the collaborator exists.

But `build_metrics_registry(cfg)` returns `None` whenever
`cfg.metrics.enabled` is `False`, and `build_memory_tier(cfg)` returns `None`
whenever `cfg.memory.enabled` is `False` (the schema default). Neither flag
has a cross-field validator tying it to the corresponding `gcp.*.enabled`
flag — `gcp.firestore.enabled=true` with default `memory.enabled=false` is a
legal, unvalidated configuration today. Without an explicit guard, that
combination would crash `build_orchestrator()` outright rather than
producing the same graceful `None` the two precedent builders return for
every other disabled/misconfigured case.

The guard is placed after the `.enabled` check and before the
`try/except ImportError` block — cheapest check first, and there's no reason
to attempt importing the GCP SDK for a builder that's going to return `None`
regardless of whether the import would have succeeded.

## D-3. Why `CloudLoggingSink`'s lifecycle stays out of the orchestrator

`configure_logging()` runs synchronously inside `main.py::cli_entry()`,
*before* `_run()`/`_health_check()` (both `async`) ever call
`build_orchestrator()`. If `CloudLoggingSink` were built inside
`build_orchestrator()` like its three siblings, two disconnected instances
would exist: the one `configure_logging()` already has wired into the
structlog processor chain (built too early for the orchestrator to see) and
a second one the orchestrator would separately construct and start (never
inserted into any processor chain, so it forwards nothing). Keeping its
lifecycle entirely in `main.py` — one instance, built once, threaded through
both `configure_logging()` and the two entry-point functions — is the only
way to avoid that split.

This also means `CloudLoggingSink` does **not** get a
`MouseDroidOrchestrator.__init__` parameter, unlike the other three
components.

## D-4. Why `_health_check()` needed a new `try`/`finally`, not just a new call

Before this change, `_health_check()` was a flat sequence ending in
`sys.exit(1)` on failure — no `try`/`finally` at all, unlike `_run()`, which
already wraps `orch_obj.run()`/`.stop()` in one. Appending a bare
`await cloud_logging_sink.close()` after the existing body would never
execute on the failure path, since `sys.exit(1)` (a `SystemExit`) would
already have unwound the stack. The fix wraps the orchestrator-build-and-run
body in its own `try`/`finally`, so `close()` runs on every exit path
including `sys.exit(1)` — `finally` blocks always run before a
`SystemExit`/`BaseException` propagates further. `start()`/`close()` are each
independently wrapped in their own `try/except Exception` (not the
`try/finally`'s exception path) so a Cloud Logging outage is logged and
swallowed rather than blocking or failing the check itself — mirroring the
existing OTA weight-update poller's boot-resilience pattern one function
away in `orchestrator.py::_start_cloud_subsystems`.

**Addendum, found during this bundle's own review round:** `_run()`'s
narrower wrapping (only `orch_obj.run()`/`.stop()`) turned out to have the
identical gap in miniature — `build_orchestrator()`, the `isinstance` check,
and `orch_obj.start()` all ran outside any try/finally, so a failure there
after `cloud_logging_sink.start()` already succeeded left it never closed.
Fixed the same way: an outer try/finally now wraps the whole body from
`build_orchestrator()` onward, with the pre-existing `orch_obj.run()`/
`.stop()` try/finally nested unchanged inside it (so `orch_obj.stop()`'s own
semantics — not called if `orch_obj.start()` itself failed — are untouched;
only the `cloud_logging_sink` close guarantee widens). Confirms this
addendum's own point: independent verification, not assumed symmetry, is
what catches this class of gap — three separate review agents converged on
the same finding here.

## D-5. Why `CloudLoggingSinkProtocol` gained `start`/`close`

The protocol previously declared only `__call__` — sufficient for
`configure_logging()`, which treats the sink purely as a structlog
processor. But `main.py` needs the lifecycle too. Rather than invent a
second, narrower protocol just for `main.py`'s view, `CloudLoggingSinkProtocol`
was widened to describe the concrete `CloudLoggingSink` class's full public
shape (`start`, `__call__`, `close`) — consistent with every other protocol
in this file, each of which already describes its concrete class's complete
contract rather than a partial slice. Caught by `mypy --strict`: annotating
`main.py`'s new parameters as `CloudLoggingSinkProtocol | None` and calling
`.start()`/`.close()` on them failed type-checking against the original,
narrower protocol.

## D-6. LIFO teardown ordering

`_start_cloud_subsystems` now starts, in order: `cloud_sink`,
`cloud_experience_exporter`, `cloud_metrics_exporter`,
`cloud_firestore_sync`, then the OTA weight-update pollers.
`_stop_cloud_subsystems` already stopped its two precedent components in
the reverse of their start order (pollers → experience_exporter → sink) —
the two new components are inserted into the stop sequence maintaining that
same reversal: pollers → `cloud_firestore_sync` → `cloud_metrics_exporter` →
`cloud_experience_exporter` → `cloud_sink`.
