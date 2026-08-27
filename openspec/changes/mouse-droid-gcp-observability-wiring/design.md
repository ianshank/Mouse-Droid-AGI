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

## D-7. Post-merge hotfix — genuine SDK-availability detection

**Context:** the first PR for this bundle (#206) was merged by the operator
before a Copilot review round landing while it was still open could be
acted on. Per this repo's own convention for a merged designated branch,
remediation shipped as a fresh hotfix on a restarted branch, not a reopened
PR. Every finding below was independently re-verified against the actual
merged source before being accepted, then re-verified again by a dedicated
review agent that pulled the real GitHub review rather than trusting any
paraphrase.

**The bug:** none of `cloud/logging_sink.py`, `cloud/monitoring_exporter.py`,
`cloud/firestore_sync.py` — nor the two pre-existing siblings,
`cloud/pubsub_sink.py`, `cloud/experience_exporter.py` — imports
`google.cloud.*` at module scope; every one defers that import into its own
`start()` method. So every builder's `except ImportError` around the
lightweight `mousedroid.cloud.<x>` wrapper import (D-2's precedent idiom)
never fired for a genuinely-missing SDK: `from mousedroid.cloud.monitoring_exporter
import CloudMetricsExporter` succeeds regardless of whether
`google-cloud-monitoring` is installed, since that import touches only the
wrapper module, not the SDK. Combined with `_start_cloud_subsystems`'s four
`.start()` calls being unwrapped (D-6 documents the LIFO *order*; none of
the four was exception-safe), an operator enabling
`gcp.monitoring.enabled`/`gcp.firestore.enabled` — or, latently, the two
pre-existing `gcp.pubsub.enabled`/`gcp.storage.enabled` toggles — without
installing the `[gcp]` extra got a rover that failed to boot entirely, not
the graceful degrade every `cloud_*_not_available` log event implies.

**The fix, deliberately widened to all five builders, not just the three
new ones** (an independent review agent's scope correction — a fix
touching only two of `_start_cloud_subsystems`'s four identical calls, or
only three of five builders sharing one `except ImportError` idiom, would
leave an inconsistent half-fixed contract sitting in the same method/idiom
this hotfix already touches):

- Each of `build_cloud_telemetry_sink`, `build_cloud_experience_exporter`,
  `build_cloud_logging_sink`, `build_cloud_metrics_exporter`,
  `build_cloud_firestore_sync` now calls
  `mousedroid.common.imports.module_available("google.cloud.<x>")` — a
  spec-only probe (no import side effect, consistent with these classes'
  own deliberately-deferred-import design) against the *actual* SDK
  submodule each concrete class's `start()` will need, before attempting
  the wrapper import. A spec-only check (not `module_importable`, which
  performs a real import) was chosen specifically because these builders
  never eagerly import the SDK today — probing with a real import at build
  time would change that property as a side effect of a detection fix.
- All four `_start_cloud_subsystems` `.start()` calls — `cloud_sink`,
  `cloud_experience_exporter`, `cloud_metrics_exporter`,
  `cloud_firestore_sync` — are now each wrapped in their own
  `try/except Exception: _log.warning(...)`, matching the pre-existing
  OTA-poller pattern three lines below in the same method.

**Test consequence, shipped in the same commit as the fix above (not a
follow-up — there is no intermediate state where both the fix and the old
tests are simultaneously correct):** this repo's CI installs no `[gcp]`
extra in any job, so `tests/integration/test_factory_integration.py::
test_build_orchestrator_threads_gcp_observability_collaborators` and three
"enabled path" cases in `tests/unit/factory/test_factory_cloud_observability.py`
previously passed only because the detection bug existed. Fixing detection
correctly flips all four to see `None`; each gained a
`pytest.importorskip("google.cloud.<x>")` guard so they skip cleanly in
this repo's own CI instead of failing. The three `_none_when_module_not_importable`
tests in the same factory file additionally gained a
`monkeypatch.setattr("mousedroid.factory.module_available", lambda name: True)`
line so they keep isolating the *wrapper*-`ImportError` branch specifically
— without it, the new, earlier SDK-availability guard would return `None`
first for an unrelated reason (the SDK genuinely absent in CI) and the test
would silently stop proving what it claims to.

**Also in this hotfix, unrelated to the boot-crash bug but confirmed
real during the same Copilot review round:** `MouseDroidOrchestrator.__init__`
gained a bare `*,` immediately before `tool_registry`, closing a
keyword-only-argument gap that predates F-032 (every param from `cognitive_core`
onward was already positionally callable) — F-032's own two new params just
happened to land in the exposed region. Confirmed zero exploitation via an
AST sweep of every `MouseDroidOrchestrator(` call site in the tree (all
45+ pass pure keyword arguments), so this is latent hardening, not a live-bug
fix. Three documentation/test-quality nits also landed: a third stale
`docs/architecture.md` `classDiagram` (the `class Factory` block) that the
pre-merge doc-reconciler pass missed; the Level 3d intro's blanket
"protected by CircuitBreaker" claim, now qualified to match the
already-accurate per-component Resilience table a few lines below; and the
`test_logging_sink.py`/`test_firestore_sync.py` arity-check tests, which
called `inspect.signature(member)` and discarded the result — now assert
the actual parameter count.
