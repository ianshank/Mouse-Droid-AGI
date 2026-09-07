# Proposal — Cloud Logging queue + allowlist (dormant footgun)

- change_id: mouse-droid-cloud-log-queue
- project: mouse-droid
- status: active
- feature_id: F-039
- epic: GCP Digital Twin
- owner: ianshank
- created: 2026-09-07
- rev: A

## Why

F-032 wired `CloudLoggingSink` into `configure_logging`. The class docstring
said "asynchronously" / "Fire-and-forget" while `__call__` called sync
`Logger.log_struct`. That is an invariant-5/10 defect in the class.

It is **not** a default 30 Hz stall. Healthy `tick()` emits
`_log.debug("tick_complete")`. `LoggingConfig.level` and
`GCPLoggingConfig.min_level` default INFO, so DEBUG never reaches the sink.
`gcp.logging.enabled` defaults False. The overlay that would stall must drop
**two** log levels to DEBUG.

## What Changes

- `__call__` filters, allowlists, redacts, `put_nowait` onto
  `GCPLoggingConfig.queue_maxsize` (default 256).
- Drain task in `start()`; SDK in `asyncio.to_thread`. Tests call `flush()`.
- Module-level `CLOUD_LOG_ALLOWED_KEYS` / `CLOUD_LOG_REDACTED_KEYS` (not YAML).
- Tests: INFO drops `tick_complete`; both-DEBUG forwards; `__call__` does not
  block when `log_struct` sleeps.

## Impact

Callers that asserted `log_struct` immediately after `__call__` must `flush()`.
Production `tick()` never flushes.

## Charter

No CHARTER §3 carve-out: default-OFF egress; no new actuation; no LLM in 30 Hz.

## Validation

`bash scripts/validations/F-039.sh`
