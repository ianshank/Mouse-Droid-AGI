# Cloud Logging queue

## Purpose

Make `CloudLoggingSink` fire-and-forget without claiming a default 30 Hz stall.

## Requirements

### Requirement: `__call__` SHALL NOT invoke the Cloud Logging SDK

The processor SHALL filter by `min_level`, copy allowlisted scalar keys,
redact mission/NL keys, and `put_nowait` onto a bounded queue. A full queue
SHALL increment `drop_count` and SHALL NOT block.

### Requirement: drain SHALL run off the caller thread

`start()` SHALL spawn an asyncio drain task. SDK `log_struct` SHALL run in
`asyncio.to_thread`. Tests MAY call `flush()` to drain on the caller thread.

### Requirement: default INFO SHALL drop per-tick DEBUG events

When `LoggingConfig.level` is INFO, structlog SHALL drop `tick_complete`
before the sink. Forwarding SHALL require both that level and
`GCPLoggingConfig.min_level` to be DEBUG.

### Requirement: allowlist SHALL NOT be YAML-widenable

`CLOUD_LOG_ALLOWED_KEYS` and `CLOUD_LOG_REDACTED_KEYS` SHALL be module-level
frozensets. Redaction SHALL win.
