# Peer review — Cloud Logging queue + allowlist

## Verdict table

| Claim | Verdict |
|---|---|
| `__call__` was sync `log_struct` despite "fire-and-forget" docs | **CONFIRMED** — F-032 accepted this as risk |
| Default overlays stall 30 Hz | **REJECTED** — `tick_complete` is DEBUG; both levels default INFO |
| Stall exists if both levels are DEBUG | **CONFIRMED** — pinned; `__call__` itself must not block after the queue |
| Mission text must not dump into Cloud Logging | **CONFIRMED** — redacted keys + non-YAML allowlist |

## Load-bearing pins

1. `GCPLoggingConfig.queue_maxsize` FieldInfo default is 256.
2. `test_default_info_levels_drop_tick_complete_before_sink`.
3. `test_both_debug_levels_forward_tick_complete`.
4. `CLOUD_LOG_REDACTED_KEYS` disjoint from `CLOUD_LOG_ALLOWED_KEYS`.
