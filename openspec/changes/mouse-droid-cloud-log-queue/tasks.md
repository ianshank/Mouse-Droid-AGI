# Tasks — Cloud Logging queue + allowlist

Quality gate:

```
python -m ruff check src/mousedroid/cloud/logging_sink.py src/mousedroid/config/schema/gcp_cloud.py tests/unit/cloud/test_logging_sink.py tests/unit/cloud/test_logging_sink_tick_level.py tests/regression/test_f039_aqa.py tests/regression/test_f039_backwards_compat.py
bash scripts/validations/F-039.sh
```

**Phase 1 — Queue**

- [x] 1.1 `__call__` never calls the SDK; bounded `queue.Queue`.
- [x] 1.2 `start()` spawns `_drain_loop`; SDK in `asyncio.to_thread`.
- [x] 1.3 `GCPLoggingConfig.queue_maxsize` default 256.

**Phase 2 — Allowlist**

- [x] 2.1 Module frozensets; redaction wins; event name is Cloud `message`.
- [x] 2.2 Existing filter test retargeted to `elapsed_ms` / `emergency`.

**Phase 3 — Honest tests**

- [x] 3.1 INFO drops `tick_complete`; both-DEBUG forwards after flush.
- [x] 3.2 `__call__` elapsed < 50 ms when `log_struct` sleeps 300 ms.
- [x] 3.3 `scripts/validations/F-039.sh` + `features.yaml` F-039.
