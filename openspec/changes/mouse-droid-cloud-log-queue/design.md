# Design — Cloud Logging queue + allowlist

## D-1. Queue, do not thread

AGENTS.md forbids `threading.Thread` for application logic. The drain is an
`asyncio.Task` spawned in `start()`. `queue.Queue.put_nowait` is the
non-blocking producer. SDK I/O runs in `asyncio.to_thread`.

## D-2. Honest stall test

The first ranking claimed one overlay away from stalling 30 Hz. Rebuttal:
`tick_complete` is DEBUG and both levels default INFO. The test that would
fail a stall exists, and it **only** fails when both levels are DEBUG. Default
INFO is pinned as a non-event.

## D-3. Allowlist is not YAML

A YAML-widenable key list would let an operator dump mission text into Cloud
Logging. `CLOUD_LOG_ALLOWED_KEYS` and `CLOUD_LOG_REDACTED_KEYS` are module
frozensets. Redaction wins.

## D-4. `flush()` is a test/close seam

Production `tick()` must never drain the queue. Tests that skip `start()`
call `flush()` after the processor if they assert on `log_struct`.
