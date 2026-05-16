# ADR-010 — Closed-Loop Cloud Retraining + OTA Weight Updates (Tier C1)

## Status

Accepted — 2026-05-16.

## Context

After PRs #85-#89 the rover ships LMDB experience shards to GCS and Pub/Sub.
The cloud side of the loop (Vertex AI retraining → HuggingFace Hub upload →
Jetson polls Hub → atomic engine swap) was missing. Operators need:

- A safety-critical integrity check on every downloaded artifact (SHA-256
  against a manifest in the same HF repo).
- An atomic swap that does **not** leave the orchestrator with two
  half-updated engines mid-tick.
- Observable Prometheus metrics for every stage of the loop so Grafana /
  alert rules can surface OTA failures within one scrape interval.
- A separate cloud container image — `Dockerfile.jetson` uses an
  L4T (ARM64) base that won't run on Vertex AI's x86_64 workers.

## Decision

### 1. Swap timing: **after** `_select_action`, **before** execute

The orchestrator's `tick()` body runs `_update_world_model(observation)`
BEFORE `_select_action(safety_ctx)`. If we swap mid-tick the latent
`(h, z)` produced by the OLD world-model would feed the NEW world-model's
`observe_step` on the NEXT tick — one-tick cross-model contamination that
cannot be eliminated by reference assignment alone.

We swap at one fixed seam in `tick()` — immediately after `_select_action`
returns — and reset `(h, z)` + `prev_action` + the latent recovery buffer
to zeros when the swap targets the world-model (default
`cfg.cloud.weight_update.reset_state_on_swap = True`). Trade-off is one
tick of context loss, which is acceptable for an OTA event operators
expect to happen at minute-scale, not 30 Hz.

Reference assignment in CPython is atomic at the interpreter level; the
orchestrator's `tick()` is single-coroutine on the event loop, so no lock
is needed. The new engine is fully materialised via the
`weight_update_loader` callback BEFORE the reference swap, so a loader
failure does NOT corrupt the live model.

### 2. SHA-256 integrity contract

Every HF Hub repo containing an OTA artifact MUST publish a sibling
`sha256.txt` (filename configurable via
`cfg.cloud.weight_update.sha256_manifest_filename`) carrying a single line
with the hex-encoded SHA-256 of the artifact. The poller downloads the
manifest, computes the digest of the freshly downloaded artifact, and
**refuses the swap** on any mismatch — incrementing
`mousedroid_cloud_weight_update_sha256_mismatches_total{repo_id}` and
emitting a `cloud_weight_update_sha256_mismatch` WARN.

`verify_sha256(local_path, expected_hex, log_event_prefix=...)` in
`src/mousedroid/utils/weights_manager.py` is the reusable building block.
Caller supplies the expected digest — never hardcoded.

### 3. Separate cloud Docker image

`docker/Dockerfile.cloud` uses
`pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` (x86_64) so it runs on
Vertex AI's `n1-standard-8 + NVIDIA_TESLA_T4` workers. The Jetson image
(`Dockerfile.jetson`) stays L4T-only.

### 4. Trainer idempotency

The Jetson exporter may upload shard N then crash before persisting its
high-water-mark, causing it to re-upload N on restart. The cloud trainer
checks a `shard_consumed_marker` GCS object before processing each shard
(via `--shard-consumed-marker-uri` on `train_offline_rl.py`). When the
marker exists the trainer short-circuits with a
`cloud_train_shard_already_consumed` log event.

### 5. Default OFF — byte-identical pre-Tier-C1 behavior

`cfg.cloud.weight_update.poll_interval_s` defaults to `0.0`. The factory
returns `None` from `build_weight_update_poller(cfg)` in that case, and
the orchestrator's `_apply_pending_weight_update()` short-circuits when
`self._weight_update_poller is None`. Existing YAML files load unchanged
and the Prometheus exposition output is byte-identical until the first
swap.

## Consequences

- **Positive**: Operators can deploy retrained policies fleet-wide without
  Jetson SSH. Every stage is observable; alert rules can page on
  SHA-256 mismatches (any non-zero rate is safety-critical).
- **Positive**: The swap seam is one location — adding a fifth
  `_select_action` branch in the future does not silently bypass the OTA
  loop (the swap runs in `tick()`, not inside `_select_action`).
- **Negative**: World-model swaps lose one tick of recurrent state by
  default. Operators that prefer continuity over correctness can set
  `cfg.cloud.weight_update.reset_state_on_swap = False`, accepting the
  cross-model contamination risk.
- **Negative**: The artifact-load callback (`weight_update_loader`) is
  engine-specific (ONNX session reload / TensorRT context refresh).
  Tier C1 ships the seam but defaults to `None` in production; the
  concrete loader lands in a follow-up PR once the HF Hub policy repo
  exists.

## Alternatives Considered

- **Lock the orchestrator during swap.** Rejected: `tick()` is already
  single-coroutine; a lock would add overhead without any safety benefit.
- **Swap inside `_select_action`.** Rejected: that function has four
  return sites today (cognitive / VLA / VLA-strict-timeout / nav_agent);
  inserting at one site silently misses the others. Wrapping the call
  site in `tick()` covers all branches at one seam.
- **Restart-on-update instead of in-process swap.** Rejected: 30 Hz
  control loop interruption is unacceptable; the safety monitor would
  fire emergency_stop during the restart window.
- **Skip the SHA-256 check, rely on TLS.** Rejected: TLS protects the
  transport, not the artifact at rest in HF Hub. A compromised maintainer
  account would let an attacker push a malicious policy — the SHA-256
  manifest is the defense in depth.

## References

- Plan: Tier C1 stories C1.1–C1.7 in
  `~/.claude/plans/please-create-a-comprehensive-sunny-hennessy.md`.
- Implementation: `src/mousedroid/cloud/weight_update_poller.py`,
  `src/mousedroid/orchestrator/orchestrator.py::_apply_pending_weight_update`,
  `src/mousedroid/utils/weights_manager.py::verify_sha256`,
  `src/mousedroid/telemetry/metrics.py` (4 new families).
- Tests: `tests/unit/cloud/test_weight_update_poller.py`,
  `tests/unit/test_weights_manager.py`,
  `tests/unit/orchestrator/test_weight_update_swap.py`,
  `tests/smoke/test_prometheus_format_tier_c.py`.
