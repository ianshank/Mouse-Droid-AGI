# Phase 6 — On-Device Incremental Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. This is a **multi-sprint** feature — execute in its own git worktree per workstream.

**Goal:** Let the rover update its own policy/world-model weights *between* cloud retraining cycles from fresh on-device experience, gated by a SHA-256 integrity contract and a safety-regression bound that reverts to cloud weights on underperformance.

**Architecture:** A new `learning/on_device/` subsystem wires the existing continual-learning primitives (`learning/ewc.py`, `learning/progressive.py`) and the replay loop (`training/replay/lmdb_reader.py` + `training/replay/mixer.py` — there is NO `harness/replay_buffer.py`) into an *online update path* that runs OUTSIDE the 30 Hz reactive loop. On-device-updated weights land in a **separate slot** from cloud-pulled weights (per ADR-010) so the orchestrator can A/B between them; a regression gate on a held-out replay sample is authoritative and reverts to cloud weights (emitting a new Prometheus counter) when the updated policy underperforms.

**Tech Stack:** Python 3.10/3.11, PyTorch, Pydantic v2 config, structlog, the existing EWC/PNN + replay + cloud-OTA (C1) machinery. Default-OFF, backwards-compatible.

---

## Context — why

Tier C closed the *cloud* loop: the rover pulls cloud-retrained weights via the C1 `HuggingFaceWeightUpdatePoller` (SHA-256-verified) and runs closed-loop missions. The remaining autonomy gap (`docs/planning/NEXT_STEPS.md:75-102`) is **on-device** incremental learning — adapting between cloud cycles without a GCP round-trip, which matters for an off-network rover that accumulates fresh experience faster than the cloud cadence. The risk is regression: an unsupervised on-device update can degrade a cloud-validated policy. This plan makes on-device learning **safe by construction** (separate slot + regression gate + auto-revert) and **observable** (new counter), defaulting OFF so existing deployments are byte-identical.

**Prerequisite (shipped):** `training/upload_weights.py::sync_gcs_to_hf` (Tier C Closeout) — the cloud loop is fully closed (GCS → HF Hub), so on-device updates have a canonical cloud baseline to A/B against and revert to.

---

## Non-negotiable invariants (inherit from CLAUDE.md + ADR-010)

- **30 Hz loop stays LLM/training-free.** The online update step runs on the slow cadence / a background task, never inside `tick()`'s hot path. `torch.no_grad()` still governs all inference.
- **Default OFF + backwards-compatible.** New config under a single `OnDeviceLearningConfig` (Optional, `enabled: bool = False`). Existing YAML loads unchanged; a deployment with the flag off is byte-identical.
- **Separate weight slot.** On-device weights NEVER overwrite the cloud-pulled slot. The orchestrator selects between `{cloud, on_device}` slots; revert = point back at `cloud`.
- **SHA-256 integrity** reused from C1 (per ADR-010) for any on-device checkpoint written/loaded.
- **Safety gate is authoritative.** No on-device weight is promoted to active without passing the held-out regression bound. Failure → revert + counter.
- **Protocol-DI + factory.** New components are `@runtime_checkable Protocol`s built only in `factory.py`. `mypy --strict` clean; ≥85% coverage; full test-tier mirror.

---

## Workstream / task breakdown (each WS is a shippable PR)

### WS1 — Config schema + safety counter (foundation)
**Files:** `src/mousedroid/config/schema.py`, `src/mousedroid/telemetry/metrics.py`, regression + AQA tests.

- [ ] Add `OnDeviceLearningConfig` Pydantic model (Optional on `Settings`, default `None`): `enabled: bool = False`, `trigger_min_new_records: int`, `update_steps: int`, `regression_tolerance: float` (ge=0), `held_out_fraction: float` (0..1), `slot_dir: str`, `ewc_lambda: float`, `learning_rate: float`. Every value config-driven — NO hardcoded thresholds. **`slot_dir` is NOT an absolute host path.** Default it to an experience-root-relative leaf (e.g. `"on_device_slot"`) and have the orchestrator/factory resolve it UNDER the existing configured experience root `cfg.experience.path` (`ExperienceConfig.path` in `src/mousedroid/config/schema.py`) — i.e. the on-device weight slot lives at `<cfg.experience.path>/on_device_slot`. Do NOT hardcode `/home/jetson/...`; that absolute path is already (and only) the operator-overridable default of `ExperienceConfig.path`, so deriving from it inherits any operator override for free.
- [ ] Add the new Prometheus family `{ns}_on_device_learning_reverted_total{reason}` to `MetricsRegistry` (pure-add, gated, low-cardinality `reason` frozenset: `{regression_bound, integrity_mismatch, exception}`); seed in `generate_metrics_sample()`.
- [ ] Regression test: existing YAML loads unchanged with the field absent. AQA test: field-hygiene + counter label cardinality. **TDD: failing test → minimal impl → green.**

### WS2 — On-device update protocol + EWC/PNN online path
**Files:** `src/mousedroid/learning/on_device/__init__.py`, `protocol.py`, `ewc_online.py`; extend `learning/ewc.py` / `learning/progressive.py` with online-update entry points.

- [ ] Define `OnDeviceLearnerProtocol` (`update(batch) -> UpdateResult`, `snapshot() -> CheckpointRef`, `restore(ref)`). 
- [ ] Implement an EWC-regularized online updater that consumes replay batches and applies bounded gradient steps with the Fisher penalty (reuse `learning/ewc.py` Fisher machinery — do not reimplement). PNN column path optional behind the same protocol (`learning/progressive.py`).
- [ ] Unit tests: update produces a finite loss, EWC penalty applied, no NaN/Inf; property test (Hypothesis) over arbitrary batch shapes that the updater never corrupts the base weights in place.

### WS3 — Replay-triggered update loop (outside hot path)
**Files:** `src/mousedroid/learning/on_device/replay_trigger.py`; orchestrator slow-cadence seam; reuse the replay loop (`training/replay/lmdb_reader.py` + `training/replay/mixer.py`).

- [ ] Trigger fires when ≥ `trigger_min_new_records` fresh records accumulate (gated by the Tier-A replay/VLA/VLM telemetry already wired). Runs the updater for `update_steps`, writes to the **on-device slot** (SHA-256 stamped).
- [ ] Wire at the POST_TICK / slow-cadence seam ONLY (mirror the C2 mission-lifecycle seam ordering). Integration test through `build_orchestrator()` proving the DI graph and that the hot loop is untouched.

### WS4 — Safety-regression gate + auto-revert + A/B slot selection
**Files:** `src/mousedroid/learning/on_device/safety_gate.py`; orchestrator weight-slot selector.

- [ ] On a held-out replay sample (`held_out_fraction`), score the on-device-updated policy vs the cloud baseline. Promote only if it does NOT drop below `cloud_score - regression_tolerance`. Else revert to the cloud slot and `inc_on_device_learning_reverted_total("regression_bound")`.
- [ ] Integrity-mismatch and exception paths also revert + increment with the matching `reason`.
- [ ] Tests: promote-path, revert-path (each reason), and that a revert restores byte-identical cloud-slot behavior. Property test: across arbitrary score sequences, the active policy never stays below the regression bound for more than one evaluation window.

### WS5 — Factory wiring + operator runbook + docs
**Files:** `src/mousedroid/factory.py` (`build_on_device_learner(...)`, default-None), `docs/runbooks/jetson-on-device-learning.md`, `docs/architecture/c4-on-device-learning.md`, CHANGELOG.

- [ ] `build_on_device_learner` returns the protocol type, default-disabled; threaded into the orchestrator keyword-only. Factory-dispatch test pins the wiring.
- [ ] Runbook: how to enable, monitor the revert counter, and force-revert. C4 diagram of the slot/gate/trigger.

---

## Testing & verification (end-to-end)

```bash
ruff check src/ tests/ tools/ && ruff format --check src/ tests/
mypy --strict src/mousedroid/
pytest tests/unit/learning/on_device/ tests/integration/test_on_device_learning_integration.py \
       tests/property/test_on_device_safety_gate_property.py tests/regression/test_on_device_aqa.py \
       --import-mode=importlib
pytest --import-mode=importlib --cov=src/mousedroid --cov-fail-under=85
```

**Manual:** enable on a bench rover with a known-degrading synthetic experience stream; confirm the regression gate reverts and `{ns}_on_device_learning_reverted_total{reason="regression_bound"}` increments on `/metrics`; confirm the 30 Hz tick latency budget is unaffected (`tests/performance/`).

**Success = all gates green, default-off byte-identical, and a deliberately-bad update provably auto-reverts to the cloud slot.**

---

## Estimated scope

3–4 sprints (WS1 ~0.5, WS2 ~1, WS3 ~0.5, WS4 ~1, WS5 ~0.5). Sequence WS1→WS2→WS3→WS4→WS5; WS1 unblocks all. Module names verified against the tree (WS0 reconciliation): `learning/ewc.py` ✓, `learning/progressive.py` ✓ (NOT `progressive_net.py`), `training/replay/{lmdb_reader,mixer}.py` ✓ (there is NO `harness/replay_buffer.py`). Re-confirm before wiring if the tree drifts.
