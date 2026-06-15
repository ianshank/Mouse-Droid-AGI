# Runbook — Jetson On-Device Incremental Learning (Phase 6)

Let the rover refine its own policy/world-model weights **between** cloud
retraining cycles from fresh on-device experience, gated by a world-model
held-out recon+KL-loss regression bound that auto-reverts to the cloud baseline
on underperformance. Plan:
`docs/superpowers/plans/2026-06-13-phase6-on-device-incremental-learning.md`.
Architecture: `docs/architecture/c4-on-device-learning.md`.

> ⚠️ **DO NOT enable this on the rover yet.** It is **functional** and
> sim-validated, but still **default-OFF**. The WS-E2/E3 learning + gate seams
> are CLOSED (#135): the learner refines the **live RSSM** and the gate scores
> it on a **held-out replay batch** by recon+KL loss. The one remaining gate
> before driving the running model is the WS-E4 live-model hot-swap activation
> (default-OFF `enable_hot_swap`) **plus a soak gate** — see
> [Enablement status](#enablement-status-functional-soak-gated). Read that
> section first.

## What the feature is

A new `learning/on_device/` subsystem runs a **bounded** RSSM refinement on
fresh replay experience at the slow-cadence / POST_TICK seam — on its own
background task, **OUTSIDE the 30 Hz reactive control loop**. The hot loop
(RSSM → MCTS → ESP32) stays deterministic and training-free; the only torch
work runs on a worker thread via `asyncio.to_thread`.

The mouse-droid has **no learned policy net** — action selection is MCTS
planning over the RSSM world model — so on-device learning refines the **RSSM
dynamics itself** (the model MCTS plans through), not a policy. The refiner is
**λ=0** in this v1: there is **no EWC penalty term** (`train_sequence`'s loss is
already `recon + kl_beta·KL`; the RSSM-native diagonal-Fisher EWC follow-up is a
clean seam for later, and `ewc_lambda` is accepted but ignored).

Flow (one slow-cadence cycle, when armed):

1. `ReplayTriggerCoordinator.maybe_update()` probes the **new-record** count
   (records that have arrived **since the last fired cycle drained the store**,
   not the absolute store size — see
   [Trigger arming](#trigger-arming-new-records-not-store-size)). If
   `< trigger_min_new_records`, it logs `on_device_trigger_below_threshold` and
   returns — byte-identical to a no-op cycle.
2. When armed (`on_device_trigger_fired`), it reads one `(B, T, …)` sequence
   batch and runs `RSSMRefiner.update()` (`on_device_refine_start` /
   `on_device_refine_complete`) for `update_steps` bounded `train_sequence`
   steps. The **base RSSM is never mutated** — a deep copy is refined and
   returned as a *candidate* (`on_device_candidate_produced`). The throwaway
   reconstruction heads (`RawModalityDecoders`) are refined jointly but **never
   persisted** — only the refined RSSM `state_dict` is.
3. The candidate state-dict is persisted to a SHA-256-stamped slot file under
   the experience root (`on_device_candidate_persisted`).
4. The `RegressionGate` scores the candidate AND the live baseline RSSM with
   the SAME fixed held-out batch + shared decoders + seed via the deterministic
   `score_dynamics` (held-out recon+KL loss, LOWER is better), then **promotes**
   (marks the slot active) or **reverts** (increments the revert counter). The
   live-model hot-swap itself is the `enable_hot_swap`-gated WS-E4 seam (see
   below); promotion today only records the active-slot pointer — **promotion is
   SEPARATE from activation**.

On-device-updated weights land in a **separate slot** from the cloud-pulled
weights (per ADR-010) — a revert simply leaves the live policy pointing at
the cloud baseline.

## The `on_device_learning:` config block

Defined by `OnDeviceLearningConfig` in `src/mousedroid/config/schema.py`,
wired as an `Optional` field on `Settings` (default `None` ⇒ disabled ⇒
existing YAML loads byte-identically). Every value is config-driven; nothing
is hardcoded.

| Field | Default | Bound | Meaning |
|---|---|---|---|
| `enabled` | `False` | — | Master switch. `False` (or block absent) ⇒ no coordinator is built, no background task is spawned. |
| `trigger_min_new_records` | `500` | `gt=0` | Minimum fresh experience records that must accumulate before a cycle fires. Also caps the replay scan. |
| `check_interval_s` | `300.0` | `gt=0` | Slow-cadence period (seconds) between trigger probes. The background task sleeps this long each tick. Defaults to 5 min so a default-on deployment never busy-polls the replay store. |
| `update_steps` | `50` | `gt=0` | Bounded gradient steps per update cycle. |
| `regression_tolerance` | `0.05` | `ge=0` | Maximum allowed recon+KL loss INCREASE above the baseline before the candidate is reverted. PROMOTE iff `candidate_loss <= baseline_loss + regression_tolerance` (lower-is-better). `ge=0` permits a zero-tolerance gate. |
| `held_out_fraction` | `0.1` | `gt=0, le=1` | Fraction of the replay sample reserved for held-out scoring. **Config seam, not wired into the WS-E3 gate** — the gate derives its disjoint held-out window from `refine_sequence_length`/`refine_batch_episodes` (the slice *after* the refine window), not from this fraction. Retained as a future seam for fraction-based held-out sizing. |
| `ewc_lambda` | `1.0` | `ge=0` | EWC Fisher-penalty strength. **Accepted but IGNORED in this λ=0 v1** — `RSSMRefiner` ships no EWC term (`train_sequence`'s loss is already `recon + kl_beta·KL`). Kept as a clean seam for the RSSM-native diagonal-Fisher EWC follow-up; the result metadata records `ewc_lambda: 0.0`. |
| `learning_rate` | `1e-4` | `gt=0` | Learning rate for the bounded gradient steps. |
| `slot_dir` | `"on_device_slot"` | relative, no `..` | Experience-root-**relative** leaf for the weight slot. Resolved as `<ExperienceConfig.path>/<slot_dir>` — NOT an absolute host path. A `field_validator` rejects absolute paths, `..` traversal, and empty values at YAML-load time. |
| `scoring_seed` | `1234` | — | Gate scoring: fixed RNG seed making the held-out recon+KL loss deterministic (same seed + held-out batch + weights ⇒ identical loss, so the promote/revert decision is reproducible). |

## How to enable

> Read [Enablement status](#enablement-status-functional-soak-gated) first. The
> steps below are the **mechanics** of enabling once the soak-gate criteria are
> met — not an instruction to enable now.

The example is a **runbook snippet only** — there is deliberately no
committed `config/*.yaml` overlay (the `config-compat` CI gate validates
changed `config/*.yaml` against the historically-deployed schema, which
predates `OnDeviceLearningConfig`; a committed overlay would fail that gate).

### Option A — YAML overlay block

Add to a production/bench overlay (e.g. a host-local copy of
`config/jetson_production.yaml`):

```yaml
on_device_learning:
  enabled: true
  trigger_min_new_records: 500
  check_interval_s: 300.0
  update_steps: 50
  regression_tolerance: 0.05
  held_out_fraction: 0.1
  ewc_lambda: 1.0
  learning_rate: 1.0e-4
  slot_dir: on_device_slot
  refine_sequence_length: 16
  refine_batch_episodes: 4
  scoring_seed: 1234
```

The revert counter is on by default (`metrics.track_on_device_learning: true`).

### Option B — environment overrides (per-host `docker.env`)

`pydantic-settings` maps the `MOUSEDROID_` prefix with `__` as the nested
delimiter. Because the block is `Optional`, the env path needs `enabled` plus
at least the fields you wish to override; the simplest enable is via a YAML
block (Option A) — env overrides then tune individual knobs:

```dotenv
MOUSEDROID_ON_DEVICE_LEARNING__ENABLED=true
MOUSEDROID_ON_DEVICE_LEARNING__CHECK_INTERVAL_S=600
MOUSEDROID_ON_DEVICE_LEARNING__REGRESSION_TOLERANCE=0.0
```

Per-host overrides live ONLY in `/etc/mousedroid/docker.env` — never commit
them (mirrors the `MOUSEDROID_LLM__*` discipline from
`docs/runbooks/jetson-claude-pilot-deploy.md`).

## Safety model

The gate is **authoritative**: no candidate is promoted without passing the
held-out regression bound.

- Both the candidate and the live baseline RSSM are scored by the SAME
  deterministic harness (`scoring.score_dynamics` — held-out recon+KL loss,
  LOWER is better) on the SAME fixed held-out batch + shared decoders +
  `scoring_seed`, so the decision is reproducible.
- **PROMOTE** iff `candidate_loss` is finite AND
  `candidate_loss <= baseline_loss + regression_tolerance`. On promote the slot
  is marked active (`active.json` pointer); the live model is never overwritten —
  the cloud-pulled slot is untouched.
- **REVERT** otherwise: the live policy stays on the cloud baseline and the
  counter increments with reason `regression_bound`. A **non-finite** candidate
  loss (a heavily-degraded candidate blows the KL up to NaN/+inf) is correctly
  `> baseline` ⇒ REVERT — the gate never assumes a finite loss. Integrity-
  mismatch (SHA-256 verify failure on slot load) maps to the `integrity_mismatch`
  reason; an update-path exception maps to `exception`.
- The candidate is the persisted slot loaded into a **deep copy** of the live
  RSSM, so the live model is **bitwise-unchanged** on BOTH revert AND promote —
  a promote only writes the `active.json` pointer; the running brain is never
  swapped here (that is the separate `enable_hot_swap` WS-E4 seam).

## Metric to watch

`{ns}_on_device_learning_reverted_total{reason}` (counter; `{ns}` =
`cfg.metrics.namespace`). Pure-add and gated by
`MetricsConfig.track_on_device_learning` (default `True`); the family is
omitted from `/metrics` until the first revert, so a default deployment
renders byte-identically. The `reason` label is a low-cardinality frozenset:

| `reason` | Meaning |
|---|---|
| `regression_bound` | Candidate held-out recon+KL loss exceeded `baseline_loss + regression_tolerance` (or was non-finite) — i.e. the candidate's dynamics prediction was worse than the live baseline by more than the tolerance. **LOWER loss is better.** |
| `integrity_mismatch` | A slot failed its SHA-256 verify on load. |
| `exception` | The update path raised. |

A healthy gate that is *doing its job* will show non-zero
`reason="regression_bound"` — that is the auto-revert protecting the
cloud-validated policy, not a fault. A spike in `integrity_mismatch` /
`exception` is the operator-actionable signal.

## Structlog events to grep

`docker logs mousedroid` / Loki. Family prefix `on_device_*`. The events below
are the **live RSSM-refinement (WS-E2/E3) + hot-swap (WS-E4) path** — the
authoritative one when `enabled: true`. (The pre-ENABLEMENT #134 Tensor-path
learner in `ewc_online.py` emits `on_device_update_start` /
`on_device_update_complete` instead, but the factory does **not** wire it for the
RSSM platform — do not grep for those on the rover.)

| Event | When |
|---|---|
| `on_device_update_loop_started` | Background task spawned (carries `interval_s`). |
| `on_device_trigger_below_threshold` | Probe found `< trigger_min_new_records` NEW records (DEBUG). |
| `on_device_trigger_fired` | Cycle armed (`new_records`, `threshold`, `update_steps`). |
| `on_device_trigger_empty_batch` | Armed but the batch was empty / store too sparse (WARN). |
| `on_device_refine_start` / `on_device_refine_complete` | The bounded RSSM refinement (`steps`, `learning_rate`, `task`, `seed`; `final_loss` on complete). |
| `on_device_candidate_produced` | Candidate produced (`n_steps`, `train_loss`, `batch_size`). |
| `on_device_slot_persisted` / `on_device_candidate_persisted` | Candidate written to its SHA-256-stamped slot (`digest`, `path`). |
| `on_device_consumed_offset_advanced` | The new-record baseline advanced after a fired cycle (DEBUG; `consumed_total`, `advanced_by`). See [Trigger arming](#trigger-arming-new-records-not-store-size). |
| `on_device_dynamics_score_computed` | Held-out recon+KL loss, per candidate + baseline (`loss`, `finite`, `seed`). |
| `on_device_candidate_promoted` | Gate PASSED — slot marked active (`candidate_loss`, `baseline_loss`, `delta`, `tolerance`, `digest`). `delta = candidate_loss - baseline_loss`; **positive ⇒ worse**. |
| `on_device_candidate_reverted` | Gate FAILED — reverted (WARN; `reason`, `candidate_loss`, `baseline_loss`, `delta`, `candidate_finite`). |
| `on_device_gate_skipped_no_held_out_batch` | No disjoint held-out window available — gate is a no-op, candidate left unpromoted (WARN). |
| `on_device_gate_slot_integrity_mismatch` | Slot failed SHA-256 verify in the gate runner — `integrity_mismatch` counted, no promote (WARN). |
| `on_device_slot_marked_active` | Active-slot pointer (`active.json`) written. |
| `on_device_update_cycle_failed` | A cycle raised; the slow loop logs and keeps running (WARN). |
| `on_device_refiner_unsupported_engine` | Live engine lacks `train_sequence` (e.g. `DualStreamRSSM` / ONNX) — refinement disabled at build (WARN). |

### WS-E4 hot-swap source events (only when `enable_hot_swap: true`)

The off-loop activation seam (`build_on_device_hot_swap_source` →
`OnDeviceWeightUpdateSource`) surfaces a promoted slot to the orchestrator's C1
atomic-swap seam. **Default-OFF** — these never appear unless `enable_hot_swap`
is on.

| Event | When |
|---|---|
| `on_device_hot_swap_source_wired` | The source was built (factory; `check_interval_s`, `device`). |
| `on_device_hot_swap_supersedes_cloud_world_model_poller` | The on-device source replaced the cloud OTA world-model poller for the swap slot (factory wiring; expected when hot-swap is on). |
| `on_device_hot_swap_source_started` / `on_device_hot_swap_source_stopped` | Background refresh loop lifecycle. |
| `on_device_hot_swap_pending` | A newly-materialised engine is awaiting the orchestrator swap (`digest`, `engine_type`). |
| `on_device_hot_swap_slot_integrity_mismatch` | The active slot failed SHA-256 on materialise — fail-closed, `integrity_mismatch` counted, live model untouched (WARN). |
| `on_device_hot_swap_refresh_failed` | The off-loop refresh raised; the loop logs and keeps running (WARN). |
| `on_device_hot_swap_swap_acknowledged` | The orchestrator applied the swap; the cached engine was evicted (DEBUG). |

Canonical triage greps:

```bash
docker logs mousedroid 2>&1 | grep -E \
  'on_device_candidate_(promoted|reverted)|on_device_dynamics_score_computed'
# hot-swap (only when enable_hot_swap: true):
docker logs mousedroid 2>&1 | grep -E \
  'on_device_hot_swap_(pending|slot_integrity_mismatch|refresh_failed|supersedes)'
```

### Trigger arming (new records, not store size)

The trigger arms on **NEW** experience — records that have arrived since the
last fired cycle drained the store — **not** the absolute store size. The
factory keeps an in-memory `consumed_offset`; `count_new_records` counts records
**beyond** that baseline, and the coordinator's `on_consumed` callback advances
it after a successful cycle (`on_device_consumed_offset_advanced`). Without this,
a store that has already crossed `trigger_min_new_records` would re-fire the
refine + gate **every cadence** on byte-identical stale data and never disarm.
The offset is **per-process / in-memory**: it resets on container restart, so
the first post-restart cycle re-fires once on the existing store, then disarms.

### Held-out-batch disjointness caveat

The gate's held-out batch is built **once** at coordinator construction over a
replay slice **DISJOINT** from the refine batch: the refine batch reads the
**first** `refine_batch_episodes × refine_sequence_length` records; the held-out
window is drawn from the records **after** it. If the store holds too few records
for a disjoint held-out window, the gate is a logged **no-op**
(`on_device_gate_skipped_no_held_out_batch`) and the candidate is left
unpromoted — a fresh / sparse Jetson never crashes; promotion simply waits for
more experience. (The refine batch itself is also pinned to that first window —
the refiner does **not** slide a cursor in this v1; a sliding-window refiner with
a re-derived disjoint held-out is separate future work.)

## The candidate slot on disk

The candidate state-dict is persisted to a content-addressed file at:

```
<ExperienceConfig.path>/<slot_dir>/<sha256-digest>.pt
```

with an `active.json` manifest (`{"active_digest": "<digest>"}`) alongside it
recording the blessed candidate. The digest stamps the filename (concurrent
candidates never collide) and is re-verified on load via the reused C1 OTA
helper (`utils.weights_manager.verify_sha256`, ADR-010). With the default
experience root and `slot_dir`, that is
`/home/jetson/mousedroid_experience/on_device_slot/<digest>.pt` — but the
slot is derived from `ExperienceConfig.path`, so any operator override of the
experience root is inherited for free, and `slot_dir` may never escape that
root (validator-enforced).

## Enablement status (functional, soak-gated)

This feature is **functional** (the learner refines the **live RSSM** and the
gate scores it on a **held-out replay batch** by recon+KL loss),
**sim-validated**, and **default-OFF**. The two original pre-enablement seams
were CLOSED by the WS-E2/E3 ENABLEMENT work (#135). The remaining gate before
driving the running model is the WS-E4 live-model hot-swap activation
(default-OFF `enable_hot_swap`) **plus a soak gate**. Do not flip
`enabled: true` (or `enable_hot_swap: true`) on the live rover before a soak gate
has passed.

### Seam (a) — refine the LIVE net, not a config-sized stand-in — CLOSED (#135)

- `build_on_device_coordinator` now wires `RSSMRefiner`
  (`learning/on_device/rssm_refiner.py`) over the **live RSSM world model**
  (deep-copied per update so the base stays bitwise-unchanged), refining the
  candidate via `train_sequence` over a `(B, T, ...)` replay sequence batch —
  the pre-ENABLEMENT `EWCOnlineLearner`-over-`nn.Linear` stand-in is gone.
- The gate-runner (`_build_on_device_gate_runner`) scores the candidate RSSM vs
  the live baseline RSSM by held-out recon+KL loss — there is no policy stand-in
  / adapter layer any more.

### Seam (b) — score on real held-out experience — CLOSED (#135)

- The gate scores against a FIXED held-out `(B, T, ...)` replay batch built over
  a slice DISJOINT from the refine batch (`_build_held_out_sequence_batch`), so
  the regression score reflects real experience the rover actually saw — the
  retired `manual_seed`-sampled seed-state path is gone.

### Remaining seam — live-model hot-swap activation (WS-E4)

- Promotion (`slot_store.mark_active`) stays SEPARATE from activation. Swapping a
  promoted slot into the running RSSM is the `enable_hot_swap`-gated
  `build_on_device_hot_swap_source` seam (default-OFF). Keep it off until the
  soak gate below passes.

### Soak-gate framing for enabling

Even after seams (a) and (b) close, treat enablement as a staged rollout:

1. **Sim soak** — run on a bench rover with a known-degrading synthetic
   experience stream; confirm `reason="regression_bound"` increments and the
   live model stays bitwise-unchanged on a revert. The deterministic sim-soak
   `tests/integration/test_on_device_sim_soak.py` drives the **full** pipeline
   (coordinator → real `RSSMRefiner` → slot store → recon-loss gate) and pins
   both a known-improving PROMOTE and a known-degrading REVERT through it
   end-to-end (plus `_tick_count == 0`); the property test
   `tests/property/test_on_device_auto_revert.py` fuzzes the decision surface.
2. **Shadow soak (≥30 days)** — enable with the live net wired but treat the
   active-slot pointer as advisory (do not hot-swap into the running policy);
   watch the revert-counter mix and the 30 Hz tick-latency budget
   (`tests/performance/`) for a sustained window (target **≥30 days** of
   real-rover operation) before trusting promotion to drive the live policy.
3. **Promote** — only then wire the live-policy hot-swap off the active-slot
   pointer.

## Verification (sim, off-rover)

```bash
ruff check src/ tests/ tools/ && ruff format --check src/ tests/
mypy --strict src/mousedroid/
pytest tests/unit/learning/on_device/ \
       tests/integration/test_on_device_sim_soak.py \
       tests/integration/test_pr134_on_device_integration.py \
       tests/integration/test_pr134_ws4_gate_integration.py \
       tests/integration/test_pr134_ws_e4_hot_swap_integration.py \
       tests/property/test_on_device_auto_revert.py \
       tests/property/test_on_device_no_inplace_corruption.py \
       tests/property/test_rssm_refiner_base_untouched.py \
       tests/regression/test_on_device_learning_aqa.py \
       --import-mode=importlib
```

Confirm a deliberately-bad update provably auto-reverts to the cloud slot and
`{ns}_on_device_learning_reverted_total{reason="regression_bound"}` increments,
while a known-improving refinement is PROMOTED (`active.json` written) — both
proven end-to-end by `tests/integration/test_on_device_sim_soak.py` with
`_tick_count == 0` (the 30 Hz hot loop never advances).

## Related runbooks / docs

- `docs/runbooks/jetson-claude-pilot-deploy.md` — the deliberative LLM tier
  (same `docker.env` per-host override discipline).
- `docs/runbooks/jetson-full-bringup.md` — full rover bring-up.
- `docs/architecture/c4-on-device-learning.md` — the C4 component view.
- `docs/architecture/ADR-010-cloud-weight-update-ota.md` — the SHA-256
  integrity + separate-slot contract this feature reuses.
