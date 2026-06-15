# C4 Component — On-Device Incremental Learning (Phase 6)

> The on-device incremental-learning path. Fresh rover experience in the LMDB
> replay store triggers a **bounded** RSSM refinement at the
> slow-cadence / POST_TICK seam, producing a SHA-256-stamped *candidate*
> weight slot that a world-model recon+KL-loss regression gate either
> promotes (marks active) or reverts (increments a Prometheus counter). The
> 30 Hz reactive control loop (RSSM → MCTS → ESP32) is shown SEPARATELY and is
> deliberately **untouched** — the only torch work runs on a worker thread via
> `asyncio.to_thread`, never inside `tick()`. Default-OFF: with
> `cfg.on_device_learning` absent/disabled, no coordinator is built and the
> orchestrator is byte-identical to pre-Phase-6.

## Component Diagram

```mermaid
flowchart TB
    subgraph HotLoop["30 Hz reactive control loop — SEPARATE / UNTOUCHED"]
        Tick["MouseDroidOrchestrator.tick()\nRSSM -> MCTS -> ESP32\n(torch.no_grad, deterministic, training-free)"]
    end

    subgraph Slow["Slow-cadence background task (OUTSIDE the hot loop)"]
        Loop["_on_device_update_loop()\nsleep(check_interval_s) each tick"]
    end

    subgraph Replay["src/mousedroid/training/replay/"]
        LMDB[("LMDB replay store\n<experience.path>")]
        Reader["LMDBReplayReader\ncount_new_records / load_batch\n(async; blocking I/O)"]
    end

    subgraph OnDevice["src/mousedroid/learning/on_device/"]
        Coord["ReplayTriggerCoordinator.maybe_update()\ntrigger >= trigger_min_new_records (NEW records,\nnot store size — consumed_offset baseline)\nall torch work via asyncio.to_thread"]
        Learner["RSSMRefiner.update(batch)\n• deep-copy live RSSM -> candidate\n• update_steps @ learning_rate (lambda=0, no EWC)\n• train_sequence over (B,T,...) batch\n• base bitwise-UNCHANGED"]
        Store["OnDeviceSlotStore\npersist -> <digest>.pt\nmark_active -> active.json\nload re-verifies SHA-256"]
        Gate["RegressionGate.evaluate()\ncandidate RSSM vs baseline RSSM\nPROMOTE iff cand_loss <= base_loss + tolerance"]
        Scorer["score_dynamics()\nheld-out recon+KL loss (LOWER better)\nheld-out batch + decoders + scoring_seed"]
        HotSwap["OnDeviceWeightUpdateSource (WS-E4)\nenable_hot_swap-gated (default OFF)\nrefresh_once: materialise active slot\nOFF-loop via asyncio.to_thread\ntake_materialized: PURE ref (in tick)"]
    end

    subgraph WorldModel["src/mousedroid/world_model/ (REUSED)"]
        RSSM["RSSM\ntrain_sequence(batch, decoders)\n-> {loss, recon, kl, ...}"]
    end

    subgraph Factory["src/mousedroid/factory.py"]
        BuildCoord["build_on_device_coordinator(cfg, *, metrics=None)\nreturns None when absent/disabled"]
        BuildGate["_build_on_device_gate_runner(cfg, *, slot_store, metrics)"]
        BuildSwap["build_on_device_hot_swap_source(cfg, *, world_model, metrics)\nreturns None unless enable_hot_swap=True"]
    end

    subgraph Config["src/mousedroid/config/schema.py"]
        Cfg["OnDeviceLearningConfig (Optional, default None)\n• enabled / enable_hot_swap (default False)\n• trigger_min_new_records / check_interval_s\n• update_steps / learning_rate / ewc_lambda\n• regression_tolerance / held_out_fraction\n• refine_sequence_length / refine_batch_episodes / scoring_seed\n• slot_dir (relative, validator-gated)"]
        MetCfg["MetricsConfig.track_on_device_learning (default True)"]
    end

    subgraph Metrics["src/mousedroid/telemetry/metrics.py"]
        Counter["{ns}_on_device_learning_reverted_total{reason}\nreason in {regression_bound,\n  integrity_mismatch, exception}\n(pure-add, omitted until first write)"]
    end

    %% Build wiring
    Cfg -. "cfg.on_device_learning" .-> BuildCoord
    BuildCoord -- "if enabled" --> Coord
    BuildCoord --> BuildGate
    BuildGate -- "gate_runner=" --> Coord
    MetCfg -. "gates" .-> Counter

    %% Runtime trigger path
    Loop -- "maybe_update()" --> Coord
    Reader -- "count_new_records / load_batch\n(to_thread)" --> Coord
    LMDB --- Reader
    Coord -- "update(batch) (to_thread)" --> Learner
    Learner -- "candidate_state_dict" --> Store
    Store -- "persist -> <digest>.pt" --> SlotFS[("<experience.path>/<slot_dir>/<digest>.pt\n+ active.json")]

    %% Gate path
    Coord -- "gate_runner(slot) (to_thread)" --> Gate
    Gate -- "score_fn(candidate)\nscore_fn(baseline)" --> Scorer
    Scorer -- "train_sequence(batch, decoders)" --> RSSM
    Gate -- "PROMOTE -> mark_active" --> Store
    Gate -- "REVERT -> inc(reason)" --> Counter

    %% WS-E4 hot-swap activation path (default-OFF; only when enable_hot_swap=True)
    Cfg -. "enable_hot_swap" .-> BuildSwap
    BuildSwap -- "if enabled + RSSM" --> HotSwap
    Store -- "load_active() digest\n+ load() re-verifies SHA-256" --> HotSwap
    HotSwap -- "corrupt slot -> inc(integrity_mismatch)" --> Counter
    HotSwap -- "pre-materialised engine\n(take_materialized: PURE ref lookup)" --> Tick

    %% Hot-loop isolation: the ONLY edge into tick() is a pure reference
    %% assignment (WS-E4); ALL construction/IO happens off-loop in refresh_once.
    Tick -. "shares event loop ONLY;\ntorch work offloaded off it" .- Loop

    classDef hot fill:#fee2e2,stroke:#dc2626,color:#000
    classDef slow fill:#dbeafe,stroke:#3b82f6,color:#000
    classDef internal fill:#e0f2fe,stroke:#0284c7,color:#000
    classDef config fill:#f3e8ff,stroke:#9333ea,color:#000
    classDef reused fill:#dcfce7,stroke:#16a34a,color:#000
    classDef store fill:#fef3c7,stroke:#f59e0b,color:#000

    class Tick hot
    class Loop slow
    class Coord,Learner,Store,Gate,Scorer,HotSwap,BuildCoord,BuildGate,BuildSwap internal
    class Cfg,MetCfg,Counter config
    class RSSM reused
    class LMDB,Reader,SlotFS,Replay store
```

## One slow-cadence cycle — `ReplayTriggerCoordinator.maybe_update`

| Step | Action | Hot-loop isolation |
|---|---|---|
| 1. Trigger probe | `count_new_records()` via `asyncio.to_thread`. Below `trigger_min_new_records` ⇒ log `on_device_trigger_below_threshold`, return `None`. | Blocking LMDB scan off the event loop. |
| 2. Batch load | `load_batch()` via `asyncio.to_thread`. Empty batch ⇒ log `on_device_trigger_empty_batch`, return `None`. | Off the event loop. |
| 3. Bounded update | `learner.update(batch)` via `asyncio.to_thread`. Deep-copies the live RSSM, runs `update_steps` of `train_sequence` refinement; base parameters bitwise-unchanged. | Gradient work on a worker thread. |
| 4. Persist | `slot_store.persist(candidate)` → write-temp-then-rename to `<digest>.pt`. | Cheap I/O on the slow cadence. |
| 5. Gate | `gate_runner(slot)` via `asyncio.to_thread` — score candidate RSSM vs baseline RSSM by held-out recon+KL loss, PROMOTE or REVERT. `None` gate ⇒ byte-identical to pre-WS-E3. | Torch recon-loss scoring on a worker thread. |

## Promote / revert decision — `RegressionGate.evaluate`

| Condition | Outcome |
|---|---|
| `candidate_loss` finite AND `candidate_loss <= baseline_loss + regression_tolerance` | **PROMOTE**: `slot_store.mark_active(slot)` (write `active.json`); log `on_device_candidate_promoted`. Live model NOT overwritten — only the pointer is recorded. |
| otherwise | **REVERT**: do NOT mark active; `metrics.inc_on_device_learning_reverted("regression_bound")`; log `on_device_candidate_reverted` (WARN). |
| slot SHA-256 verify fails on load | **REVERT** with reason `integrity_mismatch` (`SlotIntegrityError`). |
| update path raises | slow loop logs `on_device_update_cycle_failed` and keeps running; reason `exception` reserved for the counter. |

Determinism is load-bearing: both losses come from `score_dynamics` on the SAME
held-out batch + shared decoders + `scoring_seed`, so the decision is reproducible.

## Trigger arming — new records, not store size

`count_new_records` counts records **beyond** an in-memory `consumed_offset`
baseline (the count of records a prior fired cycle already drained), NOT the
absolute store size. The coordinator's `on_consumed` callback advances the
offset after a successful cycle (`on_device_consumed_offset_advanced`), so the
trigger **disarms** until fresh experience accumulates past the baseline again.
Without this, a store already past `trigger_min_new_records` would re-fire the
refine + gate every cadence on byte-identical stale data. The offset is
per-process (resets on restart: the first post-restart cycle re-fires once, then
disarms).

## WS-E4 hot-swap activation (`enable_hot_swap`, default-OFF)

Promotion (`mark_active`) is SEPARATE from activation. `OnDeviceWeightUpdateSource`
is the activation seam — wired ONLY when `enable_hot_swap=True` (and the engine
is a concrete `RSSM`); otherwise the factory returns `None` and the orchestrator
is byte-identical to #134 (no swap path at all).

| Stage | Action | Hot-loop isolation |
|---|---|---|
| Off-loop refresh | `refresh_once()` (slow cadence): `load_active()` digest → `slot_store.load()` re-verifies SHA-256 → `build_world_model` + strict `load_state_dict` + device-place. A corrupt slot is **fail-closed** (`SlotIntegrityError` ⇒ `inc("integrity_mismatch")`, no pending update). | `torch.load` + construct via `asyncio.to_thread`. |
| Supersede / evict | A newer digest evicts the prior unacknowledged pending engine before publishing (and on the corrupt-supersede branch) so `_engine_by_update` never strands an unreachable engine. | Off the event loop. |
| In-`tick()` swap | The orchestrator's `_weight_update_loader` calls `take_materialized(update)` — a **PURE reference lookup** by `id(update)`, no I/O. | The only in-`tick()` touch; constant-time ref assignment. |

The source conforms to `WeightUpdatePollerProtocol` + an `engine_type`
(`world_model`) extension, so it slots into `orchestrator._weight_update_pollers`
and — when wired — **supersedes** the cloud OTA world-model poller for that slot
(`on_device_hot_swap_supersedes_cloud_world_model_poller`).

## Safety + de-hardcode contracts

- **Separate slot (ADR-010).** On-device candidates land at
  `<ExperienceConfig.path>/<slot_dir>/<digest>.pt` — NEVER overwriting the
  cloud-pulled slot. A revert simply leaves the live policy on the cloud
  baseline.
- **No absolute path hardcoded.** `slot_dir` is experience-root-relative and
  validated (`field_validator` rejects absolute / `..` / empty), so an
  operator override of the experience root is inherited for free.
- **SHA-256 integrity reused.** `OnDeviceSlotStore` digests with the C1 OTA
  helper (`utils.weights_manager.verify_sha256`); the digest stamps the
  filename and is re-verified on load.
- **Default-OFF + pure-add metric.** `cfg.on_device_learning` absent/disabled
  ⇒ no coordinator, no task, byte-identical orchestrator. The revert counter
  is gated by `MetricsConfig.track_on_device_learning` and omitted from
  `/metrics` until the first revert.
- **Hot loop untouched.** The WS-E2/E3 produce → gate path shares only the event
  loop with `tick()`, and all torch work is offloaded off it. The **single**
  edge into the `tick()` subgraph is the WS-E4 hot-swap `take_materialized` —
  and that is a **pure reference assignment** (the engine was already built
  off-loop in `refresh_once`), so no construction / disk I/O / device-placement
  ever runs inside `tick()`. With `enable_hot_swap=False` (the default) even that
  edge does not exist.

## Enablement status (WS-E2/E3 closed; WS-E4 gated)

These are documented in full in the operator runbook
(`docs/runbooks/jetson-on-device-learning.md`):

1. **Refines the LIVE RSSM (CLOSED, #135).** `build_on_device_coordinator` wires
   `RSSMRefiner` over the live RSSM (deep-copied per update), refining via
   `train_sequence`; the gate scores candidate RSSM vs baseline RSSM. The
   pre-ENABLEMENT `nn.Linear` stand-in + policy adapter are gone.
2. **Scores real held-out experience (CLOSED, #135).** The gate scores against a
   FIXED held-out replay batch DISJOINT from the refine batch — not
   `manual_seed`-sampled latents.
3. **Live-model hot-swap (WS-E4, default-OFF).** Activating a promoted slot into
   the running RSSM is the `enable_hot_swap`-gated `build_on_device_hot_swap_source`
   seam — keep it off until a soak gate passes.

See the runbook's "DO NOT enable on the rover yet" section and the ≥30-day
soak-gate framing.

## Related diagrams

- `docs/architecture/c4-overview.md` — Levels 1 (Context) and 2 (Container).
- `docs/architecture/c4-orchestrator.md` — the 30 Hz sense-plan-act loop that
  the slow-cadence task runs alongside.
- `docs/architecture/c4-rssm-sim-pretraining.md` — the RSSM world model reused
  by the recon-loss scorer.
- `docs/architecture/ADR-010-cloud-weight-update-ota.md` — the SHA-256
  integrity + separate-slot contract this feature reuses.
- `docs/architecture/c4-llm-gateway.md` — the deliberative LLM tier (separate
  concern; also OUTSIDE the hot loop).
