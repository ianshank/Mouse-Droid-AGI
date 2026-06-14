# C4 Component — On-Device Incremental Learning (Phase 6)

> The on-device incremental-learning path. Fresh rover experience in the LMDB
> replay store triggers a **bounded** EWC-regularized update at the
> slow-cadence / POST_TICK seam, producing a SHA-256-stamped *candidate*
> weight slot that a world-model rollout-return regression gate either
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
        Coord["ReplayTriggerCoordinator.maybe_update()\ntrigger >= trigger_min_new_records\nall torch work via asyncio.to_thread"]
        Learner["EWCOnlineLearner.update(batch)\n• deep-copy base -> candidate\n• update_steps @ learning_rate\n• EWC Fisher penalty (ewc_lambda)\n• base bitwise-UNCHANGED"]
        Store["OnDeviceSlotStore\npersist -> <digest>.pt\nmark_active -> active.json\nload re-verifies SHA-256"]
        Gate["RegressionGate.evaluate()\ncandidate vs baseline\nPROMOTE iff cand >= base - tolerance"]
        Scorer["score_policy()\nmean imagined rollout return\nseed-states + scoring_seed (deterministic)"]
        Proto["PolicyProtocol\n@runtime_checkable\nact(hidden, latent) -> action\n(WS5 seam: live net here)"]
    end

    subgraph WorldModel["src/mousedroid/world_model/ (REUSED)"]
        RSSM["RSSM / Dreamer\nimagine_step(action, h, z)\n-> (h', z', predicted_reward)"]
    end

    subgraph Factory["src/mousedroid/factory.py"]
        BuildCoord["build_on_device_coordinator(cfg, *, metrics=None)\nreturns None when absent/disabled"]
        BuildGate["_build_on_device_gate_runner(cfg, *, slot_store, metrics)"]
    end

    subgraph Config["src/mousedroid/config/schema.py"]
        Cfg["OnDeviceLearningConfig (Optional, default None)\n• enabled (default False)\n• trigger_min_new_records / check_interval_s\n• update_steps / learning_rate / ewc_lambda\n• regression_tolerance / held_out_fraction\n• rollout_horizon / n_scoring_rollouts / scoring_seed\n• slot_dir (relative, validator-gated)"]
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
    Scorer -- "policy.act()" --> Proto
    Scorer -- "imagine_step()" --> RSSM
    Gate -- "PROMOTE -> mark_active" --> Store
    Gate -- "REVERT -> inc(reason)" --> Counter

    %% Hot-loop isolation (NO edge crosses into HotLoop)
    Tick -. "shares event loop ONLY;\ntorch work offloaded off it" .- Loop

    classDef hot fill:#fee2e2,stroke:#dc2626,color:#000
    classDef slow fill:#dbeafe,stroke:#3b82f6,color:#000
    classDef internal fill:#e0f2fe,stroke:#0284c7,color:#000
    classDef config fill:#f3e8ff,stroke:#9333ea,color:#000
    classDef reused fill:#dcfce7,stroke:#16a34a,color:#000
    classDef store fill:#fef3c7,stroke:#f59e0b,color:#000

    class Tick hot
    class Loop slow
    class Coord,Learner,Store,Gate,Scorer,Proto,BuildCoord,BuildGate internal
    class Cfg,MetCfg,Counter config
    class RSSM reused
    class LMDB,Reader,SlotFS,Replay store
```

## One slow-cadence cycle — `ReplayTriggerCoordinator.maybe_update`

| Step | Action | Hot-loop isolation |
|---|---|---|
| 1. Trigger probe | `count_new_records()` via `asyncio.to_thread`. Below `trigger_min_new_records` ⇒ log `on_device_trigger_below_threshold`, return `None`. | Blocking LMDB scan off the event loop. |
| 2. Batch load | `load_batch()` via `asyncio.to_thread`. Empty batch ⇒ log `on_device_trigger_empty_batch`, return `None`. | Off the event loop. |
| 3. Bounded update | `learner.update(batch)` via `asyncio.to_thread`. Deep-copies the base, runs `update_steps`, applies the EWC penalty; base parameters bitwise-unchanged. | Gradient work on a worker thread. |
| 4. Persist | `slot_store.persist(candidate)` → write-temp-then-rename to `<digest>.pt`. | Cheap I/O on the slow cadence. |
| 5. Gate | `gate_runner(slot)` via `asyncio.to_thread` — score candidate vs baseline, PROMOTE or REVERT. `None` gate ⇒ byte-identical to pre-WS4. | Torch rollout scoring on a worker thread. |

## Promote / revert decision — `RegressionGate.evaluate`

| Condition | Outcome |
|---|---|
| `candidate_score >= baseline_score - regression_tolerance` | **PROMOTE**: `slot_store.mark_active(slot)` (write `active.json`); log `on_device_candidate_promoted`. Live policy NOT overwritten — only the pointer is recorded. |
| otherwise | **REVERT**: do NOT mark active; `metrics.inc_on_device_learning_reverted("regression_bound")`; log `on_device_candidate_reverted` (WARN). |
| slot SHA-256 verify fails on load | **REVERT** with reason `integrity_mismatch` (`SlotIntegrityError`). |
| update path raises | slow loop logs `on_device_update_cycle_failed` and keeps running; reason `exception` reserved for the counter. |

Determinism is load-bearing: both scores come from `score_policy` on the SAME
seed-states + `scoring_seed`, so the decision is reproducible.

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
- **Hot loop untouched.** No edge crosses into the `tick()` subgraph; the two
  paths share only the event loop, and all torch work is offloaded off it.

## WS5 pre-enablement seams (NOT yet wired)

These are documented in full in the operator runbook
(`docs/runbooks/jetson-on-device-learning.md`):

1. **Stand-in net, not the live net.** `build_on_device_coordinator` builds a
   config-sized `nn.Linear` stand-in candidate, and the gate-runner wraps the
   same stand-in for BOTH candidate and baseline adapters — so the end-to-end
   path runs and is tested, but the score delta is trivially zero. The
   `PolicyProtocol` seam is in place for WS5 to share the live policy net.
2. **Sampled seed-states.** The gate's seed-states are `manual_seed`-sampled
   latents, not yet encoded from a held-out replay slice through the world
   model.

Until both close, enabling is *safe* but does no *useful* learning — see the
runbook's "DO NOT enable on the rover yet" section and the ≥30-day soak-gate
framing.

## Related diagrams

- `docs/architecture/c4-overview.md` — Levels 1 (Context) and 2 (Container).
- `docs/architecture/c4-orchestrator.md` — the 30 Hz sense-plan-act loop that
  the slow-cadence task runs alongside.
- `docs/architecture/c4-rssm-sim-pretraining.md` — the RSSM world model reused
  by the rollout-return scorer.
- `docs/architecture/ADR-010-cloud-weight-update-ota.md` — the SHA-256
  integrity + separate-slot contract this feature reuses.
- `docs/architecture/c4-llm-gateway.md` — the deliberative LLM tier (separate
  concern; also OUTSIDE the hot loop).
