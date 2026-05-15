# Post-Phase-2 Sprint Plan — Deep Reasoning

**Authored:** 2026-04-27
**Scope:** Sequence the next 3 PRs after Phase 2 of the Physical AI roadmap
landed on `feat/phase2-real-episode-replay` (PR #60).

---

## 1. State of the world

| Track | Status |
|---|---|
| Phase 1 — Domain randomization | ✅ merged |
| Phase 2 — Real-episode replay (LMDB + mixer + golden RSSM) | ✅ on PR #60 |
| Phase 3a — VLA Protocol + `MockVLA` | ✅ landed |
| Phase 3b — `DistilledVLAOnnx` + HF pull | ✅ landed (CI matrix `[vla]` still TBD) |
| Phase 4 — VLM-derived dense rewards (Law-1 gated) | ✅ landed |
| CI: production-config validation gate | ✅ landed (commit `aeb0c15`) |
| Phase 2.1 — BC supervised loss into PPO | ✅ **landed (PR-A1)** — call site + dedicated `bc_optimizer` + sim/real mixer integration + perf budget regression; see [CHANGELOG.md](../../CHANGELOG.md) `## [Unreleased]`. |
| Phases 5–6 — Real-physics sim / on-device co-training | ⛔ blocked |

The four Physical-AI gaps are closed at the protocol layer. What remains is
the **last mile of integration**: real episodes still don't flow into the
*policy gradient*, the VLA path has no CI proof, and we have no telemetry to
know whether any of this is actually helping on hardware.

## 2. Candidate sprints — ranked

### Tier A — ship-now

**A1. Phase 2.1 — BC supervised loss into Constitutional-RL PPO** ⭐ recommended next

- *Value:* Highest. Phase 2 wired the **data path**; without 2.1 real episodes
  only tune the RSSM, not the policy. Largest ROI item left.
- *Risk if deferred:* `OfflineRLConfig.real_supervised_weight` and
  `OfflineRLTrainer.bc_update` are already public surface (committed in
  364bfba) — every week they sit behind a `weight==0` no-op is credibility
  debt and blocks Phase 6.
- *Why it was deferred:* The naive read of "BC into PPO" pointed at
  `training/train_constitutional_rl.py`, which is a numpy-MLP custom
  gradient path that *also* runs in RSSM-latent space (replay records hold
  observations, not latents — adding an obs→latent encoder bridge is a
  much larger PR).
- **Refined scope:** Insert the auxiliary BC term in the **torch
  `train_offline_rl.py` path** instead. Reasons:
  - `OfflineRLTrainer.bc_update` is already implemented (committed in
    364bfba) — the BC loss, optimizer step, and `weight==0` no-op guard all
    exist; only the call site is missing.
  - `OfflineRLDataset` already reads from the same LMDB store the Phase-2
    replay reader covers, in the same `(state, action, reward, next_state,
    done)` shape — no encoder bridge needed.
  - This is the textbook **TD3+BC** pattern (Fujimoto & Gu 2021): BC
    against the same batch as the actor's Q-loss is a well-known offline-RL
    stabilizer.
  - Surface stays small: one call site change in `train_offline_rl.py`,
    no schema changes (`real_supervised_weight` already exists), no new
    factory wiring.
- *Plan:*
  1. After each `trainer.update_step(...)` call in `train_offline_rl.py`,
     call `trainer.bc_update(states=batch["states"],
     actions=batch["actions"], weight=offline_cfg.real_supervised_weight)`.
  2. Aggregate `bc_loss` into the existing `epoch_losses` dict so the
     epoch-summary log line picks it up automatically.
  3. Emit a one-shot `offline_rl_bc_active` structured log when
     `real_supervised_weight > 0` so operators see the regularizer is on.
  4. Byte-identity at `weight=0`: `bc_update` early-returns
     `{"bc_loss": 0.0}` without stepping the optimizer (proven by the
     existing implementation; we add a regression test).
- *Acceptance:*
  - Unit byte-identity at `weight=0` (parameter-tensor equality).
  - Golden delta within ±1% at `weight=0.1` over a fixed-seed run.
  - Property test: gradients are finite for all bounded inputs.
  - Full suite green; coverage ≥85%.
- *Surface estimate:* ~3 source files, ~25 tests, no schema changes.

**A2. Phase 3b CI matrix entry — `[vla]` extras + ONNX smoke**

- *Value:* Medium-high. Without it, ORT API drift / provider-list / HF-pull
  regressions are invisible until manual extras run.
- *Plan:* New `vla-extras` CI job, py3.11 only,
  `pip install -e ".[dev,vla]"`,
  `pytest tests/unit/vla -m "not slow" --no-cov`.
  Advisory (`continue-on-error: true`) for first green week, then promote.

### Tier B — operational telemetry & safety

**B1. Replay/VLA/VLM runtime telemetry** (Prometheus counters + Grafana panel)
- New counters:
  `mousedroid_replay_records_total{outcome="ok|schema_mismatch"}`,
  `mousedroid_vla_inference_seconds`,
  `mousedroid_vla_timeout_total{mode}`,
  `mousedroid_vlm_progress_cache_hits_total`.
- Pure additions; existing `telemetry/metrics.py` registry already exists.

**B2. Validate-pillar in CI/nightly**
- The GH-Actions-runnable subset is the **10 pytest stages** (factory probes
  stay Jetson-only). Tag pillar tests with markers, add a nightly cron job.

### Tier C — voice / quality polish (low urgency)
- Streaming TTS for long utterances; wake-word detection;
  coverage-badge automation; mutation testing for `voice/` + `hardware/audio/`.

### Tier D — explicitly blocked
- **Phase 5** — gated on Phase 3b ≥30 days in production.
- **Phase 6** — depends on **Phase 2.1** closing the policy loop. ← this is
  why A1 is the right next move.

## 3. Recommended sprint sequence (next 3 PRs)

```
PR-A1  Phase 2.1: BC into PPO                                ✅ LANDED 2026-05-15
PR-A2  Phase 3b CI matrix + telemetry counters               🔄 IN REVIEW (PR #87)
PR-B2  Pillar nightly + Grafana panels + alert rules         📝 DRAFT (stacked on PR-A2)
```

**Why this order:**
1. **A1 first** because it converts already-shipped infrastructure into
   actual learning signal and unblocks Phase 6.
2. **A2 + B1 together** next because they touch overlapping surface
   (`vla/policy.py`, factory wiring, `ci.yml`) and a single PR amortizes
   review cost. Both are pure additions: zero behavior change at default
   config.
3. **B2 last** because it's value-multiplied by A1+A2+B1 — once new code
   paths exist *and* are instrumented, the nightly pillar run becomes a
   real regression net rather than a ritual.

## 4. Risks & mitigations (PR-A1)

| Risk | Mitigation |
|---|---|
| Numpy-MLP BC gradients introduce numerical drift | Byte-identity guard at `weight=0` + finite-gradient property test (Hypothesis) + golden curve at `weight=0.1`. |
| Mixer drains LMDB faster than PPO consumes | Reuse `MixerConfig.from_settings`; the existing async iterator is bounded by the consumer. |
| Schema-version mismatch silently disables BC | Existing `replay_schema_mismatch` structured log already covers this; surface count via `BCNumpyHead.last_skipped_records`. |
| Real-batch shape ≠ sim-batch shape | Validate at the mixer boundary; raise `ValueError` with structured log; default to no-op (skip step). |
| Hardcoded learning-rate / batch-size for the BC update | Add `OfflineRLConfig.bc_lr` and `OfflineRLConfig.bc_batch_size` with safe defaults; **no magic numbers** in the implementation. |

## 5. Definition of Done — PR-A1

- `OfflineRLTrainer.bc_update` is no longer a `weight==0` no-op; consumed by
  `train_constitutional_rl.py`'s PPO loop.
- New unit tests:
  - byte-identity at `weight=0`
  - bounded divergence at `weight=0.1`
  - finite-gradient Hypothesis property
  - empty-LMDB no-op
  - schema-mismatch counted-and-skipped
- New regression: golden policy-loss curve at fixed seed (mirrors the Phase
  2 RSSM golden pattern under `tests/regression/_*_golden_helper.py`).
- `ruff` + `ruff format` + `mypy --strict` + hardcoded-values gate clean.
- Full suite green (3298+ tests).
- `CHANGELOG.md` + `NEXT_STEPS.md` updated; "Phase 2.1" entry struck through.

---

*This document is the agreed plan-of-record. Update inline as PRs land.*
