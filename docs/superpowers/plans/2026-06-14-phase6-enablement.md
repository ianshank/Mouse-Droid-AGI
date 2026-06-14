# Phase-6 On-Device Learning ENABLEMENT — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task (fresh implementer → spec review → quality review per WS). Steps use checkbox (`- [ ]`) syntax. Execute in the worktree `.claude/worktrees/phase6-enablement` (branch `claude/phase6-enablement-2026-06-14` off trunk `336d3d5`). Prepend `PYTHONPATH="$(pwd)/src"` to every pytest/python (editable-install footgun). Commit `git -c commit.gpgsign=false commit --no-verify`.

**Goal:** Make the merged-but-inert Phase-6 on-device learning subsystem (#134) *functional* by refining the rover's ACTUAL learned component — the **RSSM world model** — and (optionally, gated) hot-swapping a sim-validated, regression-gated candidate into the live model. Still **default-OFF**; the ≥30-day soak gates ENABLING on the rover, not building/sim-validating.

**Architecture (corrected by recon — see "Premise correction"):** The mouse-droid has **no learned policy net** — action selection is MCTS planning over the RSSM (`navigation.py` → `MCTSPlanner.plan` → `RSSM.imagine_step`). So on-device learning must refine the **RSSM** (prior + reward head, the dynamics MCTS plans through), score a **candidate-RSSM vs baseline-RSSM** via deterministic imagined rollouts, and hot-swap the RSSM `state_dict` through the existing C1 atomic-swap seam. Every change is additive, `Optional`/default-OFF, hot-loop-untouched, sequence-batch-driven.

**Tech Stack:** Python 3.10–3.12, Pydantic v2, PyTorch, pytest+pytest-asyncio+hypothesis, ruff 0.8.0, mypy 2.1.0 (strict), structlog. Reuses: `world_model/rssm.py` (`RSSM.train_sequence`, `observe_step`, `imagine_step`), `training/replay/lmdb_reader.py`, `learning/on_device/*` (#134), the C1 `weight_update_loader`/`_apply_pending_weight_update` atomic swap.

---

## Premise correction (why this plan differs from "wire the live policy net")

Recon (`2026-06-14` integration map) established, with evidence:
- **No `(h,z)→action` policy `nn.Module`.** `orchestrator._select_action` (`orchestrator.py:1142`) → `MouseDroidNavigationAgent.act` (`agents/navigation.py:48`) → `MCTSPlanner.plan` (`world_model/mcts.py:157`), which scores `linspace` candidate actions via `imagine_step` rollouts. The refinement target is the **RSSM** (`rssm.py:24`), built by `build_world_model` (`factory.py:580`).
- **`RSSM` has no `forward()`.** `EWCOnlineLearner.update` (`ewc_online.py:132`) does `candidate(batch)` — invalid on RSSM. RSSM's grad path is `train_sequence(batch: dict, decoders: RawModalityDecoders)` (`rssm.py:223`) over `(B,T,…)` sequence dicts.
- **The gate builds a FRESH `RSSM(cfg.model)`** (`factory.py:3127`), divorced from the live `self._world_model` (which may be `DualStreamRSSM`). It scores random `nn.Linear` adapters (`factory.py:3154-3162`) — meaningless until the candidate is a real refined RSSM.
- **Replay records lack `valid_mask`/audio/lidar** (`experience/record.py:32`) needed by `observe_step` (`rssm.py:107`).
- **Hot-swap loader is an unfinished C1 seam** — `build_weight_update_loader` returns `None` (`factory.py:1594`); `_apply_pending_weight_update` (`orchestrator.py:905`) already does atomic swap + `reset_state_on_swap` (`:1024-1026`).

**Consequence:** the stand-in `nn.Linear` learner/gate from #134 is replaced by an RSSM-native path. The #134 default-OFF contracts, slot store, counter, and coordinator seam are REUSED unchanged.

---

## Global acceptance criteria (every WS / PR)

- [ ] `ruff check src/ tests/ tools/` + `ruff format --check src/ tests/` clean.
- [ ] `mypy src/ --strict --ignore-missing-imports` clean (~4 min).
- [ ] **No hardcoded values** — every dim/tunable from `ModelConfig`/`OnDeviceLearningConfig`; live model arch from `build_world_model(cfg)`, never a literal `RSSM(cfg.model)`.
- [ ] **Backwards-compatible / default-OFF byte-identical** — new config fields `Optional`/defaulted; `cfg.on_device_learning` absent/disabled ⇒ orchestrator + `/metrics` byte-identical (regression test). New behaviour (replay-encoded seed states, hot-swap) each behind its own default-OFF flag.
- [ ] **Hot 30 Hz loop provably untouched** — all torch work (refine, score, encode, persist) via `asyncio.to_thread` at the slow cadence; `tick_count==0` integration assertions; `tests/performance/` budget unaffected.
- [ ] **≥85% coverage per NEW module** (`--cov-report=term-missing`, torch pre-imported), not just the repo aggregate.
- [ ] `tests/regression/test_suppression_budget.py` (`_MAX_TYPE_IGNORE=8`, `_MAX_NOQA=19`) + `test_numpy_hygiene.py` pass — **zero new src suppressions** (ratchet-down preferred).
- [ ] structlog on new operational paths (reuse `on_device_*` event family); Google docstrings; `from __future__ import annotations`.

---

## WS-E0 — Foundation: thread the LIVE world model + config gates (low risk, do first)

**Files:** `src/mousedroid/config/schema.py` (`OnDeviceLearningConfig` ~:708); `src/mousedroid/factory.py` (`build_on_device_coordinator` :3008, `_build_on_device_gate_runner` :3087, `build_orchestrator` :3274/:3532); tests under `tests/unit/config/`, `tests/unit/factory/`, `tests/regression/`.

- [ ] Add config fields (all defaulted, additive): `enable_hot_swap: bool = False`; `seed_state_source: Literal["sampled","replay_encoded"] = "sampled"`; `refine_sequence_length: int = Field(16, gt=0)`; `refine_batch_episodes: int = Field(4, gt=0)`. Regression test: pre-enablement YAML loads unchanged; defaults preserve #134 behaviour.
- [ ] Thread the live world model into the coordinator: `build_on_device_coordinator(cfg, *, metrics=None, world_model=None)` (keyword-only, default `None` ⇒ #134 behaviour). `build_orchestrator` passes the already-built `wm` (`factory.py:3274`). The gate runner uses THIS `world_model` (or `build_world_model(cfg)` if `None`), **never** a fresh literal `RSSM(cfg.model)` — fixes the wrong-arch (DualStreamRSSM) bug.
- [ ] Tests: factory builds coordinator with the live WM injected; `world_model=None` path stays #134-identical; AQA for the new fields.

**Acceptance:** default-OFF byte-identical; gate now references the real model arch; no behaviour change until later WS flip the flags.

---

## WS-E1 — Replay-encoded seed states (medium risk; default-OFF via `seed_state_source`)

**Files:** `src/mousedroid/learning/on_device/seed_states.py` (new); `factory.py:_build_on_device_gate_runner` :3138-3146; `tests/unit/learning/on_device/test_seed_states.py`.

- [ ] New `record_to_observation(record: MouseDroidExperienceRecord) -> ObservationProtocol` adapter: expose `vision_features/distance_m/motor_state` from the record; **synthesize `valid_mask`** (1 for stored modalities, 0 for absent vision when `vision_features` empty); return `None`/empty for audio/lidar (encoder gates these: `rssm.py:130,137,156,167`). Pin the mask convention with a test.
- [ ] `encode_seed_states(world_model, records, n_seed, *, device) -> list[SeedState]`: roll `observe_step` from zero `(h,z)` across a held-out slice (`prev_action` = record[t-1].action), collect `(h,z)`. `torch.no_grad`, eval, device-correct.
- [ ] Wire behind `seed_state_source`: `"sampled"` (default) keeps the #134 `manual_seed` path byte-identical; `"replay_encoded"` uses the new encoder. Regression test proves the default path is unchanged.
- [ ] Tests: deterministic given fixed records; mask synthesis; device placement; empty-slice fallback to sampled with a structlog warning.

**Acceptance:** opt-in replay-grounded seed states; default path byte-identical; ≥85% module coverage.

---

## WS-E2 — RSSM-refinement learner (HIGH risk — the core refactor)

**Files:** `src/mousedroid/learning/on_device/rssm_refiner.py` (new, implements the `OnDeviceLearner` protocol or a sibling); `factory.py:build_on_device_coordinator` :3061-3063 (replace the stand-in); `tests/unit/learning/on_device/test_rssm_refiner.py`, `tests/property/test_rssm_refiner_base_untouched.py`.

- [ ] Define `RSSMRefiner`: `deepcopy(live_rssm)` → candidate; refine via `RSSM.train_sequence(batch, decoders)` (`rssm.py:223`) for `update_steps` at `learning_rate`, with the EWC penalty REUSING `learning/ewc.py` Fisher anchored on the base RSSM params (thread the on-device `ewc_lambda`; skip when `0`). Returns `OnDeviceUpdateResult(candidate_state_dict=candidate.state_dict(), …)`. **Base RSSM bitwise-unchanged** (property test, mirrors #134 `test_on_device_no_inplace_corruption`).
- [ ] Build the sequence batch from replay: `_load_replay_sequence_batch(reader, refine_batch_episodes, refine_sequence_length)` — assemble `(B,T,…)` modality/action/valid_mask dicts `train_sequence` expects (recon: confirm exact dict keys + `RawModalityDecoders` construction from `build_world_model`/decoders). Reuse the `_run_coro_blocking` async→sync bridge (`factory.py:3178`). **Spec the exact batch dict in the implementer prompt from a fresh `train_sequence` read — do NOT assume.**
- [ ] `torch.no_grad` only for eval; grads via `train_sequence`'s own path; offloaded via `asyncio.to_thread` in the coordinator (already so for `learner.update`).
- [ ] Tests: candidate diverges from base after refinement; base unchanged (property/hypothesis over seeds); EWC penalty bites (λ>0 vs 0); deterministic given fixed batch+seed; structlog.

**Highest risk:** `train_sequence` batch/decoder contract + RSSM `state_dict` round-trip into a live model of identical dims. **Mitigation:** the implementer must read `rssm.py:223` + the decoders construction + an existing `train_sequence` caller (e.g. `training/rssm_pretrainer.py`) and mirror the real batch shape; add a round-trip test (`refined.state_dict()` loads into a fresh `build_world_model(cfg)` without error).

---

## WS-E3 — RSSM-vs-RSSM regression gate (HIGH risk — scoring semantics)

**Files:** `src/mousedroid/learning/on_device/scoring.py` (extend) + `regression_gate.py` (extend); `factory.py:_build_on_device_gate_runner`; `tests/unit/learning/on_device/test_rssm_gate.py`, `tests/property/test_on_device_auto_revert.py` (extend).

- [ ] Score **candidate-RSSM vs baseline-RSSM** (baseline = the live model's current weights; candidate = the refined slot loaded into a copy) on the SAME fixed seed-states under a **deterministic fixed action policy** (NOT MCTS — MCTS-in-rollout is too heavy + nondeterministic; use a deterministic action schedule / the existing `score_policy` with a fixed-action `PolicyProtocol`). Metric = mean imagined return (reuse `score_policy`, `scoring.py:132`), with RNG/eval/device restore from the #134 hardening.
- [ ] PROMOTE iff `candidate_score >= baseline_score - regression_tolerance` → `slot_store.mark_active`; else REVERT + `inc_on_device_learning_reverted("regression_bound")`. Keep the deterministic auto-revert property test ("active policy never persists below the bound").
- [ ] The candidate adapter loads the refined RSSM state_dict into a copy of the live model (not a stand-in linear). Baseline adapter wraps the live model's current weights. Both are real RSSMs.
- [ ] Tests: promote/revert at the tolerance boundary; counter on revert; deterministic given fixed seed-states; baseline untouched on revert; synthetic-degradation always reverts.

**Highest risk:** defining a deterministic, meaningful action policy for scoring two dynamics models. **Mitigation:** fixed deterministic action schedule (e.g. zero/sweep actions seeded by `scoring_seed`); document that the gate measures *dynamics-prediction return under a fixed probe policy*, a proxy for MCTS-planning quality, and is a conservative guard (revert-biased).

---

## WS-E4 — Hot-swap via the C1 seam (medium-high risk; default-OFF via `enable_hot_swap`)

**Files:** `factory.py:build_weight_update_loader` :1570 (currently `None`); a slot→`PendingWeightUpdate` source; `orchestrator.py:_apply_pending_weight_update` :905 (reuse, do not reinvent); `tests/unit/factory/`, `tests/integration/`, `tests/regression/`.

- [ ] Implement `build_weight_update_loader` to materialize an on-device slot: `RSSM-or-DualStream = build_world_model(cfg); model.load_state_dict(slot_store.load(slot)); return model` (reuse `load_rssm_with_migration` `factory.py:695` for dim-drift safety). 
- [ ] Surface `slot_store.load_active()` as a `PendingWeightUpdate(engine_type=ENGINE_TYPE_WORLD_MODEL)` via a poller-like source, so `_apply_pending_weight_update` swaps it ATOMICALLY with the existing `reset_state_on_swap` (`orchestrator.py:1024-1026`). Gated by `enable_hot_swap` (default `False`) — promotion (`mark_active`) stays separate from activation.
- [ ] Tests: with `enable_hot_swap=False` no swap ever occurs (byte-identical); with `True` + an active slot, the live `self._world_model` is replaced atomically off the hot loop and `(h,z)` reset; integrity-mismatch on a corrupt slot is fail-closed (no swap, counter `integrity_mismatch`).

**Highest risk:** swapping the hot-loop-shared world model live. **Mitigation:** reuse the C1 single-coroutine atomic-swap + reset machinery wholesale; never touch `self._world_model` outside `_apply_pending_weight_update`.

---

## WS-E5 — Factory wiring + docs + sim soak validation (closes enablement)

**Files:** `factory.py` (final wiring); `docs/runbooks/jetson-on-device-learning.md` (update the "NOT yet enabled" seams — now closed/flagged); `docs/architecture/c4-on-device-learning.md` (update); `CHANGELOG.md`; `CLAUDE.md` (update the Phase-6 contracts section).

- [ ] Finalize `build_on_device_coordinator` to assemble RSSMRefiner + RSSM-vs-RSSM gate + (gated) hot-swap, all from config + the live WM. Default-OFF byte-identical preserved.
- [ ] **Sim soak validation**: a `tests/`-gated (or `scripts/`) deterministic sim run feeding a known-degrading vs known-improving replay stream, asserting promote-on-improve / revert-on-degrade + counter; document the soak discipline (≥30-day shadow before rover enable).
- [ ] Update runbook/C4/CLAUDE.md: which seams are now closed (live-net = RSSM, replay-encoded seed states, hot-swap) and which gates remain (≥30-day soak before `enable_hot_swap=true` on the rover). CHANGELOG `## [Unreleased]`.

**Acceptance:** the feature is functional + sim-validated end-to-end behind default-OFF flags; docs reflect the corrected architecture; full suite green.

---

## Execution methodology

- **Worktree isolation**, subagent-driven TDD (`superpowers:subagent-driven-development`): fresh implementer per WS → spec review → quality review. WS-E0→E5 are SEQUENTIAL (E0 unblocks all; E2/E3 are the high-risk core). Shared-file serialization points: `config/schema.py`, `factory.py`, `learning/on_device/scoring.py`, `regression_gate.py`.
- **Recon-grounded:** every implementer re-reads the real `train_sequence`/`observe_step`/`_apply_pending_weight_update` before coding — follow the REAL code, not this doc's summary, and note divergences.
- **Per-WS:** push → CI green (3.10/3.11/3.12 matrix) → reply/resolve bot threads → next WS. Checkpoint with the user before WS-E2 (the core refactor) and before flipping any default to ON.
- **MCPs:** Context7 for current PyTorch `autograd`/`load_state_dict` semantics if needed.

## Verification (end-to-end)

```bash
ruff check src/ tests/ tools/ && ruff format --check src/ tests/
mypy src/ --strict --ignore-missing-imports
PYTHONPATH="$(pwd)/src" pytest tests/unit/learning/on_device tests/property tests/integration -k "on_device or rssm_refin or pr134" --import-mode=importlib --no-cov
PYTHONPATH="$(pwd)/src" pytest tests/regression/test_suppression_budget.py tests/regression/test_numpy_hygiene.py --import-mode=importlib --no-cov
```
- Default-OFF byte-identical (regression); base RSSM never mutated in-place (property); deterministic auto-revert (property); hot loop `tick_count==0` (integration); per-module ≥85% coverage.

## Self-review checklist

- [ ] Premise correction front-and-center (no policy net → refine RSSM); every WS cites real `file:line`.
- [ ] No hardcoded values (live WM via `build_world_model`, dims from config); backwards-compat (Optional/default-OFF per flag); reusable (REUSE train_sequence, EWC, slot store, C1 swap — no reinvention).
- [ ] High-risk WS (E2 train_sequence batch/decoders; E3 scoring policy) flagged with mitigations + "read real code first".
- [ ] Soak gate distinguishes BUILD/sim-validate (now) from rover-ENABLE (≥30-day, operator).

## Peer-review revisions (applied 2026-06-14)

Two independent adversarial reviewers (architecture lens + safety/testability lens) verified the plan against real code. Both: **APPROVE-WITH-CHANGES**. The premise + every cited `file:line` checked out; the following BLOCKER/MAJOR fixes are now binding amendments to the WS specs above. **WS-E2/E3 require a SPIKE before implementation.**

### WS-E-SPIKE (NEW — do BEFORE committing WS-E2/E3; throwaway, not merged)
Prove the two core uncertainties with a quick prototype branch, results captured in this doc:
- [ ] **Refine + round-trip:** `deepcopy(build_world_model(cfg))` (plain `RSSM`, small dims) → build `RawModalityDecoders(candidate.cfg)` → one `train_sequence` step over a tiny synthetic `(B,T,…)` dict batch via a self-owned optimizer → `candidate.state_dict()` strict-loads into a fresh `build_world_model(cfg)`. Confirms the refiner contract holds.
- [ ] **Gate signal separates degraded from baseline:** confirm a *deliberately-degraded* RSSM scores WORSE than baseline under the chosen gate metric (held-out reconstruction/KL loss — see WS-E3 revision), and that the imagined-reward-head metric does NOT reliably do so (the self-gaming failure mode). Pick the gate metric from evidence, not assumption.

### WS-E0 (config) — amendments
- [BLOCKER→guard] The live WM may be `DualStreamRSSM` (when `cfg.model.cfc_hidden_dim>0`, e.g. `config/jetson_dual_stream.yaml`) or `DualStreamRSSMOnnx` — **neither has `train_sequence`**. Add a capability gate: refinement is enabled ONLY when `hasattr(world_model, "train_sequence")`; otherwise `build_on_device_coordinator` returns `None` (or a no-refine coordinator) with a structured `on_device_refiner_unsupported_engine` warning. Unit test: a DualStream live WM → refiner disabled, NOT a crash. (Current `jetson_production.yaml` has no `model:` block ⇒ `cfc_hidden_dim=0` ⇒ plain `RSSM`, so production is supported — but the guard is mandatory.)
- [MINOR] Add a `model_validator` rejecting `enable_hot_swap=True` unless `enabled=True` (mirror `_validate_slot_dir` / `_require_endpoints_when_enabled`), with a loud warning. `seed_state_source` stays a validated `Literal`.

### WS-E1 (seed states) — amendments
- [MAJOR] The `record_to_observation` adapter must satisfy the FULL `ObservationProtocol` (`sensing/protocol.py`): `timestamp`, `vision_features`, `distance_m`, `motor_state`, `audio_chunk` (empty array), `lidar_features` (`None`), `valid_mask`, **and `n_modalities`** — not just 3 fields + mask.
- [MAJOR] Derive the `valid_mask` length/order from the **live `world_model.encoder` enabled flags** (the `[vision, ultrasonic, motor, audio, (lidar)]` contract), NOT a literal length. Test: `len(mask) == this cfg's modality count`; mask is 0 for vision when `vision_features` is empty. (Encoder gates safely: `encoder.py:98,119`.)

### WS-E2 (refiner) — SPLIT + amendments
**Split into E2a (plumbing) + E2b (refiner) — the type changes ripple into `replay_trigger.py` shared with #134's green tests.**
- [BLOCKER] **Sibling protocol, not `OnDeviceLearner`.** `OnDeviceLearner.update(batch: Tensor)` + the coordinator's `load_batch: Callable[[], Tensor]` + `batch.shape[0]==0` guard (`replay_trigger.py:74,116`) are Tensor-typed. `train_sequence` needs `dict[str,Tensor]`. **E2a:** add an `RSSMSequenceLearner` sibling protocol returning `OnDeviceUpdateResult`, and generalize the coordinator (or add a parallel path) to a dict batch + a dict-aware empty-check — without breaking #134's Tensor path/tests.
- [BLOCKER] **Do NOT reuse `EWCAgent` for the RSSM.** `EWCAgent.consolidate` does `model(x)` (forward) — RSSM has none. **v1: drop EWC (λ=0 path), documented;** OR (follow-up) an RSSM-native Fisher = squared grads of the `train_sequence` loss w.r.t. RSSM params over a held-out batch. Pick λ=0 for v1 unless the spike shows native-Fisher is cheap.
- [MAJOR] **`train_sequence` owns no optimizer/lr.** The refiner builds its own optimizer over `candidate.parameters() + decoders.parameters()`. **Avoid a new `.backward()`** (adds a 9th `type: ignore`; budget is 8/8 — `rssm_pretrainer.py:104` already holds one). Use the `torch.autograd.grad` manual-SGD idiom (`ewc_online.py:137-140`, no suppression) OR reuse `RSSMPretrainer` (which already carries the suppression). 
- [MAJOR] **Decoders:** `RawModalityDecoders(candidate.cfg)` is built ONLY by the pretrainer (`rssm_pretrainer.py:53`), not the factory — the refiner constructs them itself. They are throwaway reconstruction heads — **persist ONLY `candidate.state_dict()` (the RSSM) to the slot**, never the decoders.
- [MAJOR] **Batch dict shape** = `{motor, action, valid_mask, + enabled modalities}` `(B,T,…)` — mirror the REAL caller `RSSMPretrainer._to_device` (`rssm_pretrainer.py:65-73`), not this doc. Build from replay via a new `_load_replay_sequence_batch` (reuse `_run_coro_blocking`).
- [MINOR] Round-trip test: same-cfg slot is a STRICT load (`load_state_dict`); `load_rssm_with_migration` (dim-drift) is unnecessary here.

### WS-E3 (gate) — REWORKED signal + amendments
- [MAJOR — changes the metric] **Gate on held-out reconstruction/KL loss, NOT imagined reward-head return.** `score_policy` sums the *candidate's own* `reward_head` along a prior rollout → a candidate that inflates its reward head scores HIGHER while being WORSE (self-gaming; the gate would promote the regression). Instead: score **candidate-RSSM vs baseline-RSSM** by `train_sequence`/`observe_step` **reconstruction+KL loss on a FIXED held-out replay batch** (lower-is-better, deterministic, measures real dynamics quality on real data). PROMOTE iff `candidate_loss <= baseline_loss + regression_tolerance`. If a return-based signal is kept as secondary, it MUST use a **fixed reference reward** (baseline head or replay ground-truth reward), never the candidate's own head.
- [MAJOR] **WM-varying scorer.** `score_policy`/`RegressionGate._build_default_score_fn` are hard-wired to ONE `world_model` + a varying policy (`regression_gate.py:120-144`). RSSM-vs-RSSM needs a new `score_dynamics(world_model, fixed_held_out_batch)` that runs each model through ITS OWN forward — a real gate refactor, not an "extend". Keep the deterministic auto-revert property test.

### WS-E4 (hot-swap) — SAFETY reworked
- [BLOCKER] **Materialize the candidate engine OFF the hot loop.** The C1 `weight_update_loader` is called SYNCHRONOUSLY inside `tick()` (`orchestrator.py:693→1006`) — `torch.load` + model construction there would blow `tick_timeout_s` → emergency_stop. Do the disk-load + construct + device-place in the slow `_on_device_update_loop`/coordinator (`asyncio.to_thread`); the poller surfaces an **already-constructed, device-correct engine**, so the hot-loop swap is a **pure reference assignment** only.
- [MAJOR] **Device parity:** materialize the swap engine on the SAME device as the live `self._world_model` (NOT `cuda-if-available` like `load_rssm_with_migration` `factory.py:712`); else `reset_state_on_swap`'s `zeros_like(self._h)` (`orchestrator.py:1048`) → cross-device op. Integration test: device parity post-swap.
- [MAJOR] **Integrity counter:** the C1 broad-except swallows `SlotIntegrityError` as a generic `cloud_weight_update_swap_failed` (`orchestrator.py:1007-1019`) — it does NOT increment `inc_on_device_learning_reverted("integrity_mismatch")`. Verify the slot off-loop and raise→count BEFORE creating the `PendingWeightUpdate`. Fail-closed (live model untouched) is preserved.
- [MAJOR] **Full poller contract:** the source must satisfy `WeightUpdatePollerProtocol` (`start/stop/pending_update/acknowledge_swap`, `cloud/protocol.py:91-106`), be registered in `weight_update_pollers`, and emit a `PendingWeightUpdate` with ALL 7 fields (`repo_id, filename, revision, sha256, local_path, downloaded_at, engine_type=ENGINE_TYPE_WORLD_MODEL`). The loader must work INDEPENDENT of `cfg.cloud.weight_update.poll_interval_s` (which short-circuits `build_weight_update_loader` to `None`, `factory.py:1589`).
- [MINOR] `load_active()` returns a **digest string** (`slot_store.py:195`), not a `CandidateSlot`. Add a slot-store helper to reconstruct `CandidateSlot(path=slot_dir / f"{digest}.pt", digest=digest)` and load via it.

### Required NEW tests (binding)
1. Regression: default-OFF byte-identical with the new `world_model=` kwarg + 4 new fields (extend `test_on_device_ws4_backwards_compat.py`); pre-enablement YAML loads unchanged.
2. Unit: DualStream/non-`train_sequence` WM → refiner disabled (not crashed).
3. Property: base RSSM `state_dict` bitwise-unchanged after refine (mirror `test_on_device_no_inplace_corruption`).
4. Property: deterministic auto-revert on a synthetically-degraded RSSM (held-out-loss gate).
5. Integration: hot-swap materialization happens OFF-loop (spy the loader → asserted NOT called from the tick); a per-tick latency assertion (NOT `tick_count==0`, since the swap runs in `tick()`).
6. Integration: device parity post-swap.
7. Unit: corrupt slot → `inc_on_device_learning_reverted("integrity_mismatch")` + `self._world_model` identity unchanged.
8. Unit: synthesized `valid_mask` length == live-encoder modality count.
9. `test_suppression_budget.py` (8/19, zero new) + `test_numpy_hygiene.py` green; per-file branch coverage ≥85% (`scripts/check_branch_coverage.py`) — exercise the refiner's error branches (empty batch, missing decoders, non-RSSM WM).

### Reviewer-confirmed-correct (no change)
Premise (no policy net → refine RSSM); default-OFF discipline; promotion (`mark_active`) separate from activation (`enable_hot_swap`); soak-gate separation (BUILD/sim now vs ≥30-day rover-enable); `_run_coro_blocking` + `asyncio.to_thread` offload pattern. Determinism is testable on this box (existing on-device property tests use a real small `RSSM`).

### Net effect on sequencing
WS-E0 → WS-E1 → **WS-E-SPIKE** → WS-E2a (plumbing) → WS-E2b (refiner, λ=0 v1) → WS-E3 (recon-loss gate) → WS-E4 (off-loop swap) → WS-E5 (docs + sim soak). Checkpoint with the user after the SPIKE (go/no-go on the core refactor) and before flipping any default ON.

## SPIKE RESULTS — VERDICT: GO_WITH_CHANGES (verified 2026-06-14, multi-agent workflow + adversarial re-run)

Both core uncertainties resolved with hard numbers (tiny RSSM hidden=8/latent=4/action=3). **Implementers MUST use these LOCKED contracts.**

### WS-E2 refiner — LOCKED (proven works)
1. **Capability guard:** `from mousedroid.world_model.rssm import RSSM` — refine ONLY when `hasattr(model, "train_sequence")`. Verified `RSSM.train_sequence` exists (rssm.py:223); `DualStreamRSSM` has NONE. `build_world_model` → RSSM iff `engine=="torch"` AND `cfg.model.cfc_hidden_dim==0`.
2. **Isolation:** `candidate = copy.deepcopy(base_rssm)` — base bitwise-unchanged (torch.equal over named_parameters, confirmed).
3. **Decoders (off the RSSM):** `from mousedroid.world_model.rssm import RawModalityDecoders; decoders = RawModalityDecoders(candidate.cfg)`. NEVER persisted into the slot (RSSM `state_dict` has no decode_* keys → deployment checkpoint byte-identical).
4. **Batch dict (B,T,…)** = exactly `RSSMPretrainer._to_device` (rssm_pretrainer.py:65-73): `motor`(always), `action`(always), `valid_mask`(len 4 OR 5, both work), `ultrasonic`/`lidar`/`vision` (each read iff `encoder.<m>_enabled`; vision (B,T,0) when off). `train_sequence` zeros h,z internally. Returns `{loss,recon,kl,posterior_std}`; `loss.requires_grad=True`.
5. **Optimizer = `autograd.grad` manual-SGD, NO `.backward()`** (stays in the 8 `type: ignore` budget — `.backward()` would add a 9th):
   ```python
   out = candidate.train_sequence(batch, decoders)
   params = list(candidate.parameters()) + list(decoders.parameters())
   grads = torch.autograd.grad(out["loss"], params, allow_unused=True)   # allow_unused=True MANDATORY — reward_head + 3 others are unused in the recon/KL graph; omitting it RAISES RuntimeError
   with torch.no_grad():
       for p, g in zip(params, grads):
           if g is not None:  # None-grad guard MANDATORY
               p -= lr * g
   ```
   ⚠️ **`EWCOnlineLearner` is NOT a drop-in:** its `update(batch: Tensor)→candidate(batch)` assumes `forward(tensor)`, which RSSM lacks; its line-137 `autograd.grad` OMITS `allow_unused` (→ raises on RSSM). Write an RSSM-aware sibling learner.
6. **Round-trip:** `fresh = build_world_model(same cfg); fresh.load_state_dict(candidate.state_dict(), strict=True)` → missing=[] unexpected=[] (confirmed).
7. **λ=0:** `train_sequence` loss == `recon + cfg.model.kl_beta*kl` exactly (no penalty term). EWC is purely additive in the caller; v1 ships λ=0 (no EWC); RSSM-native diagonal Fisher (mean per-sample grad² via autograd.grad) is feasible as a follow-up.

### WS-E3 gate — LOCKED (recon-loss, retire score_policy)
- **Metric:** held-out **reconstruction+KL loss** = `train_sequence(batch, decoders)["loss"]` on a FIXED held-out (B,T,…) batch under `model.eval()` + `torch.no_grad()`. **LOWER IS BETTER.**
- **Direction (INVERTS the current gate):** PROMOTE iff `candidate_loss <= baseline_loss + cfg.regression_tolerance`; else REVERT + `inc_on_device_learning_reverted("regression_bound")`. `regression_gate.py:170` is currently higher-is-better (`>= baseline - tol`) and `GateDecision.delta` sign assumes higher=better → **both must flip** (positive delta = worse for loss).
- **Determinism:** `torch.manual_seed(scoring_seed)` immediately before EACH `train_sequence` call (reparam noise draws from global RNG); same seed for baseline + candidate.
- **Shared decoders:** the SAME `RawModalityDecoders(cfg)` instance scores baseline AND candidate (recon heads external to RSSM).
- **Proof score_policy self-games (RETIRE from gate):** reward-head-inflated degraded model → imagined return +57.8 (looks better) but recon loss byte-identical to baseline; genuinely-degraded model → recon loss 3.35→95.7 (separates). `score_policy` sums the model's OWN `reward_head` → unsafe as a gate signal; keep only as a non-gating diagnostic.
- **deg edge:** a heavily-corrupted candidate can blow KL to very-large/non-finite → still REVERTs (large > baseline); tests must assert large/non-finite ⇒ revert, not finite-loss.

### Gate-seam rework required (WS-E3)
`RegressionGate` (regression_gate.py) carries `world_model + seed_states + score_fn(policy→float)` — NO batch, NO decoders, higher-is-better. WS-E3 adds a held-out batch + shared decoders, a `score_dynamics(world_model, batch, decoders, *, seed)` scorer (scores the WORLD MODEL, not a policy), flips the comparison + delta sign. The injectable `score_fn` seam helps but the policy→float signature needs rework.

### Test caveats (locked)
- `Settings()` bare raises (distance-sensor validator) → tests use `Settings(mock_hardware=True)`; the real rover path is `mock_hardware=False` — confirm batch construction doesn't depend on the mock shortcut.
- Held-out batch must be REPLAY-ENCODED (WS-E1), not `manual_seed`-sampled — `score_loss` correctness depends on a representative FIXED held-out batch.
