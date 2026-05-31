# MouseDroidAGI — Next Steps

Rebased on 2026-04-27 for `feat/smoke-post-pr55` after the Ten Pillars validation campaign
(20/20 PASS on Jetson Orin Nano, 2026-04-26T23:55:42Z).

---

## Current Baseline

- **Phase 1 training baseline is merged**: domain-randomized Phase 0 generation, top-level
  `domain_randomization` schema/YAML support, seeded Phase 0 wiring, and the associated unit,
  integration, and regression tests are now part of the active line.
- **Jetson deployment hardening is already landed**: `scripts/mousedroid-docker.service` runs
  `scripts/sync_jetson_overlay.sh` before preflight, so overlay sync is part of the service
  contract rather than a manual post-`git pull` step.
- **Voice rollout is already landed**: `voice.personality_to_model_map`,
  `voice.event_intensity_thresholds`, `voice.output_volume`, Piper gain application, the new TTS
  integration tests, speaker+TTS integration tests, and smoke-harness unit coverage are in-tree.
- **Ten Pillars validation campaign is complete**: `scripts/validate_pillar.sh all` runs
  20 checks (10 pytest stages + 10 factory probes) across all pillars — last result
  **Overall: PASS (20/20)** on Jetson Orin Nano (`2026-04-26T23:55:42Z`).
- **Active production scope**: camera + LiDAR + USB audio + ESP32 on Jetson. The HC-SR04
  ultrasonic path is parked, and the robot-arm platform is deferred from the active delivery plan.

---

## P0 — Physical AI Roadmap (Phases 2 → 6)

Phase 1 (per-episode domain randomization for RSSM pretraining) ships in this
branch. Subsequent phases close the remaining three Physical AI gaps and each
lands in an isolated PR off the default branch. Dependency direction is
strictly **Phase 1 → 2 → 3 → 4**; Phases 5 and 6 are deferred until Phase 3b
has been in production for ≥30 days.

### Phase 2 — Real-Episode Replay Loop (sim-to-real feedback) ✅ IN-FLIGHT (`feat/phase2-real-episode-replay`)

Wire the existing LMDB experience logger back into the offline training
pipeline so successes and failures from real-world rollouts continuously refine
the RSSM and the Constitutional-RL policy. Closes the second of four gaps.

**Scope:**
- `src/mousedroid/training/replay/lmdb_reader.py` — async streaming iterator
  over an LMDB env (chunks of 64 records via `asyncio.to_thread`; never load
  the whole DB into RAM on the 8 GB Orin) ✅
- `src/mousedroid/training/replay/mixer.py` — ratio-controlled sampler over
  `(sim_iter, real_iter)` with a deterministic `numpy.random.Generator`; ramped
  `alpha` from 0.0 → target over a configurable number of steps (RL-Co two-stage) ✅
- `training/replay_real_episodes.py` — CLI with `--dry-run`, `--use-real-replay` ✅
- `experience/record.py` already carries `schema_version = 1`; reader counts
  + skips incompatible records with structured `replay_schema_mismatch` log ✅
- `OfflineRLConfig.real_supervised_weight` field added (default `0.0`); BC-style
  supervised loss injection ✅ **(Phase 2.1 complete)** — wired into the torch
  `train_offline_rl.py` loop via `OfflineRLTrainer.bc_update` (TD3+BC pattern,
  no-op at the `weight=0` default; `tests/integration/test_phase21_bc_into_offline_rl.py`
  proves byte-identity at `weight=0` and measurable parameter divergence at
  `weight>0` for both CQL and IQL). Retrofitting BC into the numpy-MLP
  `train_constitutional_rl.py` (PPO) remains deferred to a future PR-A1.5
  pending an obs→latent encoder bridge.
- `factory.build_replay_reader(cfg) -> ReplayReaderProtocol` wiring ✅

**Acceptance:**
- Empty LMDB produces a clean no-op (logged warning, training proceeds) ✅
- Mixer's realized ratio over 10 k draws is within 1% of target ✅ (parametrized test at 0.1/0.5/0.9)
- Integration test on a 10-episode synthetic LMDB → checkpoint produced ✅ (`tests/integration/test_phase2_replay_pipeline.py`, 6 tests: end-to-end LMDB→reader→BC→checkpoint round-trip, weight=0 byte-identity, chunk-size invariance at {1,3,4,64})
- Golden RSSM loss curve at fixed seed within ±1% of baseline ✅ (`tests/regression/test_phase2_rssm_golden.py` + helper at `tests/regression/_rssm_golden_helper.py`; baseline JSON at `tests/regression/fixtures/phase2_rssm_golden_baseline.json`; 8 tests covering existence, length, finite-keys, monotone-trend, baseline-tolerance with ±1% on recon/total and ±5% on KL, and prefix-stability at n∈{1,3,10}; regenerate via `MOUSEDROID_UPDATE_GOLDEN=1 pytest tests/regression/test_phase2_rssm_golden.py`)

### Phase 3a — VLA Protocol + `MockVLA` ✅ LANDED (`feat/phase3a-vla-protocol`)

Add an end-to-end Vision-Language-Action policy alongside the existing
`llm_gateway` + navigation-agent split, gated by a new `LoopConfig.policy_selector`
flag (default `nav_agent` → backwards compatible).

**Scope:**
- `src/mousedroid/vla/policy.py` — `VLAObservation`, `VLAAction`,
  `@runtime_checkable VLAPolicyProtocol`, `MockVLA` ✅
- New top-level `VLAConfig` Pydantic block (`Settings.vla`), default
  `backend="none"` for byte-identical legacy behavior ✅
- Factory hook `build_vla_policy` next to `build_llm_gateway` in `factory.py` ✅
- Orchestrator branch on `cfg.loop.policy_selector ∈ {nav_agent, vla, auto}` ✅
- Latency budget: `inference_timeout_s` defaults to `1.0 / cfg.loop.control_hz`;
  on timeout strict `vla` mode emits `vla_timeout_safe_stop` and returns a
  zero action; `auto` and `vla` (with `fallback_on_timeout=True`) fall back
  to the nav agent ✅
- 43 new unit tests (`tests/unit/vla/test_policy.py`,
  `tests/unit/orchestrator/test_policy_selector.py`) ✅

### Phase 3b — `DistilledVLAOnnx` + HF Weights Pull ✅ LANDED (`feat/phase3b-distilled-onnx-vla`)

Plug a distilled VLA student (SmolVLA / Pi0-FAST / distilled OpenVLA) behind
the `VLAPolicyProtocol`. Reuse `weights_manager.download_weights_from_huggingface`.

**Scope:**
- `src/mousedroid/vla/policy.py::DistilledVLAOnnx` — ORT InferenceSession with
  `TensorrtExecutionProvider` → `CUDAExecutionProvider` → `CPUExecutionProvider`
  ✅ (provider intersection preserves requested order; CPU fallback)
- New `[vla]` extra in `pyproject.toml` (`onnxruntime-gpu` aarch64 /
  `onnxruntime` non-aarch64 / `transformers` / `huggingface-hub`) ✅
- Import-graph isolation test: `import mousedroid.vla.policy` MUST NOT import
  `onnxruntime`. Lazy import inside `DistilledVLAOnnx.warmup` ✅
  (subprocess-isolated test in `tests/unit/vla/test_distilled_onnx.py`)
- `factory._build_distilled_onnx_vla` reuses
  `weights_manager.download_weights_from_huggingface` with clear error
  paths for missing-file-and-no-repo / download-failure ✅
- VLAConfig extended (`model_repo_id`, `model_filename`, `cache_dir`,
  `providers`, `warmup_iterations`, `h/z/action` IO names) ✅
- ~30 new unit tests across construction, provider resolution, warmup,
  predict (incl. shape-mismatch + `no_grad`), and factory wiring ✅
- Optional CI matrix entry that installs `[vla]` and runs the unit + smoke tests
  (advisory for the first PR; promote to required after a green week) — TBD

### Phase 4 — VLM-Derived Dense Rewards (VLAC) ✅ LANDED

Replace handcrafted reward shaping in `train_constitutional_rl.py` with VLM-
derived progress rewards. Plug into the existing `MultiObjectiveRewardModel`
via a new head — do **not** fork the aggregator.

**Delivered:**
- `src/mousedroid/reward/vlm_progress.py` — `VLMProgressBackend` Protocol,
  `MockVLMProgress` constant backend, `VLMProgressHead` with bounded
  `cachetools.LRUCache` keyed on rounded-tensor SHA-1 hashes ✅
- `RewardConfig.weight_vlm_progress: float = 0.0` (off by default) plus
  `RewardConfig.vlm_progress: VLMProgressConfig` (`enabled`, `cache_size`,
  `instruction`, `mock_progress_value`, `hash_decimals`) ✅
- `MultiObjectiveRewardModel` extended with optional `vlm_head` and
  optional `prev_obs`/`curr_obs`/`instruction` kwargs; **Law-1
  multiplicative sigmoid gate preserved** — the VLM term is added inside
  the harm-gated bonus when laws are present ✅
- `factory.build_reward_model(cfg)` factory; opt-in only when **both**
  `vlm_progress.enabled` and `weight_vlm_progress > 0` ✅
- `train_constitutional_rl.py` migrated to the factory ✅
- 20 new tests including Hypothesis property test (`max_examples=50`)
  asserting `out ≈ sigmoid(harm) * weight * vlm` to `1e-4` for any
  `(harm_bias, vlm_value, weight)` — Law 1 always zeros the VLM
  contribution ✅
- All 2940 unit tests still pass; `cachetools>=5.0` added to core deps ✅

### Phase 5 (stretch) — Real Physics Simulator

Replace the synthetic-sequence data generator with a real physics simulator
(MuJoCo MJX or Isaac Sim Lite). Decisions: mecanum wheel model fidelity, Orin
Nano in-the-loop sim vs offline-only, texture/mesh randomization. Notes-only
until Phase 3b has been in production ≥30 days.

### Phase 6 (stretch) — Real-time Co-training

Fine-tune the VLA policy on-device from continuously-logged real episodes
using LoRA-style adapters so we don't blow the 8 GB Orin RAM budget. Builds on
Phases 2 and 3.

---

## P0 — Deployment Hardening (Immediate)

### Automate Jetson Config Overlay Sync
Currently `/etc/mousedroid/jetson_production.yaml` is a manually synced read-only bind-mount
that drifts from the repo after each `git pull`. This must be automated.

**Options:**
1. Add a `post-receive` git hook on the Jetson that runs:
   ```bash
   sudo cp /opt/mousedroid/config/jetson_production.yaml /etc/mousedroid/jetson_production.yaml
   sudo systemctl restart mousedroid-docker
   ```
2. Convert the bind-mount to a symlink so `/etc/mousedroid/jetson_production.yaml` always
   resolves to the repo copy (requires changing the `docker-compose.jetson.yml` mount to `rw`).
3. Add an `ExecStartPre` step to the systemd service that copies the config.

**Recommended:** Option 3 — add to `scripts/mousedroid-docker.service`:
```ini
ExecStartPre=/bin/cp /opt/mousedroid/config/jetson_production.yaml /etc/mousedroid/jetson_production.yaml
```

---

## P0 — Voice Engine Quality ✅ LANDED

### Piper Model Diversity ✅
- ✅ Config-driven model selection via `voice.personality_to_model_map`
  (`src/mousedroid/voice/tts.py`, `VoiceConfig.resolved_tts_model_path()`)
- ✅ Voice latency benchmark at `scripts/benchmark_voice_latency.py` (230 LOC)
  with personality sweep + p95 target gate; unit-tested in
  `tests/unit/test_voice_latency_bench.py`

### Phrase Bank Expansion ✅
- ✅ Navigation events (`turn_left`, `turn_right`, `arrived`), battery warnings, and
  LLM translation acknowledgements all live in `src/mousedroid/voice/phrase_bank.py`
- ✅ Per-event intensity threshold tuning is in `VoiceConfig.phrase_overrides` /
  `VoiceConfig.event_intensity_thresholds`

---

## P1 — Test Coverage ✅ LANDED

### Integration Tests for TTS Pipeline ✅
- ✅ `tests/integration/test_tts_integration.py` (62 LOC) — verifies end-to-end WAV
  generation, sample count, and normalisation/gain clipping
- ✅ `tests/integration/test_speaker_tts_integration.py` (72 LOC) — UsbSpeaker +
  PiperTTS pipeline in mock hardware mode (no real device required)

### Smoke Harness Unit Tests ✅
- ✅ `tests/unit/test_smoke_harness.py` (46 LOC) — covers blocking-override
  resolution, stage-timeout export, non-blocking timeout classification,
  SUMMARY.md table generation, and voice-failure remediation enrichment

---

## Immediate Follow-up

1. ✅ **Jetson nightly Ten Pillars workflow** — [`.github/workflows/jetson-nightly.yml`](.github/workflows/jetson-nightly.yml)
   runs `scripts/validate_pillar.sh all` on a self-hosted Jetson runner every
   night and on `workflow_dispatch`. Advisory at first; promote to required
   after one full green week. Per-pillar logs and the markdown summary upload
   as build artifacts (30-day retention).
2. ✅ **Self-hosted runner installation path** — [`scripts/jetson-runner-install.sh`](scripts/jetson-runner-install.sh)
   (non-interactive, `--dry-run` safe), [`scripts/github-actions-runner.service.template`](scripts/github-actions-runner.service.template),
   and [`docs/jetson-runner-setup.md`](docs/jetson-runner-setup.md) operator
   runbook. Operator pastes a one-time `RUNNER_TOKEN` from the GitHub UI and
   the runner registers + starts under systemd. Closes the
   `[self-hosted, jetson]` label gap that left the nightly workflow queued.
3. ✅ **Phase 2 replay loop activated in production overlay** — `config/jetson_production.yaml`
   now declares the `training.replay_mixer.*` block (defaults inert,
   `alpha_target=0.0`). Operator-tunable live triage knob
   `debug_log_every_n` surfaces throttled `mixer_draw` /
   `replay_chunk_decoded` DEBUG lines without flooding the journal.
4. ✅ **Operator-facing failure-mode playbooks** — [`docs/playbooks/esp32-fail.md`](docs/playbooks/esp32-fail.md),
   [`gpio-fail.md`](docs/playbooks/gpio-fail.md), [`replay-fail.md`](docs/playbooks/replay-fail.md),
   [`bringup-fail.md`](docs/playbooks/bringup-fail.md) close the recovery-doc
   gap (camera/lidar/voice/promtool already shipped).
5. ✅ **Real-hardware endurance opt-in** — `MOUSEDROID_ENDURANCE_FORCE_REAL=1`
   flips `MOUSEDROID_MOCK_HARDWARE=false` so the operator can rerun the
   30 Hz endurance suite against the actual rover.
   `tests/performance/test_jetson_endurance.py` writes a JSON metrics
   snapshot to `reports/endurance/<utc>.json` per run for historical
   diff'ing.
6. ✅ **LMDB → training export CLI** — [`scripts/export_experience_to_training.py`](scripts/export_experience_to_training.py)
   streams rover experience records (Phase 2 chunked reader, memory-bounded)
   into msgpack-gz shards. `--dry-run` verifies the source size before any
   destination write.
7. Run `scripts/benchmark_voice_latency.py` on Jetson for the production personalities
   (`rocky`, `scout`, `friendly`) and capture median / P95 latency before any further voice changes.
8. Install `promtool` on the Windows validation host so the Prometheus rule stage in
   `bash scripts/ci.sh` becomes enforced rather than skipped — see
   [`docs/playbooks/promtool-install.md`](docs/playbooks/promtool-install.md).
9. Rebuild the Jetson image, restart `mousedroid-docker.service`, and rerun
   `scripts/jetson_full_smoke_run.sh` against the updated production config.
10. Use the recovery playbooks in `docs/playbooks/` for any camera, LiDAR, voice,
    GPIO, ESP32, replay-loop, or full-rover-bringup failures discovered during
    the next hardware validation pass.

### Pending follow-up (deferred to a separate PR)

- **Resilience wrappers for camera + voice + LLM gateway** — drop the three
  remaining bare driver constructions in `factory.py` behind the existing
  `CircuitBreaker` + `retry_async` machinery (per-driver opt-in via a new
  `cfg.resilience.<driver>.enabled` flag, defaults `False`). ESP32 and
  LiDAR are already wrapped (`src/mousedroid/resilience/`); these three
  are the residual gap. Cleanly composes with the current branch.
- **`set -e` on `scripts/jetson_full_smoke_run.sh`** (review-agent low finding) —
  the wrapper currently uses `set -uo pipefail` but not `-e`; inner stage
  logic tracks `OVERALL_FAIL` correctly so this is intentional, but
  top-level scripting errors silently continue. Align with `jetson_smoke_test.sh`
  which uses `set -euo pipefail`. Low risk — surface-level only.
- **importlib helper consolidation** (PR #105b finding deferred again) —
  the `spec_from_file_location` pattern appears in 6+ test files across
  the repo. Consolidate behind a shared `tests/conftest.py` helper so a
  future change can update one site instead of N.

---

## PR #106 follow-ups — Rover hardware fault recovery

PR #106's diagnostic surface surfaced (and the operator confirmed) that
the current Wave Rover ESP32 is **functionally dead** on UART, ROM
bootloader, AND WiFi AP broadcast across both rover USB-C ports. Repair
requires physical hardware work that the diagnostic surface cannot
perform remotely. Sequenced follow-ups:

1. **Bench-side hardware repair** — multimeter continuity probe ESP32
   UART0 TX → CP2102N RXD on the canonical USB-C port; visual inspect
   for damaged traces / lifted pads near the BOOT button (most likely
   stress point from the 2026-05-31 BOOT-button-during-power-cycle
   diagnostic). Worst case: replace the ESP32 module / Wave Rover
   driver PCB. Documented in
   `~/.claude/projects/<this>/memory/project_pr106_usbc_smoke_progress.md`.
2. **Firmware re-flash plan** — once UART works, the chip needs Waveshare
   stock firmware (or the original custom mousedroid build). Build
   path: clone `https://github.com/waveshareteam/ugv_base_ros`, install
   Arduino IDE + ESP32 board package + the SCServo / Adafruit_SSD1306
   / etc. libraries, flash via the Waveshare ESP32 download tool
   (Factory workmode). Stock firmware uses `Serial.begin(115200)` and
   responds to JSON `{"T":1,"L":<lv>,"R":<rv>}` — operator must then
   reconcile the mousedroid driver's custom `vx/vy/omega` keys against
   the stock `L/R` keys (and align `cfg.esp32.serial_baud` to 115200
   if running stock).
3. **Live-rover smoke re-run** — `bash scripts/jetson_full_smoke_run.sh`
   end-to-end with all stages blocking; confirm `power` stage
   `estop_latency_ms` lands well under
   `ESP32Config.emergency_stop_budget_ms` and `motor` loopback shows
   non-zero encoder velocity.
4. **Decoupled merge of PR #106** — the *code* in PR #106 is verified
   and not blocked on the live rover. The USB-C enumeration gate,
   factory override, and power-chain probe all unit-tested clean and
   land safely under the current "rover detached" smoke posture. The
   live-rover *motion* validation is a separate, hardware-blocked
   operational concern tracked here rather than as a PR-merge gate.

---

## Next Engineering Phases

### P1 — Physical-AI Phase 2 Replay Loop

- Add the real-episode replay ingestion path (LMDB reader, sim:real mixer, replay CLI, and
  supervised-loss integration) on top of the now-merged Phase 1 domain-randomization baseline.
- Keep the replay loop anchored to the active Jetson sensor stack: camera, LiDAR, audio, ESP32.

### P1 — Activation Work After Replay

- Operationalize the dual-stream RSSM path with explicit rollout telemetry and activation criteria.
- Activate FAISS-backed semantic retrieval only after index population and retrieval verification.
- Run the planned Jetson LLM benchmark pass before any production model swap.

### P2 — Voice Features Beyond Current Rollout

- Streaming TTS for longer utterances.
- Wake-word detection on the USB microphone.
- Phrase-bank expansion only when it is driven by observed runtime gaps, not as speculative churn.

### P2 — CI / Quality

- Add production-config validation to the local pre-commit path. ✅ **Done** —
  `scripts/validate_configs.py` + `tests/regression/test_config_overlays_load.py`
  + `config-validate` CI job; skip-marker (`# config-validator: skip`) supports
  deploy-time YAMLs.
- Publish coverage badge automation if it provides signal beyond the existing branch gate.
- Consider mutation testing for `voice/` and `hardware/audio/` once the current rollout is stable.

---

## Deferred / Out Of Scope

- **HC-SR04 ultrasonic work**: not part of the active Jetson production baseline until the sensor
  path is ready for real-device validation.
- **Robot arm platform**: deferred from the current roadmap until the Jetson + replay-loop +
  activation work is complete.

---

## Reference: Latest Jetson Smoke Snapshot (`20260426T231226Z`)

| Stage | Status | Notes |
| ----- | ------ | ----- |
| container_health | ✅ PASS | |
| app_health | ✅ PASS | |
| camera | ✅ PASS | ribbon IMX500 via `jetson_csi` backend |
| lidar | ✅ PASS | LD19, 360° coverage |
| audio | ✅ PASS | USB mic, 1,024-sample chunk |
| speaker | ✅ PASS | USB speaker, write-timeout polling |
| oled | ✅ PASS | I²C bus 7, SSD1306 128×64 |
| gpio | ✅ PASS | Jetson.GPIO |
| serial | ✅ PASS | ESP32 CP2102N at 1 Mbps |
| hardware_pytest | ✅ PASS | |
| voice | ✅ PASS | 39,424 audio samples, Piper `en_US-lessac-medium` |
| e2e | ✅ PASS | |
| system | ✅ PASS | |
| **ten_pillars** | ✅ **PASS** | **20/20 — all 10 pillars, pytest + factory probe** |
