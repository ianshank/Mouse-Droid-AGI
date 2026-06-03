# MouseDroidAGI — Next Steps

Rebased on 2026-06-02 for `chore/post-merge-ci-deploy-hygiene` after the Anthropic Claude LLM
gateway (PR #107) was deployed LIVE to the Jetson rover (PR #111) and the CI/test-isolation
hardening landed (PR #112, PR #113). Prior baseline: Ten Pillars validation campaign
(20/20 PASS on Jetson Orin Nano, 2026-04-26T23:55:42Z).

---

## ⚡ Current Next Steps (post-deployment, prioritized)

The deliberative-brain gateway is live on the rover; these are the real, grounded
follow-ups in priority order.

1. **[Security — P0] Rotate the `ANTHROPIC_API_KEY`.** The key was exposed in a chat
   transcript — treat it as compromised. Replace it in `/etc/mousedroid/docker.env` on
   the Jetson and restart the container (`docker compose -f docker-compose.jetson.yml up -d`
   or `sudo systemctl restart mousedroid-docker`). Confirm the cloud tier still authenticates
   after the swap.
2. **[Hardware blocker — P0] ESP32 repair.** The rover's ESP32 is functionally dead, so the
   NL→GoalVector path is validated *up to the GoalVector* but the GoalVector→wheels leg cannot
   be exercised end-to-end. This is the top blocker for true autonomous navigation. See the
   sequenced bench-repair + reflash plan under **PR #106 follow-ups** below.
3. **[Ops hygiene — P1] Re-point the rover's `/opt/mousedroid` source** to trunk
   (`claude/markdown-implementation-plan-aVJ2l`). Blocked by pre-existing root-ownership drift
   in the bind-mount (the container writes files as root), so a targeted
   `sudo chown ian:ian` of the tracked files is required first before the checkout will succeed.
4. **[Durability — P1] Make the per-host `docker.env` overrides durable.** The live overrides
   `MOUSEDROID_LLM__ENABLED=true` and `MOUSEDROID_LLM__N_GPU_LAYERS=0` (CPU fallback) currently
   live only in the Jetson's `/etc/mousedroid/docker.env` and would be lost on a rover
   reflash/swap. Capture them in a host-bootstrap script or a documented host overlay so a
   re-imaged rover comes up correctly without manual intervention.
5. **[Observability — P2] LLM-gateway cost/latency telemetry.** Wire Claude API token usage +
   round-trip latency into the existing Prometheus counters for mobile-rover budget visibility
   (see PR #107 follow-up #4 below for the counter shape).
6. **[Follow-up — P2] Issue #109** — MSE-6 greeting lifecycle wiring + integration/hardware
   test tiers (separate track).

---

## Current Baseline

- **Deliberative brain (Claude gateway) is LIVE on the rover** *(2026-06-02)*:
  - **PR #107** merged the Anthropic Claude LLM gateway + cloud→local `FallbackLLMGateway`
    composite.
  - **PR #111** deployed it live to the Jetson — Claude-haiku primary + Phi-3-mini CPU
    fallback, both tiers validated. The Jetson image was rebuilt to bake the Anthropic SDK
    in so the cloud tier survives `docker compose ... --force-recreate` (previously the SDK
    had to be hot-installed after every container recreate).
  - The 30 Hz reactive control loop stays deterministic and LLM-free; only NL→`GoalVector`
    translation routes through the gateway, OUTSIDE the hot loop.
- **CI + test-isolation hardening is merged** *(2026-06-02)*:
  - **PR #112** fixed the repo-wide cv2-eviction test-isolation footgun
    (`patch.dict` + `importlib.reload` was evicting `cv2`, breaking 19 tests under
    `pytest tests/`). Full-tree `pytest tests/` is green again.
  - **PR #113** fixed the dead `config-compat` CI gate (an invalid-workflow startup failure),
    hardened `check_config_compat.py` for cross-platform use, and added an `actionlint`
    CI gate so future workflow syntax errors fail loudly at PR time.
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
- **SHA-pin GitHub action references** (CodeRabbit PR #106 finding 4) —
  `.github/workflows/ci.yml` currently uses `actions/checkout@v4` and
  `actions/setup-python@v5` tag references. Security best practice is
  SHA pinning (`actions/checkout@<sha>`) to defend against tag rebasing
  / supply-chain attacks. Best done as a single sweep across the
  workflow file with Dependabot configured to auto-bump the SHAs.
  Deferred from PR #106 because it spans multiple workflows + needs a
  Dependabot config update in the same PR for sustainable maintenance.

---

## PR #106 follow-ups — Rover hardware fault recovery ⛔ ACTIVE TOP BLOCKER

PR #106's diagnostic surface surfaced (and the operator confirmed) that
the current Wave Rover ESP32 is **functionally dead** on UART, ROM
bootloader, AND WiFi AP broadcast across both rover USB-C ports. Repair
requires physical hardware work that the diagnostic surface cannot
perform remotely. **This is the #2 current next step** (see top of file):
with the Claude gateway now live, NL→`GoalVector` is validated, but the
`GoalVector`→wheels leg can only be closed once the ESP32 is repaired.
Sequenced follow-ups:

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

## PR #107 follow-ups — Anthropic Claude LLM gateway ✅ DEPLOYED (PR #111)

PR #107 landed the Anthropic Claude backend + `FallbackLLMGateway`
composite cleanly under three rounds of review (Gemini + independent
`feature-dev:code-reviewer` + `code-explorer` + `security-auditor`),
with `mergeStateStatus: CLEAN` and the security audit returning PASS on
all 8 critical checks. **PR #111 deployed it live to the Jetson** —
Claude-haiku primary + Phi-3-mini CPU fallback, both tiers validated,
image rebuilt to bake the Anthropic SDK in so the cloud tier survives
`--force-recreate`. Status of the original follow-ups:

1. ✅ **Live cloud round-trip benchmark** — exercised during the PR #111
   live deploy; both the cloud (Claude-haiku) and local (Phi-3-mini CPU)
   tiers were validated against the rover. Note the live overlay still
   relies on `latency_target_ms: 5000` to avoid `anthropic_gateway_slow`
   spam on normal cloud round-trips. The repro one-liner below remains
   useful for re-benchmarking after a model swap:
   ```bash
   MOUSEDROID_JETSON_CONFIGS=config/jetson_claude_pilot.yaml \
       python -c "
   import asyncio
   from mousedroid.config.loader import load_settings
   from mousedroid.factory import build_llm_gateway, build_llm_injection_filter
   cfg = load_settings('config/jetson_claude_pilot.yaml')
   gw = build_llm_gateway(cfg, injection_filter=build_llm_injection_filter(cfg))
   async def go():
       await gw.start()
       for cmd in ['go forward', 'turn left then stop', 'patrol the perimeter']:
           print(cmd, '->', await gw.translate_mission(cmd))
       await gw.stop()
   asyncio.run(go())
   "
   ```
2. ✅ **Cold-network / failover behavior** — the local Phi-3-mini CPU
   fallback tier was validated as part of the PR #111 deploy, confirming
   the composite serves GoalVectors off-network. A scripted WAN-drop /
   egress-block soak (asserting `fallback_primary_to_secondary` and the
   `fallback_primary_retry_attempt` recovery event) is still worth
   capturing as a documented operator drill once the ESP32 is repaired and
   a full end-to-end mission can run.
3. **`__init__.py` lazy-import hardening** (round-3 Low finding,
   deferred) — `src/mousedroid/llm_gateway/__init__.py` currently eager-
   imports `AnthropicLLMGateway` + `FallbackLLMGateway`. Per CLAUDE.md
   invariant 1 (concrete types live behind the factory), these should
   be removed from the package surface OR wrapped in
   `TYPE_CHECKING`. The eager imports don't crash today (SDK loads
   lazily in `start()`), but they make `import mousedroid.llm_gateway`
   in a tool / test load the concrete classes unnecessarily.
4. **Cloud token-cost telemetry** *(current next step — see top of file)* —
   add a Prometheus counter for `anthropic_request_total{model,outcome}`
   (plus round-trip latency) so operators can see WAN round-trips, failure
   rates, and token spend over time for mobile-rover budget visibility.

---

## Claude Code on Jetson — install + configure runbook

Run the Claude Code agent natively on the Jetson Orin Nano so engineers
can drive the rover from a session that has filesystem + git +
`mousedroid` access without round-tripping through a workstation. This
runbook assumes the Jetson is at `ian@mousedroid.local` per
`~/.claude/projects/<this>/memory/reference_jetson_hardware.md`.

### Prerequisites (one-time)

```bash
# SSH into the Jetson
ssh ian@mousedroid.local

# Confirm L4T + arch — Claude Code ships an aarch64 Node binary.
uname -m            # expect: aarch64
cat /etc/nv_tegra_release | head -1   # expect: R36.x (JetPack 6.x)

# Node.js 18+ is required. JetPack 6 ships with Node 12 — upgrade via NodeSource:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version       # expect: v20.x
npm --version        # expect: 10.x
```

### Install Claude Code

```bash
# Global install via npm (no separate binary needed — Node CLI):
sudo npm install -g @anthropic-ai/claude-code

# Verify
claude --version    # expect: 2.x or newer

# First-run auth flow — opens a browser-link prompt. On a headless Jetson,
# copy the URL to your workstation, complete OAuth there, paste the
# returned token back into the Jetson terminal.
claude --setup-token
```

If the Jetson is **fully headless** and you cannot OAuth, supply an
Anthropic API key directly instead:

```bash
# In ~/.bashrc or systemd EnvironmentFile for service-mode use:
export ANTHROPIC_API_KEY=sk-ant-...   # NEVER commit; this is the SAME
                                       # key used by the mousedroid LLM
                                       # gateway (PR #107) — operators
                                       # can share it between agent +
                                       # gateway by setting it once in
                                       # the shell env / docker.env.
```

### Configure for the mousedroid repo

```bash
# Claude Code reads CLAUDE.md from the working tree. The repo's
# CLAUDE.md (this file's neighbour) already encodes all the project
# invariants (factory-first DI, no-hardcoded-values, structlog,
# asyncio, mypy --strict, etc.) — no extra config needed.
cd /opt/mousedroid
claude

# Useful Jetson-specific aliases (drop into ~/.bashrc):
alias claude-mousedroid='cd /opt/mousedroid && claude'
alias claude-smoke='cd /opt/mousedroid && claude "run jetson_full_smoke_run.sh and report the SUMMARY.md"'
alias claude-firmware='cd /opt/mousedroid && claude "diagnose the rover ESP32 — see SKILLS.md rover-firmware-diagnosis"'
```

### Recommended Jetson-specific settings

Edit `~/.config/claude-code/settings.json` (or run `claude config`):

```jsonc
{
  // Larger context lets Claude hold the full src/mousedroid/ tree in
  // memory for refactoring sweeps without compaction noise.
  "default_model": "claude-sonnet-4-6",

  // The Jetson's 8 GB unified memory means we should NOT spawn
  // many background subagents in parallel. Cap at 2.
  "max_parallel_subagents": 2,

  // Permission boundary: allow everything UNDER /opt/mousedroid and
  // /etc/mousedroid but DENY writes elsewhere on the Jetson (don't
  // accidentally touch /etc/systemd or ~/.ssh from an agent).
  "permissions": {
    "fileWriteRoots": ["/opt/mousedroid", "/etc/mousedroid", "/tmp"],
    "denyShellCommands": ["sudo rm -rf", "shutdown", "reboot"]
  }
}
```

### Service-mode (optional, for unattended use)

If you want Claude Code running as a background task that can be poked
via SSH (e.g., for the `coderabbit:autofix` skill firing against a
pending PR), drop this systemd unit at
`/etc/systemd/system/claude-code-agent.service`:

```ini
[Unit]
Description=Claude Code agent (operator-driven; not for production decisions)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ian
WorkingDirectory=/opt/mousedroid
EnvironmentFile=/etc/mousedroid/docker.env
ExecStart=/usr/bin/claude --listen 127.0.0.1:9229
Restart=on-failure
RestartSec=10
# Don't expose to the LAN — bind to loopback only. Use SSH port-forwarding
# (ssh -L 9229:127.0.0.1:9229 jetson) from the workstation to attach.

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable claude-code-agent.service
sudo systemctl start claude-code-agent.service
sudo journalctl -u claude-code-agent.service -f
```

### Hardening — secret hygiene + permission boundary

- The same `ANTHROPIC_API_KEY` reaches BOTH Claude Code (via env) AND
  the mousedroid LLM gateway (via `MOUSEDROID_LLM__API_KEY` SecretStr
  override). Keep it in `/etc/mousedroid/docker.env` only — never in
  `~/.bashrc` that gets shared on screen-share.
  - ⚠️ **The currently-deployed key was exposed in a chat transcript and
    MUST be rotated** (current next step #1, top of file). Generate a fresh
    key, replace it in `/etc/mousedroid/docker.env`, restart the container,
    then revoke the old one in the Anthropic console.
- The `denyShellCommands` list above is the minimum; tighten further
  if Claude Code will run with `sudo` privileges (it shouldn't —
  prefer running as user `ian` and only let it ask for sudo
  interactively when needed).
- `/etc/mousedroid/jetson_production.yaml` is bind-mounted into the
  Docker container read-only. Claude Code's `fileWriteRoots` includes
  it for edits, but operators should run `sync_jetson_overlay.sh`
  after edits to refresh the container view.
- The `--listen 127.0.0.1:9229` binding in the systemd unit is
  loopback-only on purpose; opening to `0.0.0.0` would let anyone on
  the WiFi LAN drive the rover via the agent. Use SSH local-forward
  (`ssh -L 9229:127.0.0.1:9229 jetson`) to attach from a workstation.

### Verifying the install

```bash
# Smoke-test the agent against a known-good question:
claude "summarize the mission of this repository in 3 bullets"

# Confirm it can read structured project context:
claude "what does build_llm_gateway return when fallback_backend='none'?"
# expect: a reference to factory.py + the "return primary" branch.
```

### When to NOT use Claude Code on the Jetson

- **In the 30 Hz reactive control loop.** This loop is intentionally
  LLM-free; the safety projector + MCTS + ESP32 driver are the only
  decision-makers there. Claude Code is for operator workflows
  (debugging, smoke runs, doc edits, rebases), NEVER in the rover's
  hot path.
- **During an active mission.** The agent will compete with the
  orchestrator for the Jetson's ~7 GB usable RAM. Park the
  orchestrator (`docker stop mousedroid`) before launching Claude
  Code for any meaningful work.
- **When the rover battery is below `safety.battery_critical_v`.**
  Claude Code's LLM round-trips can take 5-10 s and prolong the
  low-battery condition. Charge first.

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
