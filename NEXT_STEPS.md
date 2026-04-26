# MouseDroidAGI — Next Steps

Prioritised development items after the `hardware/jetson-full-smoke` branch merge.

---

## P0 — Physical AI Roadmap (Phases 2 → 6)

Phase 1 (per-episode domain randomization for RSSM pretraining) ships in this
branch. Subsequent phases close the remaining three Physical AI gaps and each
lands in an isolated PR off the default branch. Dependency direction is
strictly **Phase 1 → 2 → 3 → 4**; Phases 5 and 6 are deferred until Phase 3b
has been in production for ≥30 days.

### Phase 2 — Real-Episode Replay Loop (sim-to-real feedback)

Wire the existing LMDB experience logger back into the offline training
pipeline so successes and failures from real-world rollouts continuously refine
the RSSM and the Constitutional-RL policy. Closes the second of four gaps.

**Scope:**
- `src/mousedroid/training/replay/lmdb_reader.py` — async streaming iterator
  over an LMDB env (chunks of 64 records via `asyncio.to_thread`; never load
  the whole DB into RAM on the 8 GB Orin)
- `src/mousedroid/training/replay/mixer.py` — ratio-controlled sampler over
  `(sim_iter, real_iter)` with a deterministic `numpy.random.Generator`; ramped
  `alpha` from 0.0 → target over a configurable number of steps (RL-Co two-stage)
- `training/replay_real_episodes.py` — CLI with `--dry-run`, `--use-real-replay`
- `experience/record.py` already carries `schema_version = 1`; add a versioned
  reader contract that refuses incompatible records with a counter
- Auxiliary BC-style supervised loss in PPO via `OfflineRLConfig.real_supervised_weight`

**Acceptance:**
- Empty LMDB produces a clean no-op (logged warning, training proceeds)
- Mixer's realized ratio over 10 k draws is within 1% of target
- Integration test runs the pipeline end-to-end on a 10-episode synthetic LMDB
  and verifies a checkpoint is produced
- Golden RSSM loss curve at fixed seed within ±1% of baseline

### Phase 3a — VLA Protocol + `MockVLA`

Add an end-to-end Vision-Language-Action policy alongside the existing
`llm_gateway` + navigation-agent split, gated by a new `LoopConfig.policy_selector`
flag (default `nav_agent` → backwards compatible).

**Scope:**
- `src/mousedroid/vla/policy.py` — `VLAObservation`, `VLAAction`, `VLAConfig`,
  `@runtime_checkable VLAPolicyProtocol`, `MockVLA`
- Factory hook `build_vla_policy` next to `build_llm_gateway` in `factory.py`
- Orchestrator branch on `cfg.loop.policy_selector ∈ {nav_agent, vla, auto}`
- Latency budget: `inference_timeout_s` defaults to `1.0 / cfg.loop.control_hz`;
  on `TimeoutError` the safety monitor emits `vla_timeout_safe_stop`

### Phase 3b — `DistilledVLAOnnx` + HF Weights Pull

Plug a distilled VLA student (SmolVLA / Pi0-FAST / distilled OpenVLA) behind
the `VLAPolicyProtocol`. Reuse `weights_manager.download_weights_from_huggingface`.

**Scope:**
- `src/mousedroid/vla/policy.py::DistilledVLAOnnx` — ORT InferenceSession with
  `TensorrtExecutionProvider` → `CUDAExecutionProvider` → `CPUExecutionProvider`
- New `[vla]` extra in `pyproject.toml`:
  ```toml
  vla = [
      "onnxruntime-gpu>=1.18; platform_machine=='aarch64'",
      "onnxruntime>=1.18;     platform_machine!='aarch64'",
      "transformers>=4.40",
  ]
  ```
- Import-graph isolation test: `import mousedroid.vla.policy` MUST NOT import
  `onnxruntime`. Lazy import inside `DistilledVLAOnnx.warmup`.
- Optional CI matrix entry that installs `[vla]` and runs the unit + smoke tests
  (advisory for the first PR; promote to required after a green week)

### Phase 4 — VLM-Derived Dense Rewards (VLAC)

Replace handcrafted reward shaping in `train_constitutional_rl.py` with VLM-
derived progress rewards. Plug into the existing `MultiObjectiveRewardModel`
via a new head — do **not** fork the aggregator.

**Scope:**
- `src/mousedroid/reward/vlm_progress.py` — `VLMProgressHead` registers as a
  weighted term alongside truthfulness/helpfulness/safety/engagement
- `RewardConfig.weight_vlm_progress: float = 0.0` (off by default for safety)
- LRU caching keyed by `(prev_hash, curr_hash, instruction_hash)` via
  `cachetools.LRUCache(maxsize=cfg.reward.vlm_progress.cache_size)` — never
  `functools.lru_cache` (no memory cap)
- Constitutional override hypothesis test: a contrived high VLM reward that
  violates Law 1 must still be Law-1-blocked (multiplicative sigmoid gate
  preserved)

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

## P0 — Voice Engine Quality

### Piper Model Diversity
- Download and evaluate additional Piper voices (`en_US-amy-medium`, `en_GB-alba-medium`)
  for the Rocky personality; make model selection config-driven (`voice.personality_to_model_map`)
- Add a voice latency benchmark to `scripts/benchmark_latency.py`

### Phrase Bank Expansion
- Current phrase bank covers: startup, shutdown, obstacle, error, sensor_recovery
- Add: navigation events (`turn_left`, `turn_right`, `arrived`), battery warnings, LLM
  translation acknowledgements
- Support per-event `intensity_threshold` tuning in `VoiceConfig.phrase_overrides`

---

## P1 — Test Coverage

### Integration Tests for TTS Pipeline
`tests/unit/test_piper_tts.py` provides mock-only coverage. Add integration tests:
- `tests/integration/test_tts_integration.py` — verify end-to-end WAV generation using an
  actual (or mocked) Piper voice object; validate sample count and normalization range
- `tests/integration/test_speaker_tts_integration.py` — UsbSpeaker + PiperTTS pipeline test
  in mock hardware mode (PyAudio mock, no real device required)

### Smoke Harness Unit Tests
`scripts/jetson_full_smoke_run.sh` logic is not currently unit-tested. Add:
- `tests/unit/test_smoke_harness.py` — test SUMMARY.md generation, voice-failure enricher
  output, and stage timeout enforcement using subprocess mocks or a bats-style shell test

---

## P1 — Jetson Bootstrap Hardening

### Automated Bootstrap Script
`scripts/jetson_bootstrap.sh` does not currently:
- Create `/etc/mousedroid/` and install config overlay
- Create the `mousedroid` system user
- Set up the NVMe SSD mount and swap

Add a complete `scripts/jetson_bootstrap.sh` that is idempotent and can be re-run safely,
covering:
1. SSD partitioning + mount + swap
2. Docker data-root relocation to `/mnt/ssd/docker`
3. `/etc/mousedroid/` setup + overlay install
4. `git pull` + `docker compose build` + `systemctl enable`

### Health-Check Recovery Playbook
Document the full recovery procedure for each smoke stage failure in `docs/playbooks/`:
- `voice-fail.md` — model not found / Piper not installed / overlay not synced
- `lidar-fail.md` — LD19 not detected / wrong serial port / baud rate mismatch
- `camera-fail.md` — ribbon camera not initialised / libargus socket absent

---

## P2 — Voice Engine Features

### Streaming TTS
Current architecture synthesises the full phrase before playback begins. For longer phrases,
implement streaming synthesis: synthesise and enqueue audio in chunks of `chunk_size` frames
while the earlier chunks are already playing.

### Wake-Word Detection
Add a lightweight wake-word detector (e.g., openWakeWord) on the USB microphone so the droid
can acknowledge spoken commands before passing them to the LLM Gateway.

### Volume Control via Config
Expose a `voice.output_volume` (0.0–1.0) in `VoiceConfig`; apply gain in `_synthesize_sync`
before writing to the speaker queue.

---

## P2 — CI / Quality

### Pre-commit Hook: Config YAML Validation
Extend `.git/hooks/pre-commit` to run `python -c "from mousedroid.config.schema import Settings; import yaml; Settings.model_validate(yaml.safe_load(open('config/jetson_production.yaml')))"` so malformed production config is caught before push.

### Coverage Dashboard
Integrate `coverage-badge` generation into CI so the `coverage-branch.json` report is
surfaced as a shield badge in README after each push.

### Mutation Testing
Run `mutmut` or `cosmic-ray` on `src/mousedroid/voice/` and `src/mousedroid/hardware/audio/`
to identify under-tested branches before the next feature round.

---

## P3 — Architecture

### Dual-Stream RSSM + CfC Activation
`jetson_dual_stream.yaml` exists but the CfC stream requires manual human activation (it is
disabled by default pending real-robot experience data collection). Define the data threshold
(e.g. 10 000 experience episodes) and automated switchover condition.

### LLM Gateway: Phi-3 → Llama 3.1
The current model URL points to Llama 3 8B. Evaluate Phi-3 mini (4 GB, faster on Jetson) vs
Llama 3.1 8B Q4 for NL→velocity accuracy and update `config/jetson_production.yaml` after
benchmarking.

### FAISS Semantic Memory Activation
`MemoryConfig.enabled` is currently `false` in production. Define the activation criteria
(e.g. FAISS index populated with ≥ 1 000 semantic concepts) and write the enabling migration
procedure.

---

## Reference: Smoke Stage Status (`20260425T192408Z`)

| Stage | Status | Notes |
|-------|--------|-------|
| container_health | ✅ PASS | |
| app_health | ✅ PASS | |
| camera | ✅ PASS | ribbon IMX500 via jetson_csi backend |
| lidar | ✅ PASS | LD19, 360° coverage |
| audio | ✅ PASS | USB mic, 1 024-sample chunk |
| speaker | ✅ PASS | USB speaker, write-timeout polling |
| oled | ✅ PASS | I²C bus 7, SSD1306 128×64 |
| gpio | ✅ PASS | Jetson.GPIO |
| serial | ✅ PASS | ESP32 CP2102N at 1 Mbps |
| hardware_pytest | ✅ PASS | |
| **voice** | ✅ **PASS** | 39,424 audio samples, Piper en_US-lessac-medium |
| e2e | ✅ PASS | |
| system | ✅ PASS | |
