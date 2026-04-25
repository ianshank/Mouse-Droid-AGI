# MouseDroidAGI — Next Steps

Prioritised development items after the `hardware/jetson-full-smoke` branch merge.

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
