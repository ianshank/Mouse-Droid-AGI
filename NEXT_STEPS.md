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

## Immediate Follow-up

1. Integrate `scripts/validate_pillar.sh all` into CI (or Jetson nightly job) so the Ten Pillars
   campaign is run automatically on each deployment — currently only run manually on Jetson.
2. Run `scripts/benchmark_voice_latency.py` on Jetson for the production personalities
   (`rocky`, `scout`, `friendly`) and capture median / P95 latency before any further voice changes.
3. Rebuild the Jetson image, restart `mousedroid-docker.service`, and rerun
   `scripts/jetson_full_smoke_run.sh` against the updated production config.
4. Use the recovery playbooks in `docs/playbooks/` for any camera, LiDAR, or voice failures
   discovered during the next hardware validation pass.

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

- Add production-config validation to the local pre-commit path.
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
