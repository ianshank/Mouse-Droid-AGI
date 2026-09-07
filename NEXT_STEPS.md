# MouseDroid — Next Steps

Forward-looking priorities only. Landed work moves to `CHANGELOG.md` (see the
"Historical record" section there); the machine-readable source of "what's
next" is the feature catalog (`features.yaml` + `python scripts/select_next.py`
— Golden Rule per `HARNESS_SPEC.md`). Advisory size guard:
`tools/doc_hygiene.py NEXT_STEPS.md`. Priorities below were re-baselined
2026-08-07 against the vendor audit
(`docs/analysis/rover-jetson-integration-gaps.md`) and its line-level
verification (`docs/analysis/next-steps-peer-review-2026-08-07.md`).

**Phase vocabulary:** "Phase N" in this file refers exclusively to the
Physical-AI roadmap below (Phase 1 domain randomization → Phase 6 LoRA
co-training). CHARTER §5 **M6** is on-device RSSM refinement (software
landed, default-OFF, soak-gated) — a different Phase 6 than the LoRA stretch.
The May-16 `docs/planning/IMPLEMENTATION_PLAN.md` and
`docs/planning/NEXT_STEPS.md` are snapshots; this file is the living plan.
The legacy v0.3.0 execution-plan numbering lives only in those snapshots
and is annotated as such there.

---

## ⚡ Current Next Steps (prioritized)

Catalog **F-008** (USB-C rover smoke in `features.yaml`) is the hardware next
feature (`python scripts/select_next.py`). Smoke-report F-008 (telemetry :8080,
2026-05-12) is a different namespace — see
`docs/architecture/ADR-013-f-number-namespaces.md`. Closing catalog F-008 is
bench work, not a coding sprint.

1. **[Security — P0] Rotate the `ANTHROPIC_API_KEY`.** Treat the chat-exposed
   key as compromised: inventory consumers, replace on Jetson, restart,
   confirm `tools/llm_latency_probe.py --iterations 3`, revoke the old key.
   Software half is F-015 (operator leftover).
2. **[Hardware blocker — P0] ESP32 diagnosis + repair.** Re-test stock command
   set (`MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock`, 115200 baud) and probe
   `U0TX`/`U0RX` (pins 10/8 → `/dev/ttyTHS1`). Gate for catalog **F-008**.
   Time-box: 2 bench sessions.
3. **[Ops leftover of F-025] Flip `MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock`**
   in `/etc/mousedroid/docker.env` after `deployments/jetson-image.json` is
   re-pinned. Software retarget is already in tree.
4. **[Ops hygiene — P1] Re-point the rover's `/opt/mousedroid` source** to trunk.
   Targeted `sudo chown ian:ian` on tracked files before checkout.
5. **[Ops leftover of F-017] Run `scripts/host_bootstrap.sh`** after reflash and
   enable `host_env.enabled` in the Jetson overlay.
6. **[Bring-up — P1] Full rover bring-up + unified dashboard.**
   `docs/runbooks/jetson-full-bringup.md`. Probe-first motors;
   `MOUSEDROID_ESP32__ENABLED=false` if the board is unpowered.
7. **[Sensing — P1] IMU attitude.** F-036 parses stock `r`/`p`/`y` onto
   `EncoderReading` without an RSSM slot. F-040 (optional slot 5, `imu_dim=0`)
   waits until that parse is real and preferably until F-008 has hardware
   frames. Do not un-zero `battery_critical_v` by stealth.
8. **[Ops leftover of F-018] Run `bash scripts/jetson_full_validation.sh`** on
   the rover with trend journaling.
9. **[Ops leftover of F-019] Import `docs/grafana_dashboard.json`** and load
   `config/prometheus/alerts.yml` on the rover Prometheus.
10. **[Ops leftover of F-020] Review `scripts/dead_code_audit.py` output** and
    promote advisory CI stages when due (`.github/advisory_stages.yaml`).
    Promote `test-windows` and `security` (pip-audit) when their windows close.
11. **[Docs — P2] Reconcile hardware docs with chassis (audit R9).** WAVE ROVER
    is 4WD skid-steer, encoder-less, IMX708 camera, 3S 18650 UPS.
12. **[Ops leftover of F-023] Distillation spike** per
    `docs/runbooks/jetson-alayaworld-spike.md` and `scripts/compare_drift.py`.
13. **[Portfolio — P2] Record a 60-second hardware demo clip** on Jetson and
    link it in README (host as a release asset).
14. **[Portfolio — P2] Git-history purge** post-reframe PR (#167):
    `scripts/purge_history.sh` and rename slug to `mouse-droid`.
15. **[Hygiene — P3] Migrate 50 test-fixture `np.random.*` NPY002 call sites**
    off global state. `src/mousedroid` is clean; tests are baselined. Design
    decision first (shared Generator fixture vs per-file instance).
16. **[Hygiene — P3] Enumerate `check_branch_coverage.py` factory/orchestrator
    prefixes (F-042).** Unbounded `_ALLOWED_DIR_PREFIXES` exempt ~30% of
    `src/mousedroid` from the changed-line gate. Keep gating
    `factory/on_device_learning.py`, `mcp_harness.py`, `_replay_batch_helpers.py`.
17. **[Testing — P3] ADR-017 mixin split gaps (latent).** Facade completeness
    walks only `ast.Module.body`; no `factory/*.py` uses the nested-def pattern
    today. Optional: property-test orchestrator kwargs vs `_OrchestratorState`.
18. **[Testing — parked, not missing] `tests/functional/` + `tests/user_journey/`
    cover parked `AutonomousOrchestrator` (ADR-016), not production
    `MouseDroidOrchestrator`.** Production is already in e2e / integration /
    smoke. F-038 relabels the CI step; do not twin these under `mock_hardware`.
19. **[Docs — P4, blocked by F-008] `arm/CLAUDE.md` still cites `mock_arm.py`**
    (real: `hardware/mock_arm_driver.py`). Freeze stays until F-008 is `done`.

---

## P0 — Physical AI Roadmap (Phases 2 → 6)

Dependency direction is strictly **Phase 1 → 2 → 3 → 4**; Phase 6 is deferred until Phase 3b has soaked ≥30 days (Phase 5 has landed -- see below).

- **Phase 1 — domain randomization** ✅ landed (see CHANGELOG).
- **Phase 2 — real-episode replay loop** ✅ landed incl. Phase 2.1 BC injection.
- **Phase 3a/3b — VLA protocol + DistilledVLAOnnx** ✅ landed.
- **Phase 4 — VLM-derived dense rewards (VLAC)** ✅ landed.
- **Phase 5 — real physics simulator** ✅ landed — `src/mousedroid/sim/mujoco_rover_env.py`'s `RoverMuJoCoEnv` (MuJoCo skid-steer, `RoverEnvProtocol`-conformant, RSSM pretrained on its episodes) replaced the NumPy kinematic sim. Matches `docs/CHARTER.md` §5's M5 ✅. This entry previously called it deferred-stretch, written before the simulator landed and not updated after — the "deferred until 30-day soak" framing described the T3+ arm-training unfreeze gate below, not Phase 5 itself.
- **Phase 6 (stretch) — real-time LoRA co-training** — on-device fine-tuning;
  builds on Phases 2 + 3. Not CHARTER M6 (RSSM soak).

### Training arc (T-numbers)

**Arm training arc PAUSED at T2.** Unfreeze condition: **F-008** (rover hardware gate) reaches `done` on Jetson AND Phase 3b has soaked ≥30 days. Until then T3+ and arm skills stay frozen. T2 (MLflow training observability) is landed (`docs/runbooks/mlflow-local-ui.md`).

---

## Open engineering follow-ups

0a. **[Hygiene — needs a dedicated pass] First-ever vulture dead-code audit run: 447
    findings.** `scripts/dead_code_audit.py` (F-020) had been CI-wired but never
    actually run + triaged — `scripts/vulture_allowlist.py` was empty. Ran it
    2026-08-16 at the default 60% confidence, output in
    `reports/dead_code/2026-08-16.json`. 2 spot-checked findings
    were confirmed Protocol/DI false positives (vulture can't trace
    protocol-typed call sites). At this volume, needs a dedicated triage pass
    batched by module, not a rubber-stamp allowlist add or blind deletion.

1. Run `scripts/benchmark_voice_latency.py` on Jetson for the production personalities
   (`rocky`, `scout`, `friendly`) and capture median / P95 latency before any further voice changes.
2. Install `promtool` on the Windows validation host so the Prometheus rule stage in
   `bash scripts/ci.sh` becomes enforced rather than skipped — see
   [`docs/playbooks/promtool-install.md`](docs/playbooks/promtool-install.md).
3. Rebuild the Jetson image, restart `mousedroid-docker.service`, and rerun
   `scripts/jetson_full_smoke_run.sh` against the updated production config.
4. Use the recovery playbooks in `docs/playbooks/` for any camera, LiDAR, voice,
   GPIO, ESP32, replay-loop, or full-rover-bringup failures discovered during
   the next hardware validation pass.

5. **Workforce (F-024) is catalog-done.** Remaining hook/config work on a live
   Jetson is operator leftover, not a coding sprint. `performance` / `security`
   keep their advisory windows in `.github/advisory_stages.yaml`.

### Pending follow-up (deferred to a separate PR)

- **importlib helper consolidation** — partially closed by
  `tests/_script_loader.py`; sweep the remaining inline
  `spec_from_file_location` call sites onto it.
- **Scripted WAN-drop failover drill** — capture the operator drill asserting
  `fallback_primary_to_secondary` + `fallback_primary_retry_attempt` once the
  ESP32 is repaired and a full end-to-end mission can run.
- **[Cognitive integration — F-022] Soak-gate the growth pillar before enabling it.** Catalog entry is now `done` (`27b5233`); the soak gate below is the operator half.
  The `growth` pillar (VLA knowledge distillation) is now wired as a default-OFF,
  off-loop coordinator (distilled student persisted to a SHA-256 slot, never
  hot-swapped). Keep `growth.enabled` off on the live rover until a soak gate
  passes (mirror the Phase-6 on-device-learning discipline). Next-in-arc: wire the
  two remaining unwired pillars (`meta`, `scaling`) once growth has soaked.

---

## PR #106 follow-ups — Rover hardware fault recovery ⛔ ACTIVE TOP BLOCKER

PR #106's diagnostic surface surfaced (and the operator confirmed) that
the current Wave Rover ESP32 is **functionally dead** on UART, ROM
bootloader, AND WiFi AP broadcast across both rover USB-C ports. Repair
requires physical hardware work that the diagnostic surface cannot
perform remotely. Sequencing re-baselined 2026-08-07 against the vendor
audit (R2/R7 first — they cost minutes and can flip the decision):

0. **Cheap retests before any spend (audit R2/R7)** — (a) rerun the serial
   probe with the stock command set (`MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock`,
   which derives 115 200 baud and switches the probe to `{"T":130}`, a read that
   elicits a reply; the original diagnosis ran at 1 Mbaud sending legacy commands,
   under which a live stock board is indistinguishable from a dead one);
   (b) jumper the driver board's 40-pin `U0TX`/`U0RX`
   (pins 10/8) to the Jetson header → `/dev/ttyTHS1` and probe there. A reply
   on either path means the fault is the CP2102N bridge / USB-C port, not the
   ESP32 — which changes repair-vs-replace. Silence on both corroborates a
   dead/unpowered ESP32 (consistent with the absent WiFi AP).
1. **Bench-side hardware repair** — multimeter continuity probe ESP32
   UART0 TX → CP2102N RXD on the canonical USB-C port; visual inspect
   for damaged traces / lifted pads near the BOOT button (most likely
   stress point from the 2026-05-31 BOOT-button-during-power-cycle
   diagnostic). Worst case: replace the ESP32 module / Wave Rover
   driver PCB. **Time-box: 2 bench sessions, then decide repair vs
   replace.** Record the flashed firmware binary (path + hash) alongside
   F-008's `implemented_in` when the gate closes.
2. **Firmware plan — stock, not custom.** No custom mousedroid firmware
   exists in the repo (audit R1: `scripts/flash_esp32.sh` references a
   binary that was never committed). Target **stock Waveshare firmware**
   (`waveshareteam/ugv_base_general`, `General_Driver/`, Factory-workmode
   flash via the Waveshare download tool; `Serial.begin(115200)`) and let
   the **F-025 driver retarget** (prioritized item 3 above) speak its
   command set — no firmware-side customization required.
3. **Live-rover smoke re-run** — `bash scripts/jetson_full_smoke_run.sh`
   end-to-end with all stages blocking; confirm `power` stage
   `estop_latency_ms` lands well under
   `ESP32Config.emergency_stop_budget_ms` and the motor stage passes under
   its **F-025 re-scoped criterion (command dispatched without error +
   e-stop within budget)** — the chassis is encoder-less (audit R3), so the
   previous non-zero encoder-velocity criterion was unsatisfiable and has
   been retired. Note there is no per-command ACK to wait on: stock
   `General_Driver` firmware streams frames unsolicited and acknowledges
   nothing, so a clean send plus the e-stop budget is the whole observable. Then
   `scripts/validate.py --tier hardware` flips **F-008** to `done`.
4. **Decoupled merge posture** — the PR #106 *code* is merged and verified;
   the live-rover *motion* validation is the hardware-blocked operational
   concern tracked here.

---

## Operator runbooks

- Claude Code on the Jetson (install, service mode, hardening, when NOT to
  use it): `docs/runbooks/claude-code-on-jetson.md`
- Secret scanning + allowlist policy: `docs/runbooks/secret-scanning.md`
- Workforce hooks (edit-time secret scan, capability freeze gate, overrides,
  debugging): `docs/runbooks/claude-workforce-hooks.md`
- Full bring-up (probe-first motors): `docs/runbooks/jetson-full-bringup.md`
- Full validation (cold/warm phases, trend journal): `docs/runbooks/jetson-full-validation.md`

---

## Deferred / Out Of Scope

- **HC-SR04 ultrasonic work**: not part of the active Jetson production baseline until the
  sensor path is ready for real-device validation. (Note: `config/default.yaml` still ships
  a populated `ultrasonic:` block — "parked" is a roadmap status, not a config default;
  the import-graph freeze test pins that no active module imports the driver at module top
  level.)
- **Robot arm platform**: deferred from the current roadmap until the Jetson + replay-loop +
  activation work is complete. See "Training arc" above for the explicit unfreeze condition.
