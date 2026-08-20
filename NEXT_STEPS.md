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
Physical-AI roadmap below (Phase 1 domain randomization → Phase 6 co-training).
The legacy v0.3.0 execution-plan phase numbering lives only in
`docs/planning/NEXT_STEPS.md` and is annotated as such there.

---

## ⚡ Current Next Steps (prioritized)

1. **[Security — P0] Rotate the `ANTHROPIC_API_KEY`.** The key was exposed in a chat
   transcript — treat it as compromised. Inventory consumers (rover `/etc/mousedroid/docker.env`,
   GitHub Actions secrets, dev env), replace on Jetson, restart container, confirm cloud auth
   (`tools/llm_latency_probe.py --iterations 3`), revoke old key. The software half (secret-scan gate)
   is **F-015** (`done`): gitleaks CI job + `.gitleaks.toml` + `docs/runbooks/secret-scanning.md`.
2. **[Hardware blocker — P0] ESP32 diagnosis + repair — diagnostics before spend.** Re-test:
   (a) stock command set (`MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock`, 115 200 baud);
   (b) probe ESP32 `U0TX`/`U0RX` directly on driver board's 40-pin header (pins 10/8 → `/dev/ttyTHS1`)
   to isolate CP2102N bridge from ESP32. Gate for **F-008**. Time-box: 2 bench sessions.
3. **[Software blocker — LANDED] ESP32 driver retargeted at stock Waveshare firmware (F-025).**
   Landed as PR #185 (`f884abe`). Operator action: flip `MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock`
   in `/etc/mousedroid/docker.env` after `deployments/jetson-image.json` is re-pinned.
4. **[Ops hygiene — P1] Re-point the rover's `/opt/mousedroid` source** to trunk. Targeted `sudo chown ian:ian`
   on tracked files to resolve bind-mount root-ownership drift before checkout.
5. **[Durability — P1] Make per-host `docker.env` overrides durable.** **F-017** (`done`):
   `config/docker.env.example`, `host_env_keys` preflight check, and `scripts/host_bootstrap.sh`.
   Run bootstrap on rover after reflash and enable `host_env.enabled` in Jetson overlay.
6. **[Bring-up — P1] Full rover bring-up + unified dashboard.** Run `docs/runbooks/jetson-full-bringup.md`.
   Motors are probe-first; `MOUSEDROID_ESP32__ENABLED=false` prevents crash-looping if unpowered.
7. **[Sensing — P1, post-retarget] Wire onboard IMU + magnetometer (audit R4).** Consume QMI8658 +
   AK09918 via `FEEDBACK_BASE_INFO` as a 6th fusion modality behind default-OFF `Optional` config block.
8. **[Validation — P2] Run consolidated on-device validation pass (PR #116).** Execute
   `bash scripts/jetson_full_validation.sh` on rover with trend journaling (**F-018**).
9. **[Observability — P2] Wire Grafana dashboard + alerts into rover Prometheus.** Panels + alert
   rules landed (**F-019**); import `docs/grafana_dashboard.json` and load `config/prometheus/alerts.yml`.
10. **[Hygiene — P2] Review dead-code audit report** (`scripts/dead_code_audit.py`, **F-020**) and promote
    advisory CI stages when due (`.github/advisory_stages.yaml`).
11. **[Hygiene — P2] Declared budgets need consumers (F-026).** Ensure `JetsonConfig.power_mode` and
    `DocsConfig.core_max_lines` have active consumers with AQA regression tests.
12. **[Docs — P2] Reconcile hardware docs with chassis (audit R9).** WAVE ROVER is 4WD skid-steer;
    chassis is encoder-less, camera is IMX708, power is 3S 18650 UPS.
13. **[World model — P2] F-023 operator follow-ups (AlayaWorld adaptation).** Distillation spike per
    `docs/runbooks/jetson-alayaworld-spike.md` and `scripts/compare_drift.py`.
14. **[Portfolio — P2] Record 60-second hardware demo clip** on Jetson and link in README (host as release asset).
15. **[Portfolio — P2] Git-history purge** post-reframe PR (#167): run `scripts/purge_history.sh` and rename slug to `mouse-droid`.
16. **[CI — P2] Add Windows matrix job to CI.** SQE audit resolved 21 Windows issues (`48a6655`); add `windows-latest` tier for lint, typecheck, and test-fast.

---

## Current Baseline (one-screen)

- **Hardened Autonomous Architecture (2026)** — Factory-first DI (`build_autonomous_camera`, `build_autonomous_lidar`, `build_autonomous_metrics_registry`, `build_motor_controller`, `build_autonomous_orchestrator`), 30 Hz loop with sensor preflight, jitter observability, and cancellation e-stop.
- **Deliberative brain (Claude gateway) LIVE on rover** — Claude-haiku primary + Phi-3-mini fallback (PR #107/#111) with pre-egress prompt-injection filtering.
- **Active production scope**: camera + LiDAR + USB audio + ESP32 on Jetson.
- **7-Tier Test Pyramid & Workforce Governance**: 56 automated tests across Unit, Property, Integration, Functional, E2E, User Journey, Security Fuzzing, Smoke, Regression (100% PASS), 7 specialized subagents (`.claude/agents/`), and 3 reusable skills.
- **Ten Pillars campaign**: 20/20 PASS on Jetson Orin Nano (2026-04-26). Landed-work history in `CHANGELOG.md`.

---

## P0 — Physical AI Roadmap (Phases 2 → 6)

Dependency direction is strictly **Phase 1 → 2 → 3 → 4**; Phases 5 and 6 are deferred until Phase 3b has soaked ≥30 days.

- **Phase 1 — domain randomization** ✅ landed (see CHANGELOG).
- **Phase 2 — real-episode replay loop** ✅ landed incl. Phase 2.1 BC injection.
- **Phase 3a/3b — VLA protocol + DistilledVLAOnnx** ✅ landed.
- **Phase 4 — VLM-derived dense rewards (VLAC)** ✅ landed.
- **Phase 5 (stretch) — real physics simulator** — deferred until Phase-3b 30-day soak completes.
- **Phase 6 (stretch) — real-time co-training** — LoRA-style on-device fine-tuning; builds on Phases 2 + 3.

### Training arc (T-numbers)

**Arm training arc PAUSED at T2.** Unfreeze condition: **F-008** (rover hardware gate) reaches `done` on Jetson AND Phase 3b has soaked ≥30 days. Until then T3+ and arm skills stay frozen. T2 (MLflow training observability) is landed (`docs/runbooks/mlflow-local-ui.md`).

---

## Open engineering follow-ups

0b. **[Architecture — needs a decision] 3 of 5 GCP cloud components are unwired.**
    `CloudLoggingSink`/`CloudMetricsExporter`/`CloudFirestoreSync` have no
    `build_cloud_*()` factory, zero runtime callers — same shape as unwired
    `meta`/`growth`/`scaling` (`docs/architecture.md` Level 3d corrected).
    Needs a wire-or-defer call.

0a. **[Hygiene — needs a dedicated pass] First-ever vulture dead-code audit run: 447
    findings.** `scripts/dead_code_audit.py` (F-020) had been CI-wired but never
    actually run + triaged — `scripts/vulture_allowlist.py` was empty. Ran it
    2026-08-16 at the default 60% confidence, output in
    `reports/dead_code/2026-08-16.json`. 2 spot-checked findings
    were confirmed Protocol/DI false positives (vulture can't trace
    protocol-typed call sites). At this volume, needs a dedicated triage pass
    batched by module, not a rubber-stamp allowlist add or blind deletion.

0. **[Dependency — diagnosed blocker] `mlflow-skinny` 3.x breaks the file-store tracking
   default.** Open dependabot PR #145 proposes widening `mlflow-skinny` from `<3,>=2.22` to
   `<4,>=2.22`. No CI job installs the `[mlflow]` extra, so its "green" status never actually
   exercised mlflow. Manually ran the mlflow-touching test surface against 3.15.1:
   **29 failed, 17 errored** (of ~74) — mlflow 3.x hard-rejects the local filesystem
   tracking backend by default, which `MlflowExperimentLogger`
   (`src/mousedroid/training/observability/mlflow_logger.py`) relies on via its default
   tracking URI. Needs a decision before merging: set `MLFLOW_ALLOW_FILE_STORE=true` in
   the training entrypoints/CI, or migrate to a database tracking URI
   (`sqlite:///mlflow.db`). PR left open pending that decision.

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

5. **Finish the Claude workforce bundle (F-024).** Phases 1–2 landed (config,
   three hooks, AQA gate, CI stages). Still open, tracked in
   `openspec/changes/mouse-droid-claude-workforce/tasks.md`: the subagent roster
   (`.claude/agents/`), five new skills, a secretless `.mcp.json` including the
   repo's own MCP server per `docs/MCP_OPERATOR_GUIDE.md`, and the CLAUDE.md
   restructure (which also makes `DocsConfig.core_max_lines` real — see item 11
   above). `performance` / `security` keep their own advisory windows in
   `.github/advisory_stages.yaml`. Still open: decide whether repo-wide branch
   coverage should be measured (today only `tools/claude_hooks/` is, and
   advisory).

### Pending follow-up (deferred to a separate PR)

- **importlib helper consolidation** — partially closed by
  `tests/_script_loader.py`; sweep the remaining inline
  `spec_from_file_location` call sites onto it.
- **Scripted WAN-drop failover drill** — capture the operator drill asserting
  `fallback_primary_to_secondary` + `fallback_primary_retry_attempt` once the
  ESP32 is repaired and a full end-to-end mission can run.
- **[Cognitive integration — F-022] Soak-gate the growth pillar before enabling it.**
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
