# MouseDroid — Next Steps

Forward-looking priorities only. Landed work moves to `CHANGELOG.md` (see the
"Historical record" section there); the machine-readable source of "what's
next" is the feature catalog (`features.yaml` + `python scripts/select_next.py`
— Golden Rule per `HARNESS_SPEC.md`). Advisory size guard:
`tools/doc_hygiene.py NEXT_STEPS.md`.

**Phase vocabulary:** "Phase N" in this file refers exclusively to the
Physical-AI roadmap below (Phase 1 domain randomization → Phase 6 co-training).
The legacy v0.3.0 execution-plan phase numbering lives only in
`docs/planning/NEXT_STEPS.md` and is annotated as such there.

---

## ⚡ Current Next Steps (prioritized)

1. **[Security — P0] Rotate the `ANTHROPIC_API_KEY`.** The key was exposed in a chat
   transcript — treat it as compromised. Inventory consumers first (rover
   `/etc/mousedroid/docker.env`, GitHub Actions secrets, any dev-machine env), replace on
   the Jetson, restart the container (`sudo systemctl restart mousedroid-docker`), confirm
   the cloud tier authenticates (`tools/llm_latency_probe.py --iterations 3`), revoke the
   old key, and record the rotation date + verification here. The software half of this
   item (secret-scan gate) is **F-015** (`done`): gitleaks CI job + `.gitleaks.toml` +
   `docs/runbooks/secret-scanning.md`.
2. **[Hardware blocker — P0] ESP32 repair.** The rover's ESP32 is functionally dead, so the
   NL→GoalVector path is validated *up to the GoalVector* but the GoalVector→wheels leg cannot
   be exercised end-to-end. This is the top blocker for true autonomous navigation — and the
   gate for **F-008**, the only `todo` catalog feature. See the sequenced bench-repair +
   reflash plan under **PR #106 follow-ups** below. Time-box: 2 bench sessions, then a
   repair-vs-replace decision (replacement Wave Rover driver board identified up front).
3. **[Ops hygiene — P1] Re-point the rover's `/opt/mousedroid` source** to trunk
   (`claude/markdown-implementation-plan-aVJ2l`). Blocked by pre-existing root-ownership drift
   in the bind-mount (the container writes files as root), so a targeted
   `sudo chown ian:ian` of the tracked files is required first before the checkout will succeed.
4. **[Durability — P1] Make the per-host `docker.env` overrides durable.** Software half is
   **F-017** (`done`): `config/docker.env.example` now enumerates
   `MOUSEDROID_LLM__ENABLED` / `MOUSEDROID_LLM__N_GPU_LAYERS`, the `host_env_keys` preflight
   check WARNs on key drift, and `scripts/host_bootstrap.sh` (dry-run/backup/rollback-safe)
   installs the template + units. Remaining operator action: run
   `scripts/host_bootstrap.sh` on the rover after the next reflash and enable
   `host_env.enabled` in the Jetson overlay.
5. **[Validation — P2] Run the consolidated on-device validation pass (PR #116).** Execute
   `bash scripts/jetson_full_validation.sh` on the rover — now with trend journaling
   (**F-018**): Phase-2 preflight appends to the trend journal and the Phase-4 SUMMARY
   carries a Trend section. **Optional unblock for HTTP-driven `/metrics` population:**
   add an `openclaw:` block with `enabled: true` to a validation overlay (the prod config has
   none, so `POST /api/v1/mission` is unregistered and Test C skips).
6. **[Bring-up — P1] Full rover bring-up + unified dashboard.** Run
   `docs/runbooks/jetson-full-bringup.md` on the rover. **Real motors are probe-first** —
   the ESP32 is probed before bring-up and motors only go live if it responds; otherwise
   `MOUSEDROID_ESP32__ENABLED=false` keeps the container from crash-looping. ESP32 physical
   repair (item 2) remains the gating item for actual motion.
7. **[Observability — P2] Wire the Grafana dashboard + alerts into the rover's Prometheus.**
   The panels + alert rules for the LLM-gateway families are landed (**F-019**); the operator
   action is importing `docs/grafana_dashboard.json` and loading
   `config/prometheus/alerts.yml` on the monitoring host before the next endurance run.
8. **[Hygiene — P2] Review the dead-code audit report** (`scripts/dead_code_audit.py`,
   **F-020**) after each significant merge; promote the advisory `gitleaks` /
   `vulture-audit` CI stages when `scripts/check_advisory_promotions.py` flags them due.
9. **[World model — P2] F-023 operator follow-ups (AlayaWorld adaptation).** The
   bounded-context latent memory + corrupted-history drift training landed default-OFF
   (`world_model_memory` / `training.drift` blocks; ADR-015). Remaining operator actions:
   (a) run the distillation spike on the Jetson per
   `docs/runbooks/jetson-alayaworld-spike.md` and paste the numbers into
   `docs/analysis/alayaworld-distillation-spike.md` for the final go/no-go; (b) re-run
   `scripts/compare_drift.py` against real replay data once the rover accumulates enough
   records; (c) keep `world_model_memory.enabled` OFF on the live rover until a soak gate
   passes.
9. **[Portfolio — P2] Record the 60-second hardware demo clip** (droid navigating +
   avoiding obstacles on the Jetson) and drop it into the README `## ▶ Demo` slot — hosted as
   a `hardware-v6`-style GitHub Release asset or external link, **never** committed (that
   re-creates the bloat the reframe just removed). This clip is the headline portfolio artifact.
10. **[Portfolio — P2] Run the git-history purge** once the reframe PR (#167) merges:
    preserve the blobs first (`bdi_annotations.npz` → HF dataset, CAD → `hardware-v6` Release),
    then `bash scripts/purge_history.sh` (dry-run) → `--push`. Shrinks `.git` ~28 MB → ~2 MB.
    Destructive + irreversible — see `docs/runbooks/history-purge.md`. Also rename the GitHub
    repo slug `Mouse-Droid-AGI` → `mouse-droid` (Settings; URLs auto-redirect).

---

## Current Baseline (one-screen)

- **Deliberative brain (Claude gateway) is LIVE on the rover** — Claude-haiku primary +
  Phi-3-mini CPU fallback (PR #107/#111). The 30 Hz reactive loop stays LLM-free.
- **Active production scope**: camera + LiDAR + USB audio + ESP32 on Jetson. The HC-SR04
  ultrasonic path is parked, and the robot-arm platform is deferred from the active delivery
  plan.
- **Ten Pillars campaign**: 20/20 PASS on Jetson Orin Nano (2026-04-26). Full landed-work
  history: `CHANGELOG.md`.

---

## P0 — Physical AI Roadmap (Phases 2 → 6)

Dependency direction is strictly **Phase 1 → 2 → 3 → 4**; Phases 5 and 6 are
deferred until Phase 3b has been in production for ≥30 days (clock anchored to
the deployed SHA's continuous uptime — it restarts on redeploy).

- **Phase 1 — domain randomization** ✅ landed (see CHANGELOG).
- **Phase 2 — real-episode replay loop** ✅ landed incl. Phase 2.1 BC injection
  (byte-identity at `weight=0` proven).
- **Phase 3a/3b — VLA protocol + DistilledVLAOnnx** ✅ landed.
- **Phase 4 — VLM-derived dense rewards (VLAC)** ✅ landed (Law-1 gate preserved).
- **Phase 5 (stretch) — real physics simulator** — notes-only until the Phase-3b
  30-day production soak completes.
- **Phase 6 (stretch) — real-time co-training** — LoRA-style on-device fine-tuning;
  builds on Phases 2 + 3.

### Training arc (T-numbers)

**Arm training arc PAUSED at T2.** Unfreeze condition: **F-008** (rover hardware
gate) reaches `done` on the Jetson AND Phase 3b has soaked in production ≥30
days. Until then T3+ (`train_arm.py` SAC+HER) and the arm skills stay frozen
(`.claude/skills/*` carry `status: frozen`). T2 (MLflow training observability)
is landed — runbook: `docs/runbooks/mlflow-local-ui.md`.

---

## Open engineering follow-ups

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
   restructure. Also open: promote the `gitleaks` CI job from advisory to
   blocking once the 7-green-run tracker clears, and decide whether repo-wide
   branch coverage should be measured (today only `tools/claude_hooks/` is, and
   advisory).

### Pending follow-up (deferred to a separate PR)

- **Resilience wrappers for camera + voice + LLM gateway** — drop the three
  remaining bare driver constructions in `factory.py` behind the existing
  `CircuitBreaker` + `retry_async` machinery (per-driver opt-in via a new
  `cfg.resilience.<driver>.enabled` flag, defaults `False`). ESP32 and
  LiDAR are already wrapped (`src/mousedroid/resilience/`); these three
  are the residual gap.
- **`set -e` on `scripts/jetson_full_smoke_run.sh`** (review-agent low finding) —
  the wrapper currently uses `set -uo pipefail` but not `-e`; inner stage
  logic tracks `OVERALL_FAIL` correctly so this is intentional, but
  top-level scripting errors silently continue. Align with `jetson_smoke_test.sh`.
- **importlib helper consolidation** — partially closed by
  `tests/_script_loader.py`; sweep the remaining inline
  `spec_from_file_location` call sites onto it.
- **SHA-pin GitHub action references** (CodeRabbit PR #106 finding 4) —
  best done as a single sweep with Dependabot configured to auto-bump the SHAs.
- **`llm_gateway/__init__.py` lazy-import hardening** (PR #107 round-3 Low
  finding) — the package eager-imports `AnthropicLLMGateway` +
  `FallbackLLMGateway`; per CLAUDE.md invariant 1 these should leave the
  package surface or move under `TYPE_CHECKING`.
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
perform remotely. Sequenced follow-ups:

1. **Bench-side hardware repair** — multimeter continuity probe ESP32
   UART0 TX → CP2102N RXD on the canonical USB-C port; visual inspect
   for damaged traces / lifted pads near the BOOT button (most likely
   stress point from the 2026-05-31 BOOT-button-during-power-cycle
   diagnostic). Worst case: replace the ESP32 module / Wave Rover
   driver PCB. **Time-box: 2 bench sessions, then decide repair vs
   replace.** Record the flashed firmware binary (path + hash) alongside
   F-008's `implemented_in` when the gate closes.
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
   non-zero encoder velocity. Then `scripts/validate.py --tier hardware`
   flips **F-008** to `done`.
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
