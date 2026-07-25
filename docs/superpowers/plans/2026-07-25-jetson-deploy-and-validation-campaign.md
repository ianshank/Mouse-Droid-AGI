# Jetson Deploy Prep + Full Deployment & Validation Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Jetson deployment surface *truthful and safe* in the repo (WS-A, landed with
this plan), then deploy the rover from the pushed branch (WS-B) and run the complete on-device
validation campaign across every component with explicit pass/fail gates (WS-C) — without ever
arming motion while the ESP32 is dead.

**Architecture:** Three workstreams on one axis. **WS-A (repo, landed here)** fixes the drifted
deploy surface: the `PYTHONOPTIMIZE=1` contract that a dozen documents asserted but the image
never set; a production overlay that left a dead ESP32 enabled (container crash-loop); a
dev-tools build-arg mismatch that let a lean image silently skip 4 of 10 pillar checks; a
systemd unit with two fatal misconfigurations; and stale image-pin / secret-template hygiene.
**WS-B (operator)** deploys via the git-based Docker spine (bind-mounted `/opt/mousedroid`),
NOT `deploy_remote.sh` (its `rsync --delete` destroys rover work) and NOT `docker_deploy.sh`
(forces `--no-cache`). **WS-C (operator)** composes the *existing* harness —
`scripts/jetson_full_validation.sh` phases 0-4, `jetson_smoke_test.sh` stages,
`mousedroid.cli.{preflight,validate_pillars}`, the `tools/*_probe.py` family, the trend timer
and the ten-pillars nightly — into a gated campaign. Nothing new is built for validation; the
gaps closed are *gates and evidence*, not tooling.

**Tech Stack:** Jetson Orin Nano 8 GB (JetPack 6 / L4T r36.4) • Docker + compose (base
`dustynv/l4t-pytorch:r36.4.0`) • Waveshare Wave Rover chassis + ESP32 driver board (CP2102N
bridge) • LDROBOT LD19 lidar (CP2102, 230400) • Arducam IMX708 CSI • SSD1306 OLED (i2c-7) •
USB audio + Piper TTS • ruff 0.8.0 • mypy --strict • pytest (unit/property/integration/
regression/e2e/smoke/performance/hardware tiers) • structlog • Prometheus/Grafana/Loki.

## Peer-review corrections (verified against the repo, 2026-07-25)

This plan was reviewed through five independent lenses (technical correctness, operational
completeness, governance conformance, in-repo hardware docs, vendor docs). Load-bearing
corrections that changed the design:

1. **A naive "fail on any pillar SKIP" gate would false-fail every run.** `memory` and
   `curiosity` legitimately SKIP on the production overlay (`MemoryConfig.enabled` defaults
   False; the overlay declares no `memory:` block), and `--dry-run` marks all ten SKIPPED.
   → The gate discriminates on a new `PillarResult.skip_reason`; only `environment` skips fail.
2. **The Phase-2 pillar step runs in the *host venv*, not the image**, and compose already
   defaulted `INSTALL_DEV_TOOLS=true`. The Dockerfile ARG flip is drift-consistency; the
   strict-skips gate is what actually closes the hole.
3. **`ENV PYTHONOPTIMIZE=1` leaks into every `docker exec`**, including Phase-1's in-container
   `ci.sh` — a suite that has never run under `-O`. → Both in-container CI invocations now pass
   `-e PYTHONOPTIMIZE=0`.
4. **Re-pinning `deployments/jetson-image.json` to a feature-branch SHA recreates the exact
   `9c31968` failure** the record's own notes document (squash-merge orphans the commit and the
   `config-compat` gate dies repo-wide). → The re-pin is a **post-merge** step (B5).
5. **`mousedroid-docker.service` had a second fatal blocker** beyond `Type=notify`:
   `ExecStartPre=… docker compose pull` without a `-` prefix, against a local-only image.
6. **Commenting out `MOUSEDROID_TELEMETRY_TOKEN` would have *weakened* a security check** —
   `_parse_env_keys` skips `#` lines, so the `host_env_keys` drift WARN would stop firing on
   hosts missing the token, and `host_bootstrap.sh` would seed fresh hosts without it.
   → Empty value, key retained.
7. **`esp32.enabled: false` is schema-compatible with the pinned image** — verified
   `ESP32Config.enabled` (and `OpenClawConfig`) exist at the `deployments/jetson-image.json`
   SHA, so the `config-compat` gate accepts the edit.
8. **The face display IS validated** (dedicated `oled` stage in `jetson_full_smoke_run.sh` +
   `test_ssd1306_smoke`/`test_face_display_smoke` in the hardware tier); only the
   full-validation smoke *loop* lacks an `oled` entry. An earlier draft wrongly called it
   unvalidated.
9. **Vendor stock ESP32 firmware speaks 115200** (`waveshareteam/ugv_base_general`,
   `{"T":1,"L":…,"R":…}`) while this repo assumes custom firmware at 1,000,000 baud with
   `vx/vy/omega`. A stock reflash during repair would make a healthy board read as dead.
10. **A firmware-dead ESP32 still enumerating is expected**, not lucky: the CP2102N bridge is
    USB-bus-powered and independent of ESP32 firmware health.

## Context (why this plan exists)

The validation harness is mature, but the *deployment* surface around it had drifted until it
no longer described reality: documents asserted a runtime flag the image never set, the
production config would crash-loop on the rover's known-dead motor controller, and the systemd
unit could not start at all. Meanwhile the campaign itself lacked executable gates — "run the
validation script" is not a pass/fail contract, and several components (monitoring stack,
dashboard WebSockets, endurance) were never explicitly covered. This plan closes both halves:
the repo tells the truth about how it deploys, and the operator has a gated, evidence-producing
campaign.

---

## WS-A — Repo prep (landed with this plan)

- [x] **A2 — ESP32 safe-by-default.** `config/jetson_production.yaml` gains `esp32.enabled:
      false` (schema default stays `True`). Repair lever is
      `MOUSEDROID_ESP32__ENABLED=true` in `/etc/mousedroid/docker.env` — env beats YAML, so
      going live needs no commit. Slot documented in `config/docker.env.example`; probe-first
      flow rewritten in `docs/runbooks/jetson-full-bringup.md`.
- [x] **A3 — `ENV PYTHONOPTIMIZE=1`** in `Dockerfile.jetson`, with `-e PYTHONOPTIMIZE=0` on
      both in-container `ci.sh` execs in `scripts/jetson_full_validation.sh`.
- [x] **A4 — Strict pillar-skip gate.** `PillarResult.skip_reason`
      (`config_disabled|dry_run|environment`, additive, default `None`); `--strict-skips` on
      `mousedroid.cli.validate_pillars` (fails only on `environment`, rejects `--dry-run`);
      env knob `MOUSEDROID_VALIDATION_PILLARS_STRICT_SKIPS` (default 1);
      `ARG INSTALL_DEV_TOOLS=true` aligned with compose.
- [x] **A5 — Config resolution matrix** in `docs/runbooks/jetson-full-validation.md`;
      `scripts/mousedroid.service` annotated as the legacy bare-metal path.
- [x] **A6 — systemd unit fixed:** `Type=exec`, `WatchdogSec` dropped, compose-`pull`
      ExecStartPre made non-fatal.
- [x] **A7/A1 — hygiene:** telemetry-token placeholder removed (key retained, empty value +
      `openssl rand -hex 32` recipe); stale `set -e` item deleted from `NEXT_STEPS.md`; image-pin
      authority clarified in `docs/runbooks/jetson-claude-pilot-deploy.md`.

**Deferred (explicitly not in this PR):** Piper `scout`/`friendly` voice staging (build-time HF
dependency + ~120 MB on a disk-pressured rebuild; pair with the queued
`scripts/benchmark_voice_latency.py` run); `nvidia-l4t-gstreamer` for a proper in-container
`nvarguscamerasrc` path (until then the Bayer/green-channel workaround is the *intended*
camera evidence); a validation-only `openclaw:` overlay for HTTP-driven `/metrics`; CI-built
arm64 images (single rover — on-device build is the working precedent).

**Governance actions for the maintainer** (not executable from a session): confirm whether a
`main` branch exists on `origin` and, if not, create it at the trunk tip so
`scripts/check_config_compat.py`'s documented `--base-ref origin/main` is truthful and the
image pin has a durably reachable anchor; consider promoting the ten-pillars nightly to a
required check (the workflow says it is promotion-ready but branch protection is unset).

---

## WS-B — On-device deployment (operator)

- [ ] **B0 — Preconditions.** `ssh jetson` reachable; rover reaches `api.anthropic.com`;
      self-hosted runner service healthy (note the user split — runner installs default to
      `jetson`, bring-up docs assume `ian`); `/etc/mousedroid/docker.env` present, `ian:ian`,
      `600`. **Disk guard:** `df -h /` + `docker system df -v`, require ≥8 GiB free, reclaim
      with `scripts/jetson_disk_cleanup.sh`. **Memory guard:** record `swapon --show` +
      `zramctl`; NVMe swap active before any build (vendor guidance for 8 GB Orin Nano). Record
      where Prometheus/Grafana/Loki run — host-side services compete with the container's 6 G
      limit on 7.4 GB usable RAM.
- [ ] **B1 — Preserve rover WIP.** Commit rover-local state to `rover/wip-20260725`; archive a
      whitespace-insensitive diff off-device. Never `git clean`, never rsync-delete.
- [ ] **B2 — Sync source.** **First** repair the known root-ownership drift (targeted
      `sudo chown ian:ian` of tracked files under `/opt/mousedroid`; confirm `git status` runs
      clean as the operator), then fetch + checkout the branch tip. Git-bundle contingency if
      rover GitHub auth has lapsed.
- [ ] **B3 — Overlay + secrets.** `scripts/sync_jetson_overlay.sh`, then `--verify`.
      **Key rotation (P0), in this order:** inventory consumers (rover `docker.env`, GitHub
      Actions secrets, dev machines) → mint the new key → replace on rover → restart → verify
      with `tools/llm_latency_probe.py --iterations 3` → **then** revoke the old key → record
      the rotation date. Regenerate `MOUSEDROID_TELEMETRY_TOKEN`. Audit `docker.env` for stale
      `MOUSEDROID_ESP32__ENABLED=true` (would override the new safe default) and
      `MOUSEDROID_INSTALL_DEV_TOOLS=false` (would silently build a lean image). Presence-check
      secrets only — never echo.
- [ ] **B4 — Rebuild (mandatory: the new ENV busts cache from that layer).** Pre-tag the
      rollback anchor `docker tag mousedroid:jetson mousedroid:jetson-rollback-20260725`, then
      `docker compose -f docker-compose.jetson.yml build` (cached) and `up -d --force-recreate`.
      ENOSPC recovery: prune build cache → retag rollback → `up -d` on the old image.
- [ ] **B5 — Image pin (POST-MERGE ONLY).** Leave `deployments/jetson-image.json` untouched
      during the campaign. After the PR merges, re-pin to the squash/merge **trunk** commit
      (+ `deployed_at`, notes recording the PYTHONOPTIMIZE / dev-tools deltas) as a follow-up
      commit. **Do not delete the feature branch before that lands.** Never pin a
      feature-branch SHA.
- [ ] **B6 — Health.** Bounded poll of `/api/v1/health` (fixed retries × interval → a defined
      timeout, not operator judgement); container healthy; logs show `MockESP32Driver`
      resolution, no crash-loop, and `fallback_gateway_started` with `primary_ready` +
      `secondary_ready`. If behaviour differs post-deploy, first-line triage is recreate with
      `-e PYTHONOPTIMIZE=0` to isolate A3 *before* any image rollback.
- [ ] **B7 — Rollback.** `down` → retag `mousedroid:jetson-rollback-20260725` →
      `git checkout <previous SHA>` → `up -d`. If `config-compat` ever rejects a YAML edit, fix
      the YAML against the pinned schema — never re-pin the record to dodge the gate.
- [ ] **B8 — Handoff.** Record `DEPLOY_SHA`, `ROLLBACK_SHA`, run stamp, `nvpmodel -q` output and
      `jetson_clocks` state into `env.log` under the report root. (Record the *query output* —
      nvpmodel mode numbers are not stable across JetPack reflashes.)

---

## WS-C — Validation campaign (operator)

**Hosted CI owns** lint/mypy/unit/property/integration/coverage/regression/e2e/smoke,
`config-validate`, `config-compat`, `actionlint`, Docker lint. **The rover owns** everything
below. **No motion is armed anywhere in C0–C2.5.**

- [ ] **C0 — Pre-flight invariants.** Branch CI green; B6 green; key rotation complete
      *including old-key revocation and probe verification*; ESP32 resolves disabled.
- [ ] **C1 — Full validation:** `bash scripts/jetson_full_validation.sh` (all phases), with a
      live-monitor second terminal.

  | Signal | Expectation |
  |---|---|
  | Phase 0 preconditions | PASS required |
  | Phase 1 static CI | PASS; WARN acceptable **only** via the documented OOM slim-retry (recorded, never silent) |
  | Phase 2 preflight | FAIL on any non-ESP32-family check |
  | Phase 2 `smoke:serial` / `motor` / `power` | WARN, non-blocking (dead board) |
  | Phase 2 `smoke:usbc` | **Blocking** — enumeration must hold |
  | Phase 2 `smoke:hailo` | SKIP (disabled in prod overlay) — cannot fail |
  | Phase 2 `smoke:gpio` | SKIP-clean (HC-SR04 parked, no `ultrasonic:` block) |
  | Phase 2 hardware pytest | PASS under `MOUSEDROID_ESP32__ENABLED=false` |
  | Phase 2 pillars | 10/10 with strict-skips ON; an `environment` skip = FAIL (lean host venv) |
  | Phase 3 | health + `translate_mission` + live `/metrics` + LiDAR WS probe + structlog greps all PASS |
  | Phase 4 | exit 0; `SUMMARY.md` archived |
  | **Must be ABSENT** | any `on_device_*` or growth-distillation event — Phase 6 and growth are default-OFF; an occurrence means config drift → stop and triage |

  **Abort/triage criteria:** rc=137 → never loop-retry (that is the OOM guard's job); camera
  FAIL → restart `nvargus-daemon`, re-run `--phases 2` (the rover carries an **IMX708** —
  ignore the stale IMX500 doc trail); `smoke:audio` FAIL with mic-enum symptoms → reseat/
  re-enumerate USB audio and re-run the single stage before calling it a campaign FAIL (the
  harness itself distrusts mic enumeration — Phase-2 preflight runs with
  `MOUSEDROID_MICROPHONE__ENABLED=false`); missing `SUMMARY.md` → `docker start` FIRST;
  WAN dropout → re-run `--phases 3`. Trend first-run semantics: with fewer than two journal
  entries, `trend_rc=0` means *baseline established*, not PASS.

- [ ] **C1.5 — Dashboard + monitoring.** Run `scripts/jetson_probe_dashboard_e2e.py`,
      `jetson_probe_logs_ws.py`, `jetson_probe_ws_negotiation.py`,
      `jetson_probe_lidar_raw_ws.py`; browser check through `tools/dashboard_proxy.py` across
      all three transports (plain / MJPEG / WebSocket), confirming the bearer token is never
      browser-visible. Monitoring stack: import `docs/grafana_dashboard.json`, load
      `config/prometheus/alerts.yml`, assert the Prometheus target is `up` after
      `--force-recreate` (container identity changes), the four `{ns}_llm_*` families are
      visible after C1's translate probe, and Loki returns `anthropic_gateway_*` lines for the
      run window.
- [ ] **C2 — Trend + nightly + host wiring.** `scripts/host_bootstrap.sh --with-trend-timer`
      (the sanctioned installer); enable `host_env.enabled` in the overlay; confirm the trend
      timer keeps its pinned contract (non-exclusive `config,host_env_keys` checks, separate
      journal — never the full-run journal, never camera/lidar/esp32/audio); confirm the
      self-hosted runner is healthy and the ten-pillars nightly is green against the new image.
- [ ] **C2.5 — Endurance (pre-soak baseline; no motion required).** Must be invoked
      **explicitly** with `MOUSEDROID_ENDURANCE_FORCE_REAL=1` (+ duration env) — the Phase-2
      hardware pytest tier collects the endurance module but it mocks/skips without that flag,
      so "hardware pytest green" does **not** imply endurance coverage. Capture a background
      `tegrastats --interval` log into the report root: Orin throttles at TDP, and an unlogged
      throttle silently invalidates the latency baselines this run seeds.
- [ ] **C3 — F-008 closure (conditional: only if the ESP32 is repaired).** Time-boxed bench
      repair → probe-first bring-up → `MOUSEDROID_ESP32__ENABLED=true` in `docker.env` →
      `assert_power_chain` → smoke with serial/motor/power now **blocking** → re-run C1 with the
      board live → `scripts/validate.py --tier hardware` green → set F-008 `done` +
      `implemented_in` in `features.yaml` (the `src/mousedroid/arm/**` freeze gate self-disables
      by design) → endurance re-run with motion, rover lifted.

      **Firmware↔baud provenance is a gate step.** Record the flashed `.bin` path + hash and its
      protocol pairing: custom `waverover_mousedroid.bin` = 1,000,000 baud + `vx/vy/omega` JSON
      (what this repo's config assumes) vs Waveshare stock `ugv_base_general` = **115200** +
      `{"T":1,"L":…,"R":…}`. If stock firmware is flashed, reconcile `esp32.serial_baud` and the
      command schema **before** `assert_power_chain` — otherwise a healthy board reads as still
      dead, and "wrong baud after reflash" belongs in the triage list. Flash offset `0x0` is
      correct only for merged images (app-only binaries go at `0x10000`). If the driver board is
      swapped, re-check the USB bridge product string before trusting the `usbc_discovery` by-id
      globs — a classic-CP2102 replacement makes the rover glob miss and the lidar glob
      ambiguous.
- [ ] **C4 — 30-day soak.** Clock anchored to the deployed SHA's continuous uptime; restarts on
      every redeploy. On-device learning, growth distillation and `world_model_memory` stay
      default-OFF until it passes.

**Evidence capture (every gate):** the stamped report tree under the report root, a console
`tee` log, and `env.log`, mirrored off-device via `scp`. The artifact set *is* the campaign
deliverable.

## Component coverage appendix

| Component | Where validated |
|---|---|
| ESP32 / motors | `smoke:serial|motor|power` (WARN, non-blocking); motion C3-gated |
| USB-C enumeration | `smoke:usbc` (blocking); `tests/hardware/test_usbc_enumeration.py` |
| Camera (IMX708 CSI) | `smoke:camera`, `verify_sensors --sensor camera`, dashboard MJPEG; Bayer workaround is intended evidence until `nvidia-l4t-gstreamer` lands |
| LiDAR LD19 | `smoke:lidar`, `tools/lidar_telemetry_probe.py`, `jetson_probe_lidar_raw_ws.py` |
| Microphone / speaker / voice | `smoke:audio|speaker|voice`; mic-enum flake has an explicit re-run path |
| Face display (SSD1306) | hardware tier (`test_ssd1306_smoke`, `test_face_display_smoke`) + the `oled` stage in `jetson_full_smoke_run.sh`. *Follow-on:* the full-validation smoke loop has no `oled` entry |
| PCIe NVMe SSD | `smoke:pcie_ssd` (blocking) |
| Hailo-8 | `smoke:hailo` records SKIP — disabled in the prod overlay; enable via `config/jetson_hailo.yaml` only once the M.2 card + HEFs are staged |
| GPIO / HC-SR04 ultrasonic | Parked — no `ultrasonic:` block, so `smoke:gpio` SKIPs cleanly |
| LLM gateway (cloud + local) | `scripts/translate_mission.py`, `tools/llm_latency_probe.py`, `{ns}_llm_*` metric families |
| Telemetry / dashboard | C1.5 probes + `tools/dashboard_proxy.py` |
| Monitoring (Prometheus/Grafana/Loki) | C1.5 |
| Ten cognitive pillars | `validate_pillars` (Phase 2, strict-skips) + the nightly campaign |
| GCP tier | **Out of scope** — no cloud block in the production overlay |
