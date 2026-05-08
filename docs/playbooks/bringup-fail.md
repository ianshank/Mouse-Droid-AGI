# Full-Rover Bringup Failure Playbook

Use this playbook when the rover fails to come up cleanly after
`sudo systemctl restart mousedroid-docker.service` — preflight rejects
the start, the orchestrator never reaches its first tick, or the smoke
harness reports stages failed before the per-component playbooks become
relevant.

## What This Covers

- `mousedroid-docker.service` `Active: failed (Result: exit-code)`
- `scripts/preflight_check.sh` aborting before container start
- Container starts but `/health` never returns 200
- First-tick safety check fails (`is_emergency=True` on tick 1)
- `scripts/jetson_full_smoke_run.sh` reporting multiple stages red
- `scripts/validate_pillar.sh all` reporting blocking pillars FAIL

## First Checks

1. Confirm the systemd service state + recent events:
   ```bash
   sudo systemctl status mousedroid-docker.service
   sudo journalctl -u mousedroid-docker.service -n 200 --no-pager
   ```
2. Inspect the container's last lines (rover logs, JSON-formatted):
   ```bash
   docker logs mousedroid --tail 200 --since 5m
   ```
   Look for `mousedroid_starting`, `health_check_result`,
   `safety_emergency`, or any structured `error`-level events.
3. Run the preflight check manually:
   ```bash
   bash /opt/mousedroid/scripts/preflight_check.sh && echo OK
   ```
   Failures point at a specific dependency (camera, ESP32, GPIO, disk).
4. Run the per-sensor verification outside the smoke harness:
   ```bash
   docker exec mousedroid python3 scripts/verify_sensors.py
   ```
   Each sensor reports independently — narrow the failing modality.
5. Run a focused subset of the Ten Pillars (foundational + reward):
   ```bash
   bash scripts/validate_pillar.sh safety world_model memory cognitive reward
   ```
   All five default to blocking. Any FAIL aborts smoke.

## Remediation Steps

1. **`Active: failed` on the service**: read the `journalctl` output
   above. Most common causes:
   - Missing config overlay (`/etc/mousedroid/jetson_production.yaml`)
     — `ExecStartPre` runs `sync_jetson_overlay.sh`; check whether the
     repo at `/opt/mousedroid` is on the right commit.
   - Missing device passthrough — `docker-compose.jetson.yml` declares
     `/dev/serial/by-id/...`, `/dev/i2c-7`, `/dev/snd`, GPU. If any are
     absent on the host, compose refuses to start the container.
   - Pull failure on a dirty image cache —
     `docker compose -f docker-compose.jetson.yml pull` manually to see
     the underlying error.
2. **Preflight aborts before container start**: the script checks
   /dev presence, disk space, and config validity. The output line
   names which check failed. Most failures route to a per-component
   playbook:
   - ESP32 / serial → [`esp32-fail.md`](esp32-fail.md)
   - Camera → [`camera-fail.md`](camera-fail.md)
   - LiDAR → [`lidar-fail.md`](lidar-fail.md)
   - GPIO / OLED → [`gpio-fail.md`](gpio-fail.md)
3. **Container starts but `/health` 503s**: the orchestrator threw
   during component construction. `docker logs mousedroid | head -100`
   shows the first error (usually a config validation error or a
   missing weights file). Fix the YAML or stage the missing weights
   under `/opt/mousedroid/weights/`.
4. **First-tick `is_emergency=True`**: typically a stale ultrasonic /
   LiDAR reading or a low battery. The `safety_context` event line
   shows which signal tripped. With a known-clean environment, restart
   the container; if persistent, check `cfg.safety.*` thresholds in
   [`config/jetson_production.yaml`](../../config/jetson_production.yaml).
5. **Multiple smoke stages red**: read
   `reports/jetson_smoke/<stamp>/SUMMARY.md` (the harness writes one
   per run). Fix from the bottom up — `container_health` and
   `app_health` MUST be green before any sensor stage can run. If
   container_health is red, drop back to step 1.
6. **Blocking pillars FAIL on `validate_pillar.sh`**: each pillar
   writes a per-stage log to `${REPORT_DIR}/pillar_<name>_<kind>.log`.
   `safety` / `world_model` / `memory` / `cognitive` / `reward` default
   blocking; the rest are advisory. Operator can override per-pillar
   blocking with `MOUSEDROID_PILLAR_BLOCKING_<UPPER>=no` for a single
   triage run.
7. **Watchdog kills the service after 30 s**: `Type=notify` +
   `WatchdogSec=30s` means the orchestrator must call `WATCHDOG=1`
   (sdnotify) or touch the heartbeat file before timeout. If the rover
   hangs on first-tick component construction, watchdog fires.
   Temporarily disable with `cfg.loop.watchdog_enabled: false` for
   diagnostics — re-enable before going back to production.

## Cross-Reference

- [`scripts/mousedroid-docker.service`](../../scripts/mousedroid-docker.service) — systemd unit; `ExecStartPre` chain.
- [`scripts/preflight_check.sh`](../../scripts/preflight_check.sh), [`scripts/sync_jetson_overlay.sh`](../../scripts/sync_jetson_overlay.sh), [`scripts/verify_sensors.py`](../../scripts/verify_sensors.py).
- [`scripts/jetson_full_smoke_run.sh`](../../scripts/jetson_full_smoke_run.sh) — full smoke harness.
- [`scripts/validate_pillar.sh`](../../scripts/validate_pillar.sh) — Ten Pillars dispatcher.
- [`docker-compose.jetson.yml`](../../docker-compose.jetson.yml) — device passthrough + bind mounts.
- [`config/jetson_production.yaml`](../../config/jetson_production.yaml) — production overlay.
- [`docs/playbooks/camera-fail.md`](camera-fail.md), [`lidar-fail.md`](lidar-fail.md), [`voice-fail.md`](voice-fail.md), [`esp32-fail.md`](esp32-fail.md), [`gpio-fail.md`](gpio-fail.md), [`replay-fail.md`](replay-fail.md), [`promtool-install.md`](promtool-install.md) — per-component recovery.
- [`docs/jetson-runner-setup.md`](../jetson-runner-setup.md) — self-hosted runner registration so the bringup smoke runs nightly without operator action.
