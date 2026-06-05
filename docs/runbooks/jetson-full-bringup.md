# Jetson Full Bring-Up Runbook

Deploy + run **every** rover subsystem (Claude/LLM, motors, lidar, camera, sensor fusion, the 10
pillars, voice, face) and expose the **unified dashboard** over WiFi. Composes the existing tooling
— `docker-compose.jetson.yml`, the systemd unit, the per-stage smoke script, and the full-validation
wrapper (`scripts/jetson_full_validation.sh`). Every threshold/port/path comes from
`config/jetson_production.yaml`, the Pydantic schema, or an env override — no hardcoded values.

> Companion to [`jetson-full-validation.md`](jetson-full-validation.md) (validation pass) and
> [`jetson-rover-smoke.md`](jetson-rover-smoke.md) (per-stage smoke triage).

## Prerequisites

- Repo at `/opt/mousedroid`; Docker + the `nvidia` runtime; host venv at `/opt/mousedroid/venv`.
- `/etc/mousedroid/docker.env` with the per-host secrets/overrides (see step 2).
- **Rover lifted / wheels clear** before any motion step.
- `ANTHROPIC_API_KEY` rotated (it was exposed in a chat transcript — NEXT_STEPS P0).

## Step 1 — Source + overlay

```bash
git -C /opt/mousedroid pull                       # trunk
sudo bash /opt/mousedroid/scripts/sync_jetson_overlay.sh   # -> /etc/mousedroid/jetson_production.yaml
```

## Step 2 — Per-host env (`/etc/mousedroid/docker.env`)

```bash
ANTHROPIC_API_KEY=sk-ant-...            # rotated; enables the Claude cloud tier
MOUSEDROID_TELEMETRY_TOKEN=<token>      # dashboard + authed endpoints
MOUSEDROID_LLM__N_GPU_LAYERS=0          # Phi-3 fallback on CPU (world model owns the iGPU)
MOUSEDROID_MOCK_HARDWARE=false
# Do NOT set MOUSEDROID_ESP32__ENABLED yet — step 3 decides it from the probe.
```

`chown ian:ian /etc/mousedroid/docker.env && chmod 600 …` (compose runs as `ian`).

## Step 3 — ESP32 probe FIRST (decides real motors vs validate-around)

**Container NOT yet running.** Attempt real motors, but never let a dead board crash-loop the
container (`orchestrator.start()` → `esp32.connect()` retries then RAISES on a dead device).

```bash
# Rover LIFTED. Host venv. Arms the bash motor stage's motion gate.
MOUSEDROID_SMOKE_PYTHON=/opt/mousedroid/venv/bin/python \
MOUSEDROID_JETSON_CONFIGS=config/jetson_production.yaml \
MOUSEDROID_SMOKE_ALLOW_MOTION=1 \
  bash scripts/jetson_smoke_test.sh serial
# then `… motor`, then `… power`
```

- **ESP32 RESPONDS:** leave `MOUSEDROID_ESP32__ENABLED` unset (default `True`) → live
  `SerialESP32Driver`. Real motors are in play; `test_send_velocity_moves_encoders` should PASS.
- **ESP32 DEAD (probe fails / circuit-breaks):** set `MOUSEDROID_ESP32__ENABLED=false` in
  `docker.env` → `MockESP32Driver`. **Required** so the container comes up; motors stay
  blocked-on-repair. Triage with the `rover-firmware-diagnosis` skill + grep the structlog:
  `esp32_raw_line`, `esp32_serial_port_overridden`, `power_chain_probe_complete`, `usbc_endpoint_*`.

> The two motion gates are distinct: `MOUSEDROID_SMOKE_ALLOW_MOTION=1` arms the **bash** motor stage;
> `ESP32Config.smoke_test_allow_motion` (via `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true`) arms
> the **pytest** `test_motor_smoke.py` path.

## Step 4 — Build + up

```bash
docker compose -f docker-compose.jetson.yml build
docker compose -f docker-compose.jetson.yml up -d
# or, with the preflight gate: sudo systemctl start mousedroid-docker
docker inspect mousedroid | grep -A5 Health     # watchdog healthcheck -> healthy
docker logs -f mousedroid                        # each subsystem's start event
```

## Step 5 — Verify every subsystem

```bash
bash scripts/jetson_full_validation.sh           # static CI -> cold hardware -> warm live
```

Confirms camera/lidar/audio/speaker/voice/face/pcie/hailo sensors, the 10 pillars
(`validate_pillars`), the orchestrator e2e tick loop, the Claude gateway (`translate_mission` →
`tier=primary`), and the #115 `/metrics` families (in-process Test B). serial/motor/power are
non-blocking (dead-ESP32 tolerant); the wrapper does **not** arm motion — that was step 3.

## Step 6 — Dashboard over WiFi (from another device)

The telemetry server binds `0.0.0.0:8080`, advertises mDNS `mousedroid-telemetry.local`, and the
unified dashboard lives at `/` (→ `/dashboard`).

```bash
# From a phone/laptop on the same WiFi:
http://<rover-ip>:8080/?token=<MOUSEDROID_TELEMETRY_TOKEN>
# or via mDNS:
http://mousedroid-telemetry.local:8080/?token=<token>
```

The page renders **camera (MJPEG) + lidar (polar) + sensor-fusion panel + status/health** from a
single `/ws` connection (token carried via the `?token=` query, persisted to the WS/stream URLs).
`GET /api/v1/network` returns the advertised `server_url` + `mdns_name`. The sensor-fusion panel
reads the `fused` summary (`n_valid/n_modalities`, `lidar_present`, per-modality `modalities`,
`fused_norm`) plus the three-state `sensor_liveness` tiles.

> **Workstation note:** to reach the auth-gated server through Claude Preview, front it with
> `tools/dashboard_proxy.py 8081 http://<rover-ip>:8080 <token>` and open `http://127.0.0.1:8081/`.

## Triage

| Symptom | Check | Fix |
|---|---|---|
| Container crash-loops at start | `docker logs mousedroid` shows `esp32` connect failure | ESP32 dead → set `MOUSEDROID_ESP32__ENABLED=false` (step 3) |
| Dashboard 401 | token missing | append `?token=<MOUSEDROID_TELEMETRY_TOKEN>` |
| Camera tile "stream unavailable" | no raw-frame source wired | expected when the camera driver lacks `capture_raw_jpeg`; features/heatmap still flow |
| Fusion panel "older server" warning | frame has no `fused` | server predates this change — redeploy trunk |
| `translate_mission` `tier=secondary` | offline / no `ANTHROPIC_API_KEY` | expected off-network; cloud resumes when reachable |
