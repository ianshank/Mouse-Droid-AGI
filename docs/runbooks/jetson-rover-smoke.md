# Jetson + USB-C Rover Smoke Runbook

Operator-facing companion to `scripts/jetson_full_smoke_run.sh`. Every threshold
referenced below lives in `config/jetson_production.yaml` (or the Pydantic
`Settings` schema) — no values are hardcoded in this document.

> For a full validation pass that wraps this smoke run together with static CI,
> the runtime CLIs, the live `/metrics` surface, and the deliberative gateway
> checks in one ordered, cold-then-warm flow, see
> [`jetson-full-validation.md`](jetson-full-validation.md)
> (`scripts/jetson_full_validation.sh`).

## Prerequisites

- Wave Rover plugged into the Jetson Orin Nano USB-C data port (this enables
  the `rover_esp32` endpoint discovered by `usbc_discovery`).
- LD19 LiDAR connected on its CP2102 USB-UART bridge (`lidar_ld19` endpoint).
- Rover powered on — battery or bench PSU above
  `safety.battery_critical_v`.
- Docker container `mousedroid` running. Bring it up with
  `docker compose -f docker-compose.jetson.yml up -d` if necessary.
- Repository checked out at `/opt/mousedroid` (the runbook assumes that
  layout; override `MOUSEDROID_SMOKE_REPORT_ROOT` to write reports elsewhere).

## Quick start

```bash
bash scripts/jetson_full_smoke_run.sh
```

The wrapper writes a stamped report directory under
`reports/jetson_smoke/<UTC-timestamp>/` containing one log per stage plus
`SUMMARY.md`. Exit code is non-zero iff a blocking stage failed.

To check USB-C wiring without a full smoke pass:

```bash
python scripts/check_usbc_devices.py \
    --config config/jetson_production.yaml
```

Exit code 0 means every `required_endpoints` entry in
`usbc_discovery.required_endpoints` resolved against `usbc_discovery.by_id_root`.

## Stage gating (defaults)

| Stage | Default | Override env var |
|-------|---------|------------------|
| `system` | blocking | `MOUSEDROID_SMOKE_BLOCKING_SYSTEM=no` |
| `usbc` | blocking | `MOUSEDROID_SMOKE_BLOCKING_USBC=no` |
| `gpio` | blocking | `MOUSEDROID_SMOKE_BLOCKING_GPIO=no` |
| `serial` | blocking | `MOUSEDROID_SMOKE_BLOCKING_SERIAL=no` |
| `motor` | non-blocking | `MOUSEDROID_SMOKE_BLOCKING_MOTOR=yes` |
| `power` | blocking | `MOUSEDROID_SMOKE_BLOCKING_POWER=no` |
| `lidar` | blocking | `MOUSEDROID_SMOKE_BLOCKING_LIDAR=no` |
| `camera`/`audio`/`speaker`/`voice` | non-blocking | `MOUSEDROID_SMOKE_BLOCKING_<NAME>=yes` |
| `oled` | non-blocking | `MOUSEDROID_SMOKE_BLOCKING_OLED=yes` |
| `app_health` | blocking | `MOUSEDROID_SMOKE_BLOCKING_APP_HEALTH=no` |
| `hardware_pytest` | non-blocking | `MOUSEDROID_SMOKE_BLOCKING_HARDWARE_PYTEST=yes` |
| `e2e` | non-blocking | `MOUSEDROID_SMOKE_BLOCKING_E2E=yes` |
| `mcp_motor_smoke` | non-blocking | `MOUSEDROID_SMOKE_BLOCKING_MCP_MOTOR_SMOKE=yes` |
| `llm_probe` | blocking | `MOUSEDROID_SMOKE_BLOCKING_LLM_PROBE=no` |

## Motion gate (rover safety)

The motor and power-chain stages default to **dispatching zero-velocity
commands** so an untethered rover does not roll while the smoke runs
unattended. Two env vars gate motion, used by two different layers:

| Layer | Env var | Default | Purpose |
|-------|---------|---------|---------|
| Bash `jetson_smoke_test.sh motor` (direct invocation) | `MOUSEDROID_SMOKE_ALLOW_MOTION` | `0` | Bash-level skip guard on the `test_motor` stage. |
| Python `assert_power_chain` / `tests/hardware/test_motor_smoke.py` | `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION` | `false` | Pydantic-resolved gate on `ESP32Config.smoke_test_allow_motion`. |

**Through the wrapper (`jetson_full_smoke_run.sh`)** — the wrapper already
sets the bash env var when running the motor stage, so operators only
need to override the Pydantic one:

```bash
MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true \
    bash scripts/jetson_full_smoke_run.sh
```

**Direct invocation (`jetson_smoke_test.sh motor`)** — operators must set
BOTH env vars so the bash gate AND the Python gate let motion through:

```bash
MOUSEDROID_SMOKE_ALLOW_MOTION=1 \
MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true \
    bash scripts/jetson_smoke_test.sh motor
```

The actual setpoint comes from `ESP32Config.smoke_test_velocity_mps`
(setting `0.0` permanently locks the smoke harness to zero-motion); the
assertion window from `smoke_test_min_velocity_fraction` — **only when
`chassis_has_wheel_encoders: true`**. The WAVE ROVER ships encoder-less
(vendor audit R3), so with the flag `false` the motion criterion re-scopes
to "command accepted + e-stop within budget".

### Firmware command set (F-025)

`ESP32Config.command_set` selects the wire protocol: `legacy` (default,
the historical private JSON) or `waveshare_stock` (stock `General_Driver`
firmware — `CMD_ROS_CTRL` velocity, `CMD_HEART_BEAT_SET` failsafe armed at
connect, telemetry from `FEEDBACK_BASE_INFO`). Until
`deployments/jetson-image.json` is re-pinned, opt in via env only:

```bash
MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock bash scripts/jetson_smoke_test.sh serial
```

Selecting stock also derives `serial_baud=115200` unless explicitly
pinned — stock firmware at the legacy 1 Mbaud reads as line noise, which
is exactly how a live board gets diagnosed dead (audit R2). The serial
probe follows the selector (stock probes with `{"T":130}`, a read that
elicits a reply); `SMOKE_SERIAL_PROBE_JSON` overrides it for bench
experiments.

## Triage matrix

| Symptom in `SUMMARY.md` | First check |
|-------------------------|-------------|
| `usbc` FAIL | `ls /dev/serial/by-id/` — confirm both CP2102N (rover) and CP2102 (lidar) cables are seated and not held by another process. |
| `serial` FAIL after `usbc` PASS | Container may be holding the port. Restart with `docker compose -f docker-compose.jetson.yml restart mousedroid`. |
| `motor` FAIL `encoder loopback inactive` | First: is `chassis_has_wheel_encoders` right for this chassis? The WAVE ROVER is encoder-less — the loopback criterion is unsatisfiable there (audit R3; set the flag `false`). On an encoder-bearing chassis: rover power; motor/encoder wiring; rover firmware version. Re-run with `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true` only after lifting the rover off the ground. |
| `serial` WARN no response on a presumed-live board | Command-set/baud mismatch (audit R1/R2): stock firmware ignores legacy commands and runs 115 200 baud. Retry with `MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock` before declaring the board dead. |
| `power` FAIL on e-stop latency | `ESP32Config.emergency_stop_budget_ms` may need re-tuning OR the UART is congested — check `command_dispatch` log volume in `power.log`. |
| `power` FAIL on battery voltage | **Check whether the reading is real before touching the pack.** Grep the stage log for `esp32_battery_reading_unavailable` (WARN-once per connection) and `battery_reading_implausible`: either means no valid voltage came back and the number is not a measurement — chase a command-set/baud mismatch (row above), not the battery. Only when a plausible reading is present does the pack matter: charge it or raise the rail; threshold is `safety.battery_critical_v`, and the "is this even a reading" floor is `safety.battery_implausible_below_v`. |
| `lidar` FAIL | `udevadm info /dev/serial/by-id/usb-Silicon_Labs_CP2102_*` to confirm the LD19 sits on the lidar bridge, not the rover bridge. |
| `e2e` FAIL with camera reason | See `verify_sensors.py`'s `_diagnose_camera_host` output — surfaced in `e2e.log`. |
| `voice` FAIL | `SUMMARY.md` emits a "Rocky voice prerequisites" remediation section listing the specific Piper / model-path / device issue. |

## Reading the logs

Every stage writes a structlog JSON stream to its `<stage>.log` file. Useful
greps:

```bash
# All command dispatches across the run (rover motion audit):
jq -c 'select(.event=="command_dispatch")' reports/jetson_smoke/*/power.log

# USB-C resolution detail:
jq -c 'select(.event|startswith("usbc_endpoint"))' reports/jetson_smoke/*/usbc.log

# Power-chain probe summary:
jq -c 'select(.event=="power_chain_probe_complete")' reports/jetson_smoke/*/power.log

# When rover firmware drifts (stock vs custom JSON schema), the raw line is
# logged at DEBUG so operators can see exactly what came back:
jq -c 'select(.event=="esp32_raw_line" or .event=="esp32_non_json_response")' \
    reports/jetson_smoke/*/power.log

# Confirm the factory used the live by-id path instead of the stale literal:
jq -c 'select(.event=="esp32_serial_port_overridden")' reports/jetson_smoke/*/*.log
```

## Warm-state vs cold-state smoke

A "warm" smoke run (orchestrator container is up) will fail every stage
whose hardware is already owned by the running orchestrator — LiDAR
(serial port), GPIO (ultrasonic pins), USB speaker (PCM stream), and the
CSI camera (V4L2 device). These are **false negatives**, not real failures:
the orchestrator's own `/api/v1/health` endpoint confirms the same hardware
is working.

For a true cold-state smoke run:

```bash
docker stop mousedroid
bash scripts/jetson_full_smoke_run.sh
docker start mousedroid
```

If a warm smoke is the only option (e.g. you don't want to interrupt a
live run), trust the orchestrator's `/api/v1/health` for the hardware
status of LiDAR/camera/speaker, and use the smoke gate purely for
`usbc`, `system`, `power`, and `app_health`.

## Rover swap / by-id drift

`config/jetson_production.yaml` previously hardcoded a literal CP2102N
by-id path; swapping rovers (each ESP32's CP2102N has a unique serial)
broke `esp32.serial_port`. As of the USB-C smoke PR follow-up, the
factory consults `usbc_discovery.required_endpoints["rover_esp32"]`
first when:

1. `usbc_discovery.enabled: true`, AND
2. the literal `esp32.serial_port` does not exist on disk

In every other case the literal pin still wins (operator's choice is
preserved). Watch for the `esp32_serial_port_overridden` log event to
confirm the override fired during boot.
