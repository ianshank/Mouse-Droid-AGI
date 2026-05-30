# Jetson + USB-C Rover Smoke Runbook

Operator-facing companion to `scripts/jetson_full_smoke_run.sh`. Every threshold
referenced below lives in `config/jetson_production.yaml` (or the Pydantic
`Settings` schema) — no values are hardcoded in this document.

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
unattended. Override only when the rover is on rollers or tethered:

```bash
MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true \
    bash scripts/jetson_full_smoke_run.sh
```

This routes through `ESP32Config.smoke_test_allow_motion`. The actual
setpoint comes from `ESP32Config.smoke_test_velocity_mps`; the assertion
window from `smoke_test_min_velocity_fraction`.

## Triage matrix

| Symptom in `SUMMARY.md` | First check |
|-------------------------|-------------|
| `usbc` FAIL | `ls /dev/serial/by-id/` — confirm both CP2102N (rover) and CP2102 (lidar) cables are seated and not held by another process. |
| `serial` FAIL after `usbc` PASS | Container may be holding the port. Restart with `docker compose -f docker-compose.jetson.yml restart mousedroid`. |
| `motor` FAIL `encoder loopback inactive` | Rover power; motor/encoder wiring; rover firmware version. Re-run with `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true` only after lifting the rover off the ground. |
| `power` FAIL on e-stop latency | `ESP32Config.emergency_stop_budget_ms` may need re-tuning OR the UART is congested — check `command_dispatch` log volume in `power.log`. |
| `power` FAIL on battery voltage | Charge the pack or raise the rail; threshold is `safety.battery_critical_v`. |
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
```
