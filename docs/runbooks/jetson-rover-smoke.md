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

The ESP32 stages (`serial`, `motor`, `power`, `mcp_motor_smoke`) run warm by
default, through `docker exec` in the running container. When that container's
orchestrator owns the ESP32 port (`esp32.enabled: true`) the wrapper refuses
them; run `MOUSEDROID_SMOKE_ESP32_MODE=cold bash scripts/jetson_full_smoke_run.sh`
instead (see [Warm-state vs cold-state smoke](#warm-state-vs-cold-state-smoke)).

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

Two gates stand between a smoke run and a moving rover, and they work at
different layers. The Pydantic gate is the one that keeps an unattended rover
still: with `ESP32Config.smoke_test_allow_motion` closed (the default), every
velocity the `motor`, `power` and `mcp_motor_smoke` stages send is zero. The
bash gate zeroes nothing; it decides whether the `motor` stage runs at all, and
records SKIP unless its variable is `1`. Two env vars, two layers:

| Layer | Env var | Default | Purpose |
|-------|---------|---------|---------|
| Bash `jetson_smoke_test.sh motor` (direct invocation) | `MOUSEDROID_SMOKE_ALLOW_MOTION` | `0` | Bash-level skip guard on the `test_motor` stage. |
| Python `assert_power_chain` / `tests/hardware/test_motor_smoke.py` | `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION` | `false` | Pydantic-resolved gate on `ESP32Config.smoke_test_allow_motion`. |

**Through the wrapper (`jetson_full_smoke_run.sh`)** — the wrapper sets the
bash env var itself when it runs the motor stage, and it forwards
`MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION` into the container (the name only:
through `docker exec` in warm mode, into the transient container in cold mode),
so operators only need to export the Pydantic one on the host:

```bash
MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true \
    bash scripts/jetson_full_smoke_run.sh
```

Exported on the host it reaches every stage. Left unset it never arrives as an
empty string, but what happens instead differs by mode: a warm stage (`docker
exec`) runs with the variable **removed**, so the gate falls back to the YAML
and schema default (closed) whatever `/etc/mousedroid/docker.env` says for the
orchestrator, and only an export on the host can open it; a cold stage (`docker
compose run`) keeps the service's own value, so a `docker.env` that opens the
gate stays open there. The `mcp_motor_smoke` stage passes `false` itself when
the host leaves it unset. A stale `=true` left exported in your shell now takes
effect, where before the wrapper stopped it at the host: unset it when you are
done.

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
`chassis_has_wheel_encoders: true`**. Scope of the two knobs: the `motor`
stage's loopback module (`tests/hardware/test_esp32_loopback.py`) honours
`smoke_test_velocity_mps` as the ceiling on every command it sends but keeps
its own 5 % tolerance window, and `smoke_test_min_velocity_fraction` belongs to
`tests/hardware/test_motor_smoke.py` (the `mcp_motor_smoke` stage); the `power`
stage's `assert_power_chain` takes the setpoint and has no window. The WAVE
ROVER ships encoder-less (vendor audit R3), so with the flag `false` the `motor`
stage stops asking for wheel velocity and requires a `FEEDBACK_BASE_INFO`
(T:1001) frame after the command, witnessed by `EncoderReading.imu_valid` (it
needs `command_set: waveshare_stock`, below); a silent board FAILS. A non-zero
command leaves the host only when BOTH gates above are open, and never exceeds
`smoke_test_velocity_mps`: with the Pydantic gate closed the stage runs at zero
velocity and still checks that the write did not raise (stock firmware has no
per-command ACK), the frame and e-stop latency. With `esp32.enabled=false` the
test skips and the stage reports FAIL with the remedy in its reason: stack
`config/jetson_bench_esp32.yaml` or unset `MOUSEDROID_ESP32__ENABLED` (the
orchestrator stacks the overlay for `motor` and `power`; an exported
`MOUSEDROID_ESP32__ENABLED=false` beats it). Setting
`MOUSEDROID_SMOKE_BENCH_ESP32_OVERLAY=` (empty) is the explicit opt-out for a
run with no chassis attached: both stages record SKIP and never run pytest. The
`power` stage applies the same rule to its own test: pytest exits 0 when
everything skips, so a clean exit is not a PASS.

### Firmware command set (F-025)

`ESP32Config.command_set` selects the wire protocol: `legacy` (default,
the historical private JSON) or `waveshare_stock` (stock `General_Driver`
firmware — `CMD_ROS_CTRL` velocity, `CMD_HEART_BEAT_SET` failsafe armed at
connect, telemetry from `FEEDBACK_BASE_INFO`). `waveshare_stock` is shipped in
`config/jetson_production.yaml` since F-059; the env var stays the per-host
override, and the lever for any overlay that does not declare it:

```bash
MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock bash scripts/jetson_smoke_test.sh serial
```

Selecting stock also derives `serial_baud=115200`, so the command set is
the only lever you need to set. The derivation is suppressed **only** by a
*non-default* baud pin; an explicit pin that happens to equal the schema
default still derives 115200, which matters because `default.yaml` pins
exactly that value. Keying the rule on "is it pinned at all" instead
would leave the rover at the legacy 1 Mbaud, where stock firmware reads as
line noise — which is how a live board gets diagnosed dead (audit R2).

The derivation runs one way only. A per-host `MOUSEDROID_ESP32__COMMAND_SET=legacy`
on `jetson_production.yaml` (a legacy-firmware board) must also set
`MOUSEDROID_ESP32__SERIAL_BAUD=1000000`: the overlay pins 115200 explicitly and
`legacy` derives nothing, so the port would otherwise open at 115200.

The serial probe follows the selector (stock probes with `{"T":130}`, a
read that elicits a reply); `SMOKE_SERIAL_PROBE_JSON` overrides it for
bench experiments.

The `serial` stage is a probe, so it needs a reply: it opens the port
exclusively (as the driver does) and silence FAILs, where it used to record a
PASS for an open nobody answered. A port another opener inside the same
container holds FAILs too, with `Could not exclusively lock port`, rather than
sharing the UART with the holder: the wrapper's `serial` stage runs by `docker
exec` in the orchestrator's own container, so it and the driver back each other.
The lock does not reach across the container boundary. It is an advisory
`flock` on one device node, and the host and every other container get their
own node for the device, so a host process or a second container holding the
port goes unseen, as does a program that opens it without locking. What keeps a
smoke stage off the orchestrator's port is stopping the orchestrator (cold mode,
or Phase 2 of the full-validation runbook), not the lock. A stage that only
skipped (pyserial missing from the image) records `SKIPPED`, never `PASS`.

## Triage matrix

| Symptom in `SUMMARY.md` | First check |
|-------------------------|-------------|
| `usbc` FAIL | `ls /dev/serial/by-id/` — confirm both CP2102N (rover) and CP2102 (lidar) cables are seated and not held by another process. |
| `serial` FAIL `Could not exclusively lock port` after `usbc` PASS | Another opener inside the same container holds the ESP32 port (the stage and the driver both open it exclusively, so the second one fails fast). The usual holder is the orchestrator with `esp32.enabled` true; through the wrapper that case is refused up front (next row), so you only see this from a direct `docker exec`. Run it cold instead: `MOUSEDROID_SMOKE_ESP32_MODE=cold` stops the orchestrator for the stage. The lock cannot see a holder outside the container, and a stage run directly on the host (`bash scripts/jetson_smoke_test.sh serial`) shares no device node with the container's orchestrator, so it would share the UART silently: `docker stop mousedroid` first and `docker restart mousedroid` after. |
| `serial`, `motor`, `power` or `mcp_motor_smoke` FAIL `warm smoke refused` | Not a rover fault. The wrapper will not open the ESP32 port beside a running orchestrator: the container's own settings have `esp32.enabled` true (serial protocol, not `mock_hardware`), or the wrapper could not read them (the reason says which). Re-run with `MOUSEDROID_SMOKE_ESP32_MODE=cold`, which stops the container for the ESP32 stages and restarts it. |
| any ESP32 stage FAIL `cold smoke refused: the transient container cannot be created (...)` | Not a rover fault, and nothing was stopped: the dry run before the stop could not create the transient container, so the orchestrator is still running. Compose's own message is in the row. The usual cause is an unreadable `/etc/mousedroid/docker.env` (`permission denied`; `compose run` reads it even with `required: false`): run the wrapper as a user that can read it, or with `sudo`. |
| any ESP32 stage FAIL `cold smoke aborted: docker stop ... failed` | The container did not stop, so it may still own the port: the stage did not run and the wrapper restarted the container. Check `docker ps` and `docker logs mousedroid`, then re-run. |
| `serial`, `motor` or `power` SKIPPED | The stage ran nothing and says why: pyserial missing from the image (`serial`), or `MOUSEDROID_SMOKE_BENCH_ESP32_OVERLAY` set empty (`motor`, `power`). A skipped ESP32 stage never records PASS. |
| `mcp_motor_smoke` PASS with a `mock driver: ...` note | The stage stacks no bench overlay, so with `esp32.enabled: false` its velocity and e-stop checks ran against `MockESP32Driver` (only the MCP polling is real). It is not a measurement of the board, and in cold mode it opened no window of its own. |
| `motor` FAIL `encoder loopback inactive` | First: is `chassis_has_wheel_encoders` right for this chassis? The WAVE ROVER is encoder-less (audit R3): with the flag `false` the stage requires a T:1001 frame (`EncoderReading.imu_valid`) instead of wheel velocity, and a silent board FAILS (the FAIL reason reads `no FEEDBACK_BASE_INFO frame after ... in N poll(s)`, with the traceback in `motor.log`; check the command-set/baud row below before the wiring). A reason of `esp32.enabled=false` means the stage never reached the board: stack `config/jetson_bench_esp32.yaml` or unset `MOUSEDROID_ESP32__ENABLED`. On an encoder-bearing chassis: rover power; motor/encoder wiring; rover firmware version. Re-run with `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true` only after lifting the rover off the ground. |
| `serial` FAIL `No response from ESP32` on a presumed-live board | Command-set/baud mismatch (audit R1/R2): stock firmware ignores legacy commands and runs 115 200 baud. The FAIL line names the `command_set` and `baud` the probe used. `waveshare_stock` is shipped on `config/jetson_production.yaml` since F-059, so on that stack first confirm the running code honoured the key (`esp32_driver_built` in the log shows the `command_set` in force); on a config that does not declare it, retry with `MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock` before declaring the board dead. |
| `power` FAIL on e-stop latency | `ESP32Config.emergency_stop_budget_ms` may need re-tuning OR the UART is congested — check `command_dispatch` log volume in `power.log`. |
| `power` FAIL on battery voltage | **Check whether the reading is real before touching the pack.** Grep the stage log for `esp32_battery_reading_unavailable` (WARN-once per connection) and `battery_reading_implausible`: either means no valid voltage came back and the number is not a measurement — chase a command-set/baud mismatch (row above), not the battery. Only when a plausible reading is present does the pack matter: charge it or raise the rail; threshold is `safety.battery_critical_v`, and the "is this even a reading" floor is `safety.battery_implausible_below_v`. |
| `lidar` FAIL | `udevadm info /dev/serial/by-id/usb-Silicon_Labs_CP2102_*` to confirm the LD19 sits on the lidar bridge, not the rover bridge. |
| `e2e` FAIL with camera reason | See `verify_sensors.py`'s `_diagnose_camera_host` output — surfaced in `e2e.log`. |
| `voice` FAIL | `SUMMARY.md` emits a "Rocky voice prerequisites" remediation section listing the specific Piper / model-path / device issue. |
| `preflight (real)` FAIL `exit=1` once after upgrading to F-059, with `overall status downgraded ok -> degraded` under `Trend:` in `phase2_preflight.log` | Expected, not a rover fault: the `esp32` check went `ok -> warn` (it used to report a vacuous OK; it now reports WARN while `esp32.enabled=false` or under the legacy command set) and `cli.preflight --trend` counts an aggregate downgrade as a regression. Re-run once to re-baseline the journal. A second consecutive downgrade, or any `newly FAILing` line, is a real finding. The first `--trend` run after the board is enabled on `waveshare_stock` also reports `check 'esp32' slowed <prev>s -> <curr>s (+<delta>s)` (its `elapsed_s` grows from milliseconds to connect-plus-poll time, past the `--trend-slow-floor-s` and `--trend-slow-ratio` thresholds) and may report `total elapsed slowed <prev>s -> <curr>s (+<delta>s)`; same remedy, re-run once to re-baseline the journal. |
| Rover boots but will not move; `estop_latch_active_at_boot` in the container log (a log line, not a `SUMMARY.md` stage result) | A latched emergency stop, not a dead board: `config/jetson_production.yaml` arms `safety.emergency_latch` (F-059 CF-2). Look first with `python -m mousedroid.cli.rearm --config /etc/mousedroid/jetson_production.yaml --status` (exit `1` = latched). Once the hazard is gone and the path is clear, `python -m mousedroid.cli.rearm --config /etc/mousedroid/jetson_production.yaml --operator <you> --confirm-area-clear`, then restart the service. **Never drop `--config`**: the CLI reads no environment to find the overlay, so without it it loads `default.yaml`, where the latch is off, says there is nothing to clear and exits `0` whether or not the rover is latched. `exit 1` = latched therefore holds only with `--config`, and a clear without it is a silent no-op. On a container deployment run both inside the service (`docker compose -f docker-compose.jetson.yml exec mousedroid ...`); the bare-metal form and the reasoning are in [`emergency-stop-rearm.md`](emergency-stop-rearm.md). |

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
is working. The wrapper runs every stage by `docker exec` into the running
container and exits 2 when the container is not running, so stopping it by hand
does not give a cold run. The cold path for those sensors is Phase 2 of
[`jetson-full-validation.md`](jetson-full-validation.md), which stops the
container and probes from the host venv.

The ESP32 stages (`serial`, `motor`, `power`, `mcp_motor_smoke`) are the
exception, because a running orchestrator and a smoke stage must not both hold
the ESP32 UART. The serial driver opens it exclusively, but that lock only
collides inside one container (see above), so the wrapper does not lean on it:
it either refuses the stage or stops the orchestrator.
`MOUSEDROID_SMOKE_ESP32_MODE` says how the wrapper gives them the port:

| Mode | What the wrapper does |
|------|-----------------------|
| `warm` (default) | Runs them through `docker exec` in the running container. Right while that container's own ESP32 is the mock (`esp32.enabled: false`, the shipped production posture), because its orchestrator then holds no port. If the orchestrator owns the port (`esp32.enabled` true, not `mock_hardware`, serial protocol), each ESP32 stage is **refused**: recorded `FAIL` with `warm smoke refused: the orchestrator in mousedroid owns the ESP32 port; re-run with MOUSEDROID_SMOKE_ESP32_MODE=cold` (`EXPECTED-FAIL` for a stage demoted to non-blocking) and not run. A refused blocking stage stops the run like any blocking failure. The wrapper refuses the same way when it cannot tell who owns the port. |
| `cold` | Dry-runs the transient container first, with the orchestrator still up (`docker compose run --rm --no-deps --entrypoint true`): if compose cannot create it, the stages are refused with compose's message and nothing is stopped. Then stops the container, runs the ESP32 stages in a transient container from the same compose service (`docker compose run --rm --no-deps --name mousedroid-cold-smoke`: same image and device passthrough, orchestrator not started, python under the stage's `timeout` like the warm wrapper) and restarts the container before the next stage that needs it: after `power`, and again after `mcp_motor_smoke` if the orchestrator owns the port (with nothing owning a port that stage measures the mock, so it runs in the container with no window and its row says so). A leftover transient container is removed before each stage. If `docker stop` fails the stage is not run, since the container may still own the port. The restore is `docker restart`, because a `docker start` issued while a stop is still running does nothing, and an EXIT trap repeats it on every exit the shell can catch, restoring before it logs so that a dropped ssh session (a dead stdout) cannot keep it from the restart. SIGKILL cannot be caught: after a `kill -9` of the wrapper inside a cold window, `docker start mousedroid` by hand. Before `llm_probe` the wrapper waits for the restarted container's healthcheck, bounded (`MOUSEDROID_SMOKE_HEALTH_RETRIES` x `MOUSEDROID_SMOKE_HEALTH_INTERVAL_S`, 30 x 5 s), and never fails the run for it. `compose run` reads `/etc/mousedroid/docker.env`, so run the wrapper as a user that can read it. |

```bash
MOUSEDROID_SMOKE_ESP32_MODE=cold bash scripts/jetson_full_smoke_run.sh
```

`SUMMARY.md` records which mode a run used. An unknown value exits 2.

If a warm smoke is the only option (e.g. you don't want to interrupt a
live run), trust the orchestrator's `/api/v1/health` for the hardware
status of LiDAR/camera/speaker, and use the smoke gate purely for
`usbc`, `system`, and `app_health`; with the ESP32 enabled the ESP32 stages
are refused.

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
