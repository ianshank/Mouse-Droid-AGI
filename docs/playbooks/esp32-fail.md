# ESP32 Failure Playbook

Use this playbook when ESP32 motor commands time out, the serial link drops,
or the resilient driver's circuit breaker stays `OPEN`. The ESP32 is the
primary actuator path on the Jetson Orin Nano rover (Silicon Labs CP2102N
USB-to-UART bridge at 1 Mbps).

## What This Covers

- CP2102N USB enumeration failures (`/dev/serial/by-id/...` symlink missing)
- ESP32 serial command timeout / motor command not acknowledged
- Resilient driver circuit breaker `OPEN` (cumulative failures exceeded
  `cfg.circuit_breaker.failure_threshold`)
- Battery voltage reads stuck at 0.0 V (sensor not wired) — informational,
  not a fault
- Emergency stop bypass: when `OPEN`, `emergency_stop()` MUST still flow
  through (safety-critical path, see
  [`src/mousedroid/resilience/resilient_driver.py`](../../src/mousedroid/resilience/resilient_driver.py))

## First Checks

1. Confirm the USB-to-UART bridge is enumerated:
   ```bash
   dmesg | grep -i cp210x | tail -10
   ls -l /dev/serial/by-id/ | grep CP2102N
   ```
   Expect a `usb-Silicon_Labs_CP2102N_*-port0` symlink. If absent, unplug +
   replug the USB cable; check `dmesg` for `disconnect` events.
2. Verify the rover container can see the device:
   ```bash
   docker exec mousedroid ls -l /dev/serial/by-id/ | grep CP2102N
   ```
   If empty: `docker-compose.jetson.yml` device passthrough is broken — see
   [`docker-compose.jetson.yml`](../../docker-compose.jetson.yml) `devices:`.
3. Inspect the configured port + baud:
   ```bash
   docker exec mousedroid python3 -c "
   from mousedroid.config.loader import load_settings
   from mousedroid.validation.runtime import resolve_runtime_config_paths
   cfg = load_settings(*resolve_runtime_config_paths())
   print(cfg.esp32.serial_port, cfg.esp32.serial_baud)
   "
   ```
4. Read circuit breaker state from `/metrics`:
   ```bash
   curl -s http://192.168.55.1:8080/metrics | grep esp32_circuit_state
   ```
   `0` = CLOSED (normal), `1` = OPEN (fast-fail), `2` = HALF_OPEN (probing).
5. Read recent retry attempts from container logs:
   ```bash
   docker logs mousedroid --since 5m 2>&1 | grep -E 'esp32|retry|circuit'
   ```

## Remediation Steps

1. **CP2102N missing from `/dev`**: physically reseat USB; if persistent,
   check `lsusb` for the `10c4:ea60` Silicon Labs device. Replace the cable
   if `lsusb` shows nothing.
2. **Serial port mismatch**: the production overlay
   [`config/jetson_production.yaml`](../../config/jetson_production.yaml)
   pins the full `/dev/serial/by-id/...` symlink. If you swapped the USB
   cable to a different port, the symlink stays valid (it's bound to the
   chip serial). If you swapped the ESP32 board, regenerate the symlink:
   `ls -l /dev/serial/by-id/` and update the config.
3. **Circuit breaker stuck OPEN**: this means cumulative failures exceeded
   `cfg.circuit_breaker.failure_threshold` (default 5). Check upstream
   reasons first (cable, power, ESP32 firmware crash). Once root cause is
   fixed, the breaker auto-transitions to `HALF_OPEN` after
   `cfg.circuit_breaker.recovery_timeout_s`. To force a reset (last
   resort): `sudo systemctl restart mousedroid-docker.service`.
4. **Emergency stop test**: manually trigger e-stop via the telemetry
   server's WebSocket and confirm the motors halt within
   `cfg.safety.max_loop_time_ms`. This call bypasses the circuit breaker
   per the safety contract — if it doesn't halt, escalate.
5. **Battery readings 0.0 V**: this is expected on the dev kit (no battery
   ADC wired). The production overlay disables thresholds with
   `safety.battery_critical_v: 0.0`. Don't treat 0 V as a fault.

## Cross-Reference

- [`src/mousedroid/comms/serial_driver.py`](../../src/mousedroid/comms/serial_driver.py) — raw serial driver.
- [`src/mousedroid/resilience/resilient_driver.py`](../../src/mousedroid/resilience/resilient_driver.py) — circuit breaker + retry wrapper.
- [`src/mousedroid/resilience/circuit_breaker.py`](../../src/mousedroid/resilience/circuit_breaker.py) — state machine.
- [`config/jetson_production.yaml`](../../config/jetson_production.yaml) — `esp32:`, `circuit_breaker:`, `retry:` sections.
- [`tests/hardware/test_esp32_loopback.py`](../../tests/hardware/test_esp32_loopback.py), [`tests/hardware/test_esp32_edge_cases.py`](../../tests/hardware/test_esp32_edge_cases.py) — hardware-marked regression tests.
- [`docs/playbooks/bringup-fail.md`](bringup-fail.md) — full-rover bringup if the ESP32 fault is part of a wider preflight failure.
