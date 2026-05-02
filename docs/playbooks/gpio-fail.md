# GPIO Failure Playbook

Use this playbook when GPIO-related stages of the smoke test fail
(`gpio` stage in `scripts/jetson_full_smoke_run.sh`) or the SSD1306 OLED
face display fails to boot.

## What This Covers

- `Jetson.GPIO` import failure / model detection mismatch
- I²C-7 bus enumeration failure (the SSD1306 face display lives there)
- SSD1306 not responding at the configured `i2c_address` (`0x3C`)
- Pin reassignment after a hardware revision

## First Checks

1. Confirm `Jetson.GPIO` reports the expected model:
   ```bash
   docker exec mousedroid python3 -c "
   import Jetson.GPIO as GPIO
   print('model:', GPIO.model)
   print('jetson_info:', GPIO.JETSON_INFO)
   "
   ```
   Expected on Orin Nano: `model: JETSON_ORIN_NANO`.
2. Check the I²C-7 bus is exposed inside the container:
   ```bash
   docker exec mousedroid ls -l /dev/i2c-7
   ```
   If missing, `docker-compose.jetson.yml` `devices:` is missing
   `/dev/i2c-7` — re-add and restart the service.
3. Probe the I²C bus for the SSD1306 (default address `0x3C`):
   ```bash
   docker exec --privileged mousedroid i2cdetect -y -r 7
   ```
   Expect `3c` in the dump. If empty, the OLED is unpowered or wired to a
   different bus.
4. Inspect the configured face display block:
   ```bash
   docker exec mousedroid python3 -c "
   from mousedroid.config.loader import load_settings
   from mousedroid.validation.runtime import resolve_runtime_config_paths
   cfg = load_settings(*resolve_runtime_config_paths())
   print(cfg.face_display.enabled, cfg.face_display.i2c_bus, hex(cfg.face_display.i2c_address))
   "
   ```
   Default values come from
   [`config/jetson_production.yaml`](../../config/jetson_production.yaml) —
   `enabled: true`, `i2c_bus: 7`, `i2c_address: 0x3C`.

## Remediation Steps

1. **`Jetson.GPIO` model mismatch**: the L4T base image must match the
   board. If running on a non-Orin board (developer kit older revision),
   override the model detection by setting `JETSON_MODEL_NAME` env var in
   `docker-compose.jetson.yml`. Otherwise rebuild the image with the
   correct L4T tag for the board.
2. **`/dev/i2c-7` missing on host**: enable the device tree overlay for
   I²C-7 — on Orin Nano this is on pins 3/5 of the 40-pin header. Confirm
   with `sudo /opt/nvidia/jetson-io/jetson-io.py`.
3. **SSD1306 absent from `i2cdetect`**: check the wiring against
   [`config/jetson_production.yaml`](../../config/jetson_production.yaml)
   header comment (VCC→pin 1, GND→pin 6, SDA→pin 3, SCL→pin 5). With the
   OLED reseated, `i2cdetect -y -r 7` should show `3c`. If the OLED has
   an alternate address (some boards use `0x3D`), update
   `face_display.i2c_address` in the overlay and restart.
4. **Driver crash on first frame**: enable mock fallback so the rest of
   the rover keeps running while the display is repaired:
   ```yaml
   face_display:
     fallback_to_mock_on_error: true   # already the production default
   ```
   Then check `docker logs mousedroid | grep face_display` for the
   structured error event.
5. **Display works but expressions look corrupted**: confirm `width`,
   `height`, and `refresh_hz` match the panel — defaults are 128×64 at
   10 Hz; some panels are 128×32. Mismatch produces visual tearing but
   does not crash.

## Cross-Reference

- [`src/mousedroid/hardware/display/`](../../src/mousedroid/hardware/display/) — SSD1306 driver, expressions, mock fallback.
- [`config/jetson_production.yaml`](../../config/jetson_production.yaml) — `face_display:` block.
- [`tests/hardware/`](../../tests/hardware/) — `@pytest.mark.hardware` GPIO + display tests.
- [`scripts/jetson_full_smoke_run.sh`](../../scripts/jetson_full_smoke_run.sh) — `gpio`, `oled` smoke stages.
- [`docs/playbooks/voice-fail.md`](voice-fail.md) — separate USB-audio path; not GPIO-related.
- [`docs/playbooks/bringup-fail.md`](bringup-fail.md) — full-rover bringup if the GPIO fault blocks startup.
