# Rover + Jetson integration gaps (vendor-documentation audit, 2026-07-31)

Audit of the MouseDroid wiring/integration surface against the **authoritative vendor
sources** for the two platforms we actually run on:

- Waveshare **WAVE ROVER** chassis + *General Driver for Robots* board — product wiki and
  the stock ESP32 firmware source (`waveshareteam/ugv_base_general`, `General_Driver/`).
- **NVIDIA Jetson Orin Nano** — JetPack 6.2 "Super mode" power-mode set.

Scope: what the vendor hardware offers that we do **not** currently consume, plus places
where our code contradicts the vendor contract. Findings only — no code changes are made
by this document. Severity is *impact on the rover working*, not code quality.

> Companion to [`../runbooks/jetson-full-bringup.md`](../runbooks/jetson-full-bringup.md)
> (bring-up), [`../runbooks/jetson-rover-smoke.md`](../runbooks/jetson-rover-smoke.md)
> (per-stage triage), and the **F-008** blocker in `NEXT_STEPS.md`.

---

## R1 — Our ESP32 JSON protocol does not match stock Waveshare firmware (blocking)

`src/mousedroid/comms/_utils.py` defines a **three-command private protocol**:

| Ours | Meaning we assume | `General_Driver/json_cmd.h` actually defines |
| --- | --- | --- |
| `{"T":0}` | emergency stop | *(no chassis command 0; `CMD_GIMBAL_CTRL_STOP` is 135)* |
| `{"T":1,"vx","vy","omega"}` | velocity as ±255 PWM triplet | `CMD_SPEED_CTRL` — expects **`{"T":1,"L":<f>,"R":<f>}`** |
| `{"T":2}` | battery-voltage query | **`CMD_SET_MOTOR_PID`** |

Three consequences, in order of severity:

1. **`{"T":2}` is not a read — it is a write.** On stock firmware, command 2 sets the
   motor PID gains. `get_battery_voltage()` therefore pokes the motor controller with an
   argument-less PID-set on every battery poll. This runs in `assert_power_chain` before
   the velocity step, i.e. immediately before we command motion.
2. **Motion commands are silently no-ops.** Stock `CMD_SPEED_CTRL` reads keys `L` and `R`;
   we send `vx`/`vy`/`omega`. Missing keys parse as zero, so the firmware ACKs and the
   wheels never turn — indistinguishable from a dead board at the smoke-test level.
3. **`emergency_stop()` sends an unhandled command.** The real stop is
   `{"T":1,"L":0,"R":0}` (or `{"T":13,"X":0,"Z":0}`), ideally backed by the heartbeat
   failsafe (R6).

The private protocol is not wrong *in principle* — `scripts/flash_esp32.sh` implies a
custom build — but **`firmware/waverover_mousedroid.bin` does not exist in this repo, and
no firmware source does either.** There is nothing to flash that would understand these
commands, so any board we attach (repaired original or replacement) will run stock
firmware and will not respond to our driver.

> This does **not** retract the F-008 diagnosis. The operator confirmed the current board
> is unresponsive on UART *and* ROM bootloader *and* WiFi-AP broadcast; a protocol
> mismatch cannot explain an absent ROM bootloader. The point is that R1 is a **second,
> latent** fault that will surface the moment the hardware fault is cleared.

**Suggested fix:** retarget the driver at the stock command set —
`CMD_ROS_CTRL` (`{"T":13,"X":<m/s>,"Z":<rad/s>}`) maps 1:1 onto our `(vx, omega)` and
removes the PWM-scaling maths in `build_velocity_cmd` entirely. Keep the command IDs in
`_utils.py` as named constants sourced from `json_cmd.h`.

## R2 — Baud rate is 1 000 000; stock firmware is 115 200 (blocking)

`ESP32Config.serial_baud` defaults to `1_000_000` and `config/jetson_production.yaml`
pins the same. Stock `General_Driver` runs the JSON UART at **115 200**. At the wrong
baud every frame is line noise — which is exactly what a dead board looks like through
`esp32_raw_line`. Worth re-testing the "dead" board at 115 200 before the repair-vs-
replace decision; it costs one env override (`MOUSEDROID_ESP32__SERIAL_BAUD=115200`).

## R3 — WAVE ROVER has no wheel encoders (design-invalidating)

Waveshare is explicit: the WAVE ROVER ships **encoder-less** motors — speed values are
PWM duty (`0.5` = 100 %), and PID/closed-loop commands do not apply. Encoders are a
UGV01/UGV02/UGV-Rover feature.

This invalidates several things we treat as load-bearing:

- `EncoderReading.odometry_x_m` / `odometry_y_m` / `heading_rad` can never be non-zero.
- `tests/hardware/test_motor_smoke.py::test_velocity_roundtrip_clamps_and_dispatches`
  asserts encoder velocity reaches `smoke_test_min_velocity_fraction` of the setpoint.
  **That assertion is unsatisfiable on this chassis** — it is not a bring-up gate, it is
  a permanent red. F-008's "smoke passes on the rover" verification inherits the problem.
- The `motor` slot in the 5-modality fusion mask is fed from a channel that is
  structurally zero.

**Suggested fix:** either accept open-loop control and re-scope the smoke assertion to
"command accepted, e-stop within budget" (matching what the hardware can prove), or
source odometry from the IMU (R4). Decide explicitly — do not leave an impossible gate
sitting on the critical path.

## R4 — The onboard IMU and magnetometer are completely unwired (highest-value gap)

The driver board integrates a **QMI8658** 6-axis IMU and an **AK09918** 3-axis
magnetometer, exposed by stock firmware as:

| Command | Purpose |
| --- | --- |
| `CMD_GET_IMU_DATA` = 126 | one-shot IMU read → `FEEDBACK_IMU_DATA` (1002) |
| `CMD_CALI_IMU_STEP` = 127 | calibration step |
| `CMD_GET_IMU_OFFSET` = 128 / `CMD_SET_IMU_OFFSET` = 129 | offset persistence |
| `CMD_BASE_FEEDBACK` = 130 | chassis info → `FEEDBACK_BASE_INFO` (1001): wheel speed, **IMU**, **voltage** |
| `CMD_BASE_FEEDBACK_FLOW` = 131 | enable *continuous* feedback stream |
| `CMD_FEEDBACK_FLOW_INTERVAL` = 142 | stream interval (ms) |

Grep confirms **zero** IMU/gyro/accelerometer/magnetometer references in
`sensing/`, `hardware/`, or `comms/`. `RoverInertialConfig` is sim mass-properties, not a
sensor. The fusion mask (`_MODALITY_NAMES`) is `vision, ultrasonic, motor, audio, lidar`.

What we lose by not wiring it:

- **Tip-over detection.** `RoverInertialConfig` documents a deliberately top-heavy COM
  (`com_offset_xyz_m = (0, 0, 0.04)`) so the policy learns roll tendency in sim — but on
  hardware there is no roll/pitch signal for the safety monitor to act on. The one sensor
  that would close that loop is already on the board.
- **Heading.** With no encoders (R3), yaw + magnetometer is the *only* heading source.
- **A real motor-channel observation** to replace the structurally-zero encoder slot.
- Free replacement for the disabled battery monitoring (R5) — voltage rides along in the
  same `FEEDBACK_BASE_INFO` frame.

**Suggested fix:** enable the feedback flow (`{"T":131,...}` + `{"T":142,"cmd":<ms>}`) and
parse `FEEDBACK_BASE_INFO` once per slow-cadence tick, rather than polling. Add `imu` as
a 6th fusion modality behind a default-OFF `Optional` config block, per the established
additive pattern.

## R5 — Battery monitoring is disabled on a rig that has an INA219

`config/jetson_production.yaml` zeroes `battery_critical_v` / `battery_warn_v` with the
comment *"the ESP32 does not report a reliable voltage over the existing serial
protocol."* That is true of **our** protocol (R1: `{"T":2}` is a PID write), but the board
carries an **INA219** voltage/current sensor and stock firmware reports voltage in
`FEEDBACK_BASE_INFO`.

Net effect today: a 3S 18650 pack driving a Jetson has **no undervoltage protection** —
the documented 9.5 V cutoff is unenforced. Deep-discharging 18650s is a safety issue, not
just a data gap. Fixing R1/R4 fixes this for free; the config comment should then be
retracted.

## R6 — No heartbeat failsafe (`CMD_HEART_BEAT_SET` = 136)

Stock firmware supports a heartbeat: if no command arrives within the configured window,
the chassis stops on its own. We do not set it. If the Jetson wedges, the container is
OOM-killed, or the USB link drops mid-motion, **the last velocity command keeps executing
indefinitely** — the rover drives until the battery dies or it hits something. Our
watchdog (`loop.watchdog_*`) restarts the *container*; it does not stop the *wheels*.

This is the single cheapest safety win available: one command at connect time, set to a
small multiple of `keepalive_hz`.

## R7 — Direct 40-pin UART path is unused (diagnostic + resilience)

The driver board exposes the ESP32's `U0TX`/`U0RX` on the **40-pin GPIO header** (pins 10
/ 8), in parallel with the onboard CP2102N USB bridge. We only ever talk over USB-C
(`diagnostics/usbc.py` has no non-USB path), and the F-008 diagnosis was performed
entirely across the two USB-C ports.

Two uses:

1. **Fault isolation, ~2 jumper wires.** Jetson 40-pin pins 8/10 are `UART1_TX`/`UART1_RX`
   → `/dev/ttyTHS1`. If the ESP32 answers there, the fault is the USB bridge or the
   USB-C port, **not the ESP32** — which changes the repair-vs-replace decision. (If it
   stays silent, that corroborates a dead/unpowered ESP32, consistent with the absent
   WiFi AP.) Worth doing before ordering a replacement board.
2. **Permanent link.** Frees a USB port and removes the CP2102N and the by-id-drift
   failure mode (`esp32_serial_port_overridden`) from the critical path.

Requires: `MOUSEDROID_ESP32__SERIAL_PORT=/dev/ttyTHS1`, a `/dev/ttyTHS1` entry in
`docker-compose.jetson.yml` `devices:` (the container is already `privileged: true`, so
this is cleanliness rather than a hard requirement), and `usbc_discovery` tolerating a
non-`by-id` port.

## R8 — Unused board peripherals

Available on the *General Driver* board, unconsumed by us. None are blocking; listed so
the decision to skip them is deliberate rather than accidental.

| Capability | Command(s) | Possible MSE-6 use |
| --- | --- | --- |
| Onboard OLED | `CMD_OLED_CTRL` 3 / `CMD_OLED_DEFAULT` −3 | second status surface; we drive a separate SSD1306 on Jetson I²C-7 |
| LED / IO output | `CMD_LED_CTRL` 132 (`IO4`/`IO5` PWM) | MSE-6 running lights, motion-armed indicator |
| Serial-bus servo bus | `CMD_SET_SERVO_ID` 501, `CMD_SET_MIDDLE` 502, `CMD_SET_SERVO_PID` 503 | head/pan-tilt without extra hardware |
| ESP-NOW | 300–306 | rover-to-rover, or a physical e-stop fob |
| Speed-rate limits | `CMD_SET_SPD_RATE` 138 / 139 / 140 | firmware-side velocity clamp under the software clamp |
| Board-side LD19 connector | 4-pin header | powers + routes the LiDAR from the chassis instead of a second USB adapter |
| Chassis geometry | `ugv_config.h`: `WHEEL_D`, `ONE_CIRCLE_PLUSES`, `TRACK_WIDTH` | our sim/URDF values are not cross-checked against the firmware's |

## R9 — Documentation contradicts the hardware

- `README.md` calls the chassis **"Wave Rover mecanum-wheel"**. WAVE ROVER is a **4WD
  differential/skid-steer** platform. This matters beyond prose: `send_velocity(vx, vy,
  omega)` carries a lateral `vy` term that the chassis physically cannot execute, and
  `build_velocity_cmd` scales and transmits it.
- `README.md` BOM lists the camera as **IMX500** (Raspberry Pi AI Camera, onboard NPU);
  `config/jetson_production.yaml` runs **IMX708** over `jetson_csi` with
  `use_onboard_inference: false` and the comment "IMX708 has no onboard NPU".
  `CameraConfig`'s docstring still says "Raspberry Pi AI Camera (IMX500) configuration".
- `README.md` lists a **3S LiPo**; the chassis ships a **3S 18650 UPS module** with
  simultaneous charge/discharge. Different charging discipline and different failure mode.

---

## J1 — Jetson power mode: Super mode is unreachable through config

`JetsonConfig.power_mode` is `Literal["15W", "7W"]`. JetPack 6.2 added **25 W** and
**MAXN_SUPER** for Orin Nano (up to 67 TOPS, ~+50 % memory bandwidth) — the schema cannot
express either.

Two separate problems:

1. **The field is dead.** `grep` finds no consumer of `power_mode` anywhere outside
   `schema.py`. It is documentation that looks like configuration.
2. **The setup script under-provisions.** `scripts/jetson_system_setup.sh` runs
   `nvpmodel -m 0`. On a JetPack-6.2 Orin Nano, MAXN_SUPER is a *different* mode index
   (`nvpmodel -m 2` per NVIDIA's Super-mode announcement) — so the rover currently runs
   the legacy budget while the 30 Hz loop shares the iGPU between the world model and
   (per `MOUSEDROID_LLM__N_GPU_LAYERS=0` on the rover) a CPU-bound Phi-3 fallback.

**Suggested fix:** widen the `Literal` to include `25W` / `MAXN_SUPER`, then either wire
it to `nvpmodel` at preflight or delete the field. Verify the mode index on-device with
`nvpmodel -q` and `cat /etc/nvpmodel.conf` before changing the setup script — indices are
per-module and per-JetPack, so this must be read off the actual rover, not assumed.
Re-measure the loop-latency budgets after any change; the `performance` CI stage is
Jetson-calibrated.

## J2 — `dla_enabled` on a module whose DLA availability is unconfirmed

`JetsonConfig.dla_enabled` feeds `efficiency/tensorrt.py`. NVIDIA's own materials are
inconsistent about whether Orin Nano exposes a DLA at all (Orin NX/AGX have 2). We ship
`false`, so nothing is broken today — but the flag is a trap for a future operator.
Confirm on-device (TensorRT `getNbDLACores()`) and, if the answer is zero, make the field
raise rather than silently degrade.

---

## Priority

| # | Finding | Severity | Blocks |
| --- | --- | --- | --- |
| R1 | Private JSON protocol vs stock firmware | **Blocking** | any motion, post-repair |
| R2 | 1 Mbaud vs 115 200 | **Blocking** | any serial link |
| R6 | No heartbeat failsafe | **Safety** | untethered operation |
| R5 | Battery monitoring disabled despite INA219 | **Safety** | 18650 pack protection |
| R3 | Encoder assertions unsatisfiable on this chassis | High | F-008 verification |
| R4 | IMU + magnetometer unwired | High | tip-over safety, heading |
| R7 | 40-pin UART path unused | Medium | F-008 repair-vs-replace decision |
| J1 | Super mode unreachable; `power_mode` dead | Medium | inference headroom |
| R9 | Docs contradict hardware (mecanum, IMX500, LiPo) | Medium | operator trust |
| R8 | Unused board peripherals | Low | — |
| J2 | `dla_enabled` unverified | Low | — |

**Recommended order:** R2 + R7 first (they are *diagnostics* — they may change the F-008
repair-vs-replace decision and cost almost nothing), then R1 + R6 + R5 as one driver
change against the stock command set, then R3's scope decision, then R4.

---

## Sources

- [WAVE ROVER — Waveshare Wiki](https://www.waveshare.com/wiki/WAVE_ROVER)
- [`waveshareteam/ugv_base_general`](https://github.com/waveshareteam/ugv_base_general) —
  `General_Driver/json_cmd.h` (command constants), `General_Driver/ugv_config.h`
  (chassis geometry, `mainType` 1 = WAVE ROVER)
- [`waveshareteam/ugv_rpi`](https://github.com/waveshareteam/ugv_rpi) — `base_ctrl.py`
  (115 200 baud; `{"T":1,"L","R"}` usage; feedback framing)
- [08 Sub-controller JSON Command Set — Waveshare Wiki](https://www.waveshare.com/wiki/08_Sub-controller_JSON_Command_Set)
- [06 Retrieving Chassis Feedback Information — Waveshare Wiki](https://www.waveshare.com/wiki/06_Retrieving_Chassis_Feedback_Information)
- [General Driver for Robots — Waveshare](https://www.waveshare.com/general-driver-for-robots.htm)
- [JetPack 6.2 Brings Super Mode to Jetson Orin Nano and Orin NX — NVIDIA](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/)
- [JetPack SDK 6.2 — NVIDIA Developer](https://developer.nvidia.com/embedded/jetpack-sdk-62)
