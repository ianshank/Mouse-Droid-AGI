# Spec — rover gate

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

Delta for the F-008 smoke gate and its preflight. "The gate" is `scripts/jetson_smoke_test.sh` as
invoked by F-008's `validation_command`; "the production stack" is `config/jetson_production.yaml`
loaded over the schema defaults; "the bench stack" is the production stack plus
`config/{jetson_bench_esp32}.yaml` (D-11). Every motion scenario presumes the rover on blocks,
wheels clear, operator at the power switch (CHARTER §3).

## Requirement: the motor stage MUST measure the chassis — never the mock, never a skipped assertion

The stage runs `tests/hardware/test_esp32_loopback.py` and greps
`test_send_velocity_moves_encoders PASSED` and `test_emergency_stop_latency PASSED`
(`scripts/jetson_smoke_test.sh::test_motor`). That pairing is load-bearing and is pinned (D-1). On an
encoder-less chassis (`esp32.chassis_has_wheel_encoders: false`) the criterion is a
FEEDBACK_BASE_INFO (T:1001) frame after the command, witnessed by `EncoderReading.imu_valid`; stock
firmware echoes the commanded speed into `L`/`R`, so a velocity assertion there measures nothing.

### Scenario: a silent board fails the motor stage
- **GIVEN** the bench stack and an ESP32 that returns no T:1001 frame
- **WHEN** `bash scripts/jetson_smoke_test.sh motor` runs with `MOUSEDROID_SMOKE_ALLOW_MOTION=1`
- **THEN** after up to `esp32.max_consecutive_timeouts` polled reads without a T:1001 frame,
  `test_send_velocity_moves_encoders` FAILS — not skips — naming the port, baud and command set
  (Ruling 33)
- **AND** `emergency_stop()` ran in `finally` before the failure was reported
- **AND** the stage records FAIL with that reason

### Scenario: an encoder-less chassis passes on the frame, not on wheel velocity
- **GIVEN** the bench stack, `chassis_has_wheel_encoders: false`, and a board that answers
  `{"T":130}` with a T:1001 frame
- **WHEN** the velocity test runs
- **THEN** it passes on `imu_valid is True` from any of the polled reads and never asserts
  `left_velocity_mps` or `right_velocity_mps`

### Scenario: the mock is a configuration decision, not a measurement
- **GIVEN** the production stack alone (`esp32.enabled: false`)
- **WHEN** the velocity test runs
- **THEN** it skips with a message naming `esp32.enabled=false` and the bench overlay as the remedy
- **AND** the stage's reason grep matches that message

### Scenario: the second motion gate is closed by default
- **GIVEN** the bench stack, which never sets `esp32.smoke_test_allow_motion`
- **WHEN** the velocity or e-stop-latency test runs behind `MOUSEDROID_SMOKE_ALLOW_MOTION=1`
- **THEN** every `send_velocity` carries 0.0 m/s (`_commanded_velocity` returns 0.0 while the
  Pydantic gate is closed)
- **AND** command acceptance, the T:1001 frame and e-stop latency are still measured

### Scenario: the open gate is bounded by the smoke setpoint (Ruling 32)
- **GIVEN** a bench-only overlay that sets `esp32.smoke_test_allow_motion: true` on the command line
- **WHEN** either motion test commands a velocity
- **THEN** `_commanded_velocity` returns `min(max_velocity_mps × fraction, smoke_test_velocity_mps)`
  — never above `esp32.smoke_test_velocity_mps` (default 0.05 m/s), whatever `max_velocity_mps`
  says (`5a04660`)
- **AND** `smoke_test_velocity_mps: 0.0` locks all smoke motion even with both gates open

### Scenario: the stage ↔ test pairing cannot drift silently
- **GIVEN** `tests/regression/test_f059_aqa.py::test_esp32_stages_run_their_hardware_modules_and_grep_passed`
  (parametrised over the motor and power stages) and the AST pins beside it
- **WHEN** `test_motor` stops running the loopback module, stops grepping
  `test_send_velocity_moves_encoders PASSED`, or the velocity test loses its
  `chassis_has_wheel_encoders` / `imu_valid` branch or the `_commanded_velocity` gate
- **THEN** a regression test fails

## Requirement: the preflight `esp32` check MUST probe the board; an unprobed board is WARN, never OK

`_check_esp32` connects through `factory.build_esp32_driver`, polls `{"T":130}` up to
`esp32.max_consecutive_timeouts` times and requires one T:1001 frame (`collect_esp32_probe`, D-3). No
new `PreflightStatus` member is added: `enabled=false` and a non-stock command set are WARN with the
reason in `detail`; the aggregate DEGRADED is the documented meaning (R-4). The probe is run cold:
the serial driver takes no exclusive lock until task 4.1, so a live container that owned the port
would contend for the frames rather than fail (Ruling 28).

### Scenario: a disabled board is WARN
- **GIVEN** `mock_hardware=false` and `esp32.enabled: false`
- **WHEN** `_check_esp32` runs
- **THEN** the result is WARN with `esp32.enabled=false` in `detail`
- **AND** no driver is built and no port is opened

### Scenario: legacy has no probe
- **GIVEN** `esp32.enabled: true` and `command_set: legacy`
- **WHEN** `_check_esp32` runs
- **THEN** the result is WARN ("no probe") and `collect_esp32_probe` is not called

### Scenario: a silent stock board is FAIL
- **GIVEN** `command_set: waveshare_stock` and a board that never answers
- **WHEN** `_check_esp32` runs
- **THEN** the result is FAIL after exactly `max_consecutive_timeouts` polls, with
  "no FEEDBACK_BASE_INFO" in `detail`
- **AND** the driver was disconnected

### Scenario: an answering board is OK with the frame named
- **GIVEN** a board that answers with a T:1001 frame carrying `v`
- **WHEN** `_check_esp32` runs
- **THEN** the result is OK with "FEEDBACK_BASE_INFO" in `detail`
- **AND** the probe wrote frame types `[136, 130, 130]` (heartbeat arm, encoder poll, battery
  poll) and never T:13

### Scenario: a connect error fails fast
- **GIVEN** a device node that is missing or unreadable
- **WHEN** `run_preflight(..., check_names={"esp32"})` runs
- **THEN** the `esp32` check is FAIL with the exception message in `detail`, within the serial
  timeout — not a hang
- **AND** no lock error is reported — none exists until 4.1; the docs say "FAIL on silence or a
  connect error; run it cold" (PR-C `48820c4`, Ruling 28)

### Scenario: the trend timer never schedules the probe
- **GIVEN** `scripts/mousedroid-trend.service`
- **WHEN** its `MOUSEDROID_TREND_CHECKS` value is read
- **THEN** it names no exclusive-device check (`esp32` included)
- **AND** a regression test pins that value

### Scenario: the first `--trend` run after the change downgrades once
- **GIVEN** a journal whose last `esp32` entry is OK and a rover whose production stack keeps
  `esp32.enabled: false`
- **WHEN** `python -m mousedroid.cli.preflight --trend` runs
- **THEN** it reports OK→WARN for `esp32` and exits 1 once
- **AND** the runbook instructs re-running to re-baseline, with the reason recorded; thresholds
  are never widened to hide it

## Requirement: the e-stop latch MUST be armed on production and declared off, explicitly, on the lidar-only stack

### Scenario: production arms the latch
- **GIVEN** `config/jetson_production.yaml`
- **WHEN** loaded over the schema defaults
- **THEN** `safety.emergency_latch.enabled is True`

### Scenario: the validation stack does not inherit the latch
- **GIVEN** the `scripts/jetson_validate.sh` default stack `jetson_production.yaml,jetson_lidar_only.yaml`
- **WHEN** loaded
- **THEN** `safety.lidar_unavailable_policy == "emergency"` — the premise: this is the stack that
  would latch on a 0.5 s dropout
- **AND** `safety.emergency_latch.enabled is False`
- **AND** the raw YAML of `config/jetson_lidar_only.yaml` declares
  `safety.emergency_latch.enabled: false` — explicit, not inherited (R-7)

### Scenario: every other overlay still ships the latch off
- **GIVEN** every `config/*.yaml` except `baselines.yaml` and `jetson_production.yaml`
- **WHEN** loaded
- **THEN** the latch is off; the inverted pin
  `tests/regression/test_estop_latch_backwards_compat.py::test_every_shipped_overlay_still_loads_with_the_latch_off`
  is parametrised minus production and its docstring declares the reversal

### Scenario: operators can re-arm
- **GIVEN** `docs/runbooks/jetson-rover-smoke.md` and `docs/runbooks/jetson-full-bringup.md`
- **WHEN** a rover boots latched (`estop_latch_active_at_boot` in the log)
- **THEN** both runbooks name
  `python -m mousedroid.cli.rearm --config /etc/mousedroid/jetson_production.yaml --status` and
  `python -m mousedroid.cli.rearm --config /etc/mousedroid/jetson_production.yaml --operator <you> --confirm-area-clear`
- **AND** the documented command names the production config (Ruling 36): without `--config` the
  CLI calls `load_settings()` with no path, loads `default.yaml`, sees no latch and exits 0 ("nothing
  to clear") — a silent no-op on the production container — until task 4.7 adds the
  `MOUSEDROID_CONFIG` fallback

## Requirement: battery thresholds MUST be armed on production, and the power gate MAY be stricter than the runtime

### Scenario: thresholds armed, runtime floor kept
- **GIVEN** `config/jetson_production.yaml`
- **WHEN** loaded
- **THEN** `(battery_warn_v, battery_critical_v) == (10.5, 9.5)`
- **AND** `0.0 < battery_implausible_below_v < battery_critical_v`

### Scenario: a missing voltage fails the power stage (R-6)
- **GIVEN** a real driver and a board that answers the poll without `v` (read as 0.0)
- **WHEN** `bash scripts/jetson_smoke_test.sh power` runs
- **THEN** `tests/hardware/test_power_chain_smoke.py` fails
  `battery_voltage_v >= battery_critical_v`
- **AND** the test gains no lower floor

### Scenario: a missing voltage never e-stops the runtime
- **GIVEN** the same 0.0 V reading inside the orchestrator
- **WHEN** the safety monitor evaluates it
- **THEN** `battery_implausible_below_v` (1.0) routes it to log-only, not e-stop

### Scenario: the motor stage's battery check keeps its guard (R3-5)
- **GIVEN** `tests/hardware/test_esp32_loopback.py::test_battery_voltage_in_plausible_range`
- **WHEN** the reading is 0.0
- **THEN** it passes (guard `if v > 0.0`) — the motor stage is a motion gate, the power stage is
  the telemetry gate
- **AND** the asymmetry is declared in that test's docstring; no new test is required

### Scenario: the stack pin inverts for one value only
- **GIVEN** `tests/regression/test_jetson_deploy_prep_backwards_compat.py::test_default_plus_production_overlay_stack_loads`
- **WHEN** re-pinned
- **THEN** it asserts `battery_critical_v == 9.5` and `battery_warn_v == 10.5` through the raw
  declaration, keeps `esp32.enabled is False` and `min_valid_sensors == 1`
- **AND** its docstring declares the CF-1 reversal

## Requirement: `esp32.enabled` MUST become true in exactly one file — the bench overlay

### Scenario: the overlay is one key
- **GIVEN** `config/{jetson_bench_esp32}.yaml`
- **WHEN** parsed
- **THEN** it equals `{"esp32": {"enabled": True}}`
- **AND** stacked after production, `esp32.model_copy(update={"enabled": False})` equals
  production's `esp32` and `safety` is unchanged

### Scenario: the motion gate stays closed
- **GIVEN** the bench stack
- **WHEN** loaded
- **THEN** `esp32.smoke_test_allow_motion is False`

### Scenario: the ESP32 stages stack it; the rest of the suite does not
- **GIVEN** `MOUSEDROID_JETSON_CONFIGS` unset (the singular `MOUSEDROID_JETSON_CONFIG` is the base
  when set), set to a CSV, or already containing the overlay
- **WHEN** `test_motor` or `test_power` runs
- **THEN** `esp32_stage_configs_csv()` yields production + overlay, the given CSV + overlay, or the
  given CSV unchanged (no double append)
- **AND** `test_pytest` keeps `CONFIGS_CSV` — stacking there would open the port for every module
- **AND** `tests/unit/scripts/test_jetson_smoke_orchestrator.py::test_esp32_stages_stack_the_bench_overlay`
  pins both stages over eight CSV cases

### Scenario: the no-chassis checkpoint opts out
- **GIVEN** `MOUSEDROID_SMOKE_BENCH_ESP32_OVERLAY=` (set but empty)
- **WHEN** the motor or power stage runs
- **THEN** no overlay is stacked and the stage records SKIP **without running pytest**, with the
  reason `bench ESP32 overlay opted out (MOUSEDROID_SMOKE_BENCH_ESP32_OVERLAY is empty): no chassis declared, stage not run` (`df5bcd3`) — a mock PASS would be a green that measures nothing
  (Ruling 31)
- **AND** the evidence record says the stage was skipped, not that the chassis was measured

### Scenario: a stacked run whose test skips is a FAIL
- **GIVEN** the bench overlay stacked and a board configured off by a later source or absent, so
  the hardware test skips and pytest exits 0
- **WHEN** the power or motor stage reads the output
- **THEN** it records FAIL with the skip reason — `test_power` requires
  `test_power_chain_within_budget PASSED` exactly as `test_motor` requires its own test (`57d6996`)
- **AND** `tests/regression/test_f059_aqa.py::test_power_chain_test_skips_when_the_board_is_configured_off`
  pins the skip message the stage greps

### Scenario: the orchestrator's own stack keeps the mock
- **GIVEN** the rover's `/etc/mousedroid` overlay and the compose `command:`
- **WHEN** the container starts during the bench
- **THEN** it loads production only (`esp32.enabled: false`) until F-008 is `done`
- **AND** the overlay is forwarded per command, never copied to `/etc/mousedroid`

## Requirement: the port MUST be opened exclusively, the serial stage MUST fail on silence, and a warm smoke run MUST refuse when the container owns the port (R-5 — kept; lands in Phase 4, tasks 4.1/4.2, PR-I)

Revision 1 put the exclusive open in Phase 1; R-5 moved it behind the warm/cold decision (D-13) and
kept the requirement. Until 4.1 lands no lock exists anywhere (Ruling 28), which is why the preflight
probe above is run cold and why the bench (Phase 3) opens the port from `docker exec` while the
container holds the mock.

### Scenario: the driver opens the port exclusively
- **GIVEN** `src/mousedroid/comms/serial_driver.py::_open_serial`
- **WHEN** it opens the port
- **THEN** it passes `exclusive=True` (`_SERIAL_EXCLUSIVE`, a `Final[bool]` beside
  `_SERIAL_IO_WORKERS` — not a config field: two openers must collide loudly)
- **AND** `tests/unit/comms/test_serial_driver.py::test_open_serial_requests_exclusive_access` pins
  the call

### Scenario: the serial stage is a probe, not an open/write
- **GIVEN** `scripts/jetson_smoke_test.sh::test_serial` after 4.1
- **WHEN** the board returns nothing to the probe frame
- **THEN** the stage records FAIL — `PASS: serial open/write (no response…)` no longer exists
- **AND** a port held elsewhere is FAIL with the lock error in the reason, never a silent PASS
- **AND** `tests/unit/scripts/test_smoke_harness.py` pins the FAIL on silence

### Scenario: a warm smoke run refuses when the container owns the port
- **GIVEN** `scripts/jetson_full_smoke_run.sh` with `MOUSEDROID_SMOKE_ESP32_MODE` unset (warm) and
  a container whose own settings have `esp32.enabled` true and `mock_hardware` false
- **WHEN** an ESP32 stage (`serial`, `motor`, `power`, `mcp_motor_smoke`) is reached
- **THEN** the stage records FAIL "warm smoke refused: the orchestrator in `<container>` owns the
  ESP32 port; re-run with `MOUSEDROID_SMOKE_ESP32_MODE=cold`"
- **AND** no port is opened from the stage

### Scenario: cold mode stops, runs, and always restarts
- **GIVEN** `MOUSEDROID_SMOKE_ESP32_MODE=cold`
- **WHEN** the ESP32 stages run
- **THEN** the container is stopped under an EXIT trap that restarts it (the
  `scripts/jetson_full_validation.sh` pattern) and the stages run through
  `docker compose … run --rm --no-deps … --entrypoint python3`
- **AND** the container is running again when the script exits, whatever the stages returned

### Scenario: warm-with-mock is sanctioned
- **GIVEN** a container whose own stack keeps `esp32.enabled: false` (the bench posture)
- **WHEN** the ESP32 stages run warm with the bench overlay forwarded per command
- **THEN** they run — the orchestrator holds the mock and no port — and the summary says so

### Scenario: the documented motion lever reaches the container
- **GIVEN** either mode
- **WHEN** the wrapper builds its `docker exec` / `compose run` command
- **THEN** `-e MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION` is forwarded (today it forwards no
  `MOUSEDROID_ESP32__*` variable, so the runbook's "zero-velocity by default" sentence was true by
  accident — corrected in 4.6)
- **AND** `tests/unit/scripts/test_smoke_harness.py` pins `MOUSEDROID_SMOKE_ESP32_MODE`, the
  forwarded `-e` and "warm smoke refused"
