# Design: `mouse-droid-rover-gate-closeout` (F-059)

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

Each section is a decision with its rejected alternatives, the regression pins it **inverts** (named
by `file::test` symbol — every inversion declares the reversal in its docstring), the pins it **adds**,
and the task that lands it. D-1 … D-13 are the sections `peer-review.md` names (D-14 was added in fix
round 1, 2026-10-19, for Ruling 22); where the review's
disposition column is the only surviving record of a section's wording, the section is reconstructed
from it and says so. Revision-2 wording that round 3 overturned is struck through in place.

**Governing constraints on every section.**

- **Dual motion gate.** Nothing moves a wheel unless bash `MOUSEDROID_SMOKE_ALLOW_MOTION=1` **and**
  Pydantic `esp32.smoke_test_allow_motion=true` are both open; rover on blocks, operator present
  (CHARTER §3). The bench overlay never sets the Pydantic gate.
- **One place for the flip.** `esp32.enabled` becomes `true` only in the bench overlay (D-11);
  production keeps `false` until F-008 is `done`.
- **Code before config.** A rover still on the `032942b` image has no 1.0 V implausibility floor,
  so CF-1 thresholds installed before trunk code would e-stop on a missing reading (D-4, D-6).
- **Fence and retarget land together.** The rsync fence (2.4a) and the retarget (2.4b) are one PR
  (PR-E; Rulings 18/23), fence commit first, so no trunk commit carries the fenced script with the
  hazardous default; the script header carries the hazard warning (D-6).
- **Hot-loop purity extends to the cloud SDKs** (CHARTER §4 invariant 10). Task 5.4 adds `anthropic`
  and `claude_agent_sdk` to `tests/regression/test_hot_loop_import_purity.py::_FORBIDDEN_MODULES`
  (Ruling 22; D-14).
- **Status notes, not ticks.** Where a section's task has landed on a PR branch the "Lands in" line
  names the branch and SHA; ticks in `tasks.md` happen at PR merge. Dates, never weekday labels.
- **No mock of the device under test** in `tests/hardware/**`; unit/integration ESP32 tests use
  `tests/integration/test_f025_integration.py::FakeSerialPort`.
- **Ratchet budgets are at ceiling** (`noqa`, `type: ignore`, `hardcoded-ok`): no new marker; vendor
  constants go in `src/mousedroid/constants.py` or the schema.

---

## D-1 — The loopback motor smoke measures the chassis: two halves plus the overlay flag

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

**Decision (two halves, per R-1).**

1. **`tests/hardware/test_esp32_loopback.py::test_send_velocity_moves_encoders` gains the
   encoder-less branch and loses the skip on that branch.** On a chassis with
   `chassis_has_wheel_encoders=False` the criterion is a FEEDBACK_BASE_INFO (T:1001) frame after the
   command, witnessed by `EncoderReading.imu_valid` — a field the legacy codec never sets — polled up
   to `esp32.max_consecutive_timeouts` reads, mirroring `collect_esp32_probe` (D-3), with
   `emergency_stop()` in `finally` so the wheels stop whatever the assertion does (Ruling 33). A
   silent board **FAILs** after the last poll; it does not skip. `esp32.enabled=false` is a config
   decision (the factory returns
   `MockESP32Driver`), not a measurement, so that case alone skips, with the remedy ("stack the bench
   overlay") in the skip reason. The encoder-bearing branch keeps its historical >5 %-of-max velocity
   criterion. The test name is unchanged — `scripts/jetson_smoke_test.sh::test_motor` greps it.
2. **A new AQA reads the smoke script and pins the file ↔ test-name pairing** so the gate cannot
   silently drift to a module that does not carry the criterion.

Plus the **second motion gate** (plan review-focus 2): a module helper
`_commanded_velocity(settings, fraction)` returns `0.0` unless `esp32.smoke_test_allow_motion` is
True, and otherwise a velocity **bounded by `esp32.smoke_test_velocity_mps`** (default 0.05 m/s —
the setpoint `tests/hardware/test_motor_smoke.py` already honours; `0.0` locks all smoke motion even
with both gates open — Ruling 32); ~~(rev 2) 0.2 × and 0.3 × `max_velocity_mps`~~ → the fractions
(0.2 for `test_send_velocity_moves_encoders`, 0.3 for `test_emergency_stop_latency`) scale inside
that bound, never above it. With the Pydantic gate closed the stage still proves command acceptance,
the T:1001 frame and e-stop latency — at zero velocity.

**The overlay flag.** `chassis_has_wheel_encoders: false` is set on `config/jetson_production.yaml`
(task 2.1), not flipped as the schema default. ~~Rationale: `MockESP32Driver` reports velocity, so a
default flip would break the mock path.~~ **Refuted (round 1):**
`src/mousedroid/comms/mock_driver.py::MockESP32Driver.read_encoders` returns `EncoderReading()` —
zeros. The decision stands on **invariant 6 alone**: an existing field's default is a behaviour
change for every YAML that omits it, and the WAVE ROVER is one chassis, not the default chassis.

**Why.** Under stock firmware `WaveshareStockCodec.parse_encoders` echoes the commanded speed into
`L`/`R`, so a velocity assertion measures nothing on this rover — it would pass coincidentally
(R-1). The T:1001 frame is the only thing the board can prove it did.

**Pins inverted.** None in `tests/regression/` — the hardware module's own skip is removed on the
encoder-less branch. **Pins added** (task 1.2, in `tests/regression/test_f059_aqa.py`):
`test_motor_stage_runs_the_loopback_module_and_greps_its_test_name` (R-1 pin 1 — reads
`scripts/jetson_smoke_test.sh::test_motor`, asserts the module path, the `PASSED` grep and the
`esp32_stage_configs_csv` call) and
`test_loopback_motion_tests_branch_on_the_encoder_flag_and_the_motion_gate` (AST over the hardware
module: `chassis_has_wheel_encoders` and `imu_valid` in the velocity test; `_commanded_velocity(` in
both motion tests; `smoke_test_allow_motion` in the helper).

**Rejected.** Running `tests/hardware/test_motor_smoke.py` from the motor stage instead (it reads the
flag but is not what the F-008 `validation_command` greps; swapping the module would also change the
e-stop latency evidence). Making the skip a `xfail` (a silent board must be red, not yellow).

**Lands in** task 1.2 (PR-B — landed `14b83b1` and `57d6996` on `claude/f059-pr-b-gate-measures-board`;
a fix round for Rulings 31–33 is pending); the overlay flag in 2.1 (PR-D — landed `1263a5d`).
Asymmetry with the power stage is recorded in D-4 (R3-5).

## D-2 — The production overlay speaks the stock dialect at 115200

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

**Decision.** `config/jetson_production.yaml` gains `command_set: waveshare_stock`, `serial_baud:
115200` and `chassis_has_wheel_encoders: false` (D-1), keeping `enabled: false` and `serial_port`.
Env stays the per-host override, no longer the only way in.

**Wire frames** (unchanged from revision 1, confirmed all three rounds): velocity
`{"T":13,"X":<m/s>,"Z":<rad/s>}` (`CMD_ROS_CTRL`); poll `{"T":130}` → `{"T":1001,"L","R","r","p","y","v",…}`
(`CMD_BASE_FEEDBACK` → `FEEDBACK_BASE_INFO`, `src/mousedroid/comms/command_set.py` line 66);
heartbeat `{"T":136,"cmd":<ms>}` (`CMD_HEART_BEAT_SET`); battery is read from `v` in the T:1001 frame,
not written as a PID (S-11). **Derived baud:** `ESP32Config._apply_command_set_coupling` sets
`serial_baud = WAVESHARE_STOCK_BAUD` (115200, `src/mousedroid/config/schema/hardware.py` line 103)
when the field is still at the legacy default; the overlay states 115200 explicitly so the YAML reads
true on its own.

**Why.** Both factory images are `ugv_base_general` builds and neither contains the legacy protocol
(proposal C-1). A board that only ever spoke stock, polled at 1 Mbaud in a dialect it never knew, is
indistinguishable from a dead one — the misdiagnosis of 2026-05-31.

**Pins inverted** (task 2.1; each docstring declares the reversal):
`tests/regression/test_f025_backwards_compat.py::test_shipped_overlays_keep_legacy_baud_without_the_env_lever`
(parametrised to `("jetson_production.yaml", "waveshare_stock", WAVESHARE_STOCK_BAUD)`; `default.yaml`
stays legacy at 1 000 000),
`tests/regression/test_f025_backwards_compat.py::test_shipped_overlays_load_unchanged` and
`tests/regression/test_f036_backwards_compat.py::test_shipped_overlays_still_load` (both end with
`command_set == "waveshare_stock" if overlay in _STOCK_OVERLAYS else "legacy"`). The schema/legacy-YAML
pins in the same files are kept. **Pin added:**
`tests/regression/test_f059_aqa.py::test_production_overlay_speaks_stock_firmware_on_an_encoder_less_chassis`
(also asserts `esp32.enabled is False` — the flip is D-11's).

**Config-compat.** `python scripts/check_config_compat.py --platform jetson --changed-files
config/jetson_production.yaml` exits 0 because the pinned schema ignores the keys (R-9, D-9). No
`deployments/jetson-image.json` edit rides in PR-D (Ruling 29): task 4.5 re-pins to the SHA the rover
ran, after PR-H.

**Rejected.** Changing the schema default to stock (invariant 6; `default.yaml` and every other
overlay would flip). Keeping the env-only lever (the overlay would keep lying about the board it is
for).

**Lands in** task 2.1 (PR-D — landed `1263a5d` on `claude/f059-pr-d-config-flips`; docs `b770987`).

## D-3 — Preflight probes the board through the public driver protocol

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

**Decision.** A new `collect_esp32_probe(cfg) -> Esp32ProbeResult` in
`src/mousedroid/validation/runtime/{_esp32}.py` (exported from
`src/mousedroid/validation/runtime/__init__.py`) builds the driver through
`src/mousedroid/factory/hardware.py::build_esp32_driver`, `connect`s, polls `read_encoders()` up to
`esp32.max_consecutive_timeouts` times until a frame with `imu_valid` arrives (under
`waveshare_stock` that is a parsed T:1001 frame), reads `get_battery_voltage()` once (0.0 → `None`),
and **always** `disconnect`s. It never sends a motion command. Under `legacy` it raises `ValueError`
rather than fabricate — no polled frame exists to witness.
`src/mousedroid/validation/preflight.py::_check_esp32` becomes: `mock_hardware` → OK;
`esp32.enabled=false` → **WARN** with the remedy in `detail`; non-stock command set → **WARN** ("no
probe"); probe `responded` → OK with "FEEDBACK_BASE_INFO after N poll(s)"; otherwise **FAIL**.
Connect errors propagate and `run_preflight` records them as FAIL with the message in `detail`
(plan review-focus 4: a missing or unreadable device node must fail fast, not hang).
~~(rev 2) a port held elsewhere fails with a lock error~~ → **(fix round 1, after PR-C `48820c4`,
Ruling 28)** the driver takes **no exclusive lock until task 4.1** (`_open_serial` passes no
`exclusive=`; nothing in `src/` uses `flock` or `TIOCEXCL`), so there is no lock error to report yet.
The probe is therefore **run cold** — container stopped, or holding the mock as it does throughout
the bench — because a live container that owned the port would contend for the frames rather than
fail; the runbook sentence is "FAIL on silence or a connect error; run it cold".

~~`enabled=false` yields a `SKIP` outcome.~~ **R-4: `PreflightStatus` has `OK/WARN/FAIL/DEGRADED`
only, and this change adds no member.** `enabled=false` is WARN; the aggregate `DEGRADED` is the
documented meaning ("everything works but watch this"). A mocked board is never OK — reporting OK
there was S-4.

**Two fences for the trend timer (R-11).** `scripts/mousedroid-trend.service` runs `cli.preflight
--trend` on a timer with the container up. Its shipped `MOUSEDROID_TREND_CHECKS=config,host_env_keys`
is already safe (the unit's own comment: non-exclusive checks only); the gap is that nothing pins it.
Fence 1: the regression pin that the unit's check list never names `esp32` (and that the comment
survives). Fence 2: the exclusive open (D-13) makes any future collision loud rather than silent.
**One-time `--trend` downgrade:** `scripts/jetson_full_validation.sh` runs `cli.preflight --trend`
blocking, so the first run after this change sees `esp32 ok -> warn` and exits 1 once — and the
first run after the board is enabled also reports `esp32` *slowed*, its `elapsed_s` growing from
milliseconds to connect-plus-poll time; the runbook step re-runs once to re-baseline the journal and
says why (plan review-focus 6; `48820c4`).

**Pins inverted** (task 1.4; class docstring declares the reversal):
`tests/unit/validation/test_preflight_real_branches.py::TestEsp32RealBranch::test_driver_ok` (a
`SimpleNamespace()` driver — no `connect` — was OK; it is replaced) and
`::TestEsp32RealBranch::test_none_driver_fails` (replaced by the probe-shaped class:
`test_disabled_board_warns_never_ok`, `test_legacy_warns_without_probing`,
`test_silent_board_fails`, `test_answering_board_ok`,
`test_a_connect_error_fails_fast_with_the_reason` — ~~`test_a_port_held_elsewhere_fails_fast_with_the_reason`~~,
renamed in `48820c4` when the lock claim was dropped). **Pins added:** unit
`tests/unit/validation/{test_runtime_esp32_probe}.py` (fake driver through the factory seam — first
frame is a response; silent board bounded by `max_consecutive_timeouts`; disconnect on raise; 0.0 V
→ `None`; legacy refuses), integration
`tests/integration/{test_f059_esp32_probe_integration}.py` (`FakeSerialPort`: written frames are
`[136, 130, 130]`, never T:13; wrong-typed frames never count),
`tests/regression/test_f059_aqa.py::test_preflight_esp32_warns_rather_than_ok_when_the_board_is_mocked`,
and the trend-unit pin `tests/regression/test_f059_aqa.py::test_trend_unit_never_runs_exclusive_device_checks`
(R-11 — carried into 1.4; it reads **every** `Environment=MOUSEDROID_TREND_CHECKS=` line because
systemd honours the last assignment).

**Rejected.** Opening the port directly in preflight (bypasses the factory and the codec — the probe
must use the same path the orchestrator does). A `SKIP` enum member (R-4). Probing under `legacy`
by waiting for unsolicited frames (no bounded criterion).

**Lands in** tasks 1.3 and 1.4 (PR-C — landed `a70ce29`, `f3e6644`, `9439b14` and the fix round
`48820c4` on `claude/f059-pr-c-preflight-probe`). Docs: `docs/runbooks/jetson-full-validation.md`
line 158 ("construction-only" → "a probe, not a construction check … FAIL on silence or a connect
error … run it cold"), `docs/runbooks/jetson-rover-smoke.md` (the one-time downgrade row).

## D-4 — Battery thresholds re-armed; the gate is allowed to be stricter than the runtime

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

**Decision (CF-1).** `config/jetson_production.yaml` `safety:` gets `battery_warn_v: 10.5` and
`battery_critical_v: 9.5` (3S LiPo). `battery_implausible_below_v` stays 1.0, so the runtime routes a
0.0 V reading (the `get_battery_voltage` sentinel for "no reading") to log-only — a missing reading
can never e-stop a moving rover. `mock_battery_v=12.0` clears both thresholds, so mock paths are
unchanged.

**The R-6 decision, in writing.** With CF-1, a board that answers the poll **without** `v` reads
0.0 and `tests/hardware/test_power_chain_smoke.py` fails (`battery_voltage_v >= battery_critical_v`,
guarded only by `battery_critical_v > 0.0`, no lower floor). **This is desired:** the smoke gate may
be stricter than the runtime. The test gains no floor; a frame without voltage is a red `power` stage
(success criterion 3c).

**The R3-5 asymmetry, kept on purpose.** `tests/hardware/test_esp32_loopback.py::test_battery_voltage_in_plausible_range`
— in the module the **motor** stage runs — guards with `if v > 0.0` and passes on a missing reading.
Two tests in the same gate therefore disagree about 0.0 V, and that is correct: the motor stage is a
motion gate (did the chassis accept a command and answer?), the power stage is the telemetry gate.
One docstring sentence records it (task 1.2); no new test.

**Code before config (plan review-focus 3).** A rover still running `032942b` has no 1.0 V floor
(the field post-dates the pin), so installing these thresholds before trunk code trips
`battery_critical` on a missing reading. The bench ladder orders code first (3.3), and
`bash scripts/deploy_remote.sh --config-only` refuses a remote tree without the field (D-6).

**Pins inverted** (task 2.2):
`tests/regression/test_jetson_deploy_prep_backwards_compat.py::test_default_plus_production_overlay_stack_loads`
— it pins **three** production-overlay values (`battery_critical_v == 0.0`, `esp32.enabled is False`,
`min_valid_sensors == 1`); only the first inverts (to 9.5, and `battery_warn_v` 10.5, both asserted
against the raw YAML declaration because the values now equal the base and "override wins" cannot be
pinned by inequality); `esp32.enabled is False` **stays** (EX-1 / D-11); `min_valid_sensors == 1`
stays. **Pin added:**
`tests/regression/test_f059_aqa.py::test_production_overlay_arms_battery_thresholds_and_the_power_gate_has_no_floor`
(thresholds; `0.0 < battery_implausible_below_v < battery_critical_v`; the power test's assertion
line is present and floorless).

**Rejected.** Keeping 0.0 until a real voltage is observed (the alternative recorded in CF-1 — it
keeps the power stage vacuous for exactly the retest that needs it). Adding a floor to the power
test (turns a missing reading back into a green).

**Lands in** task 2.2 (PR-D — landed `906daf2`).

## D-5 — Latch ON in production, explicitly OFF on the lidar-only stack; the AQA loads the stack

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

**Decision (CF-2).** `config/jetson_production.yaml` sets `safety.emergency_latch.enabled: true`;
`config/jetson_lidar_only.yaml` sets `safety.emergency_latch.enabled: false` **explicitly**, with the
reason in a comment; the runbooks point operators at
`python -m mousedroid.cli.rearm --config /etc/mousedroid/jetson_production.yaml --status` then
`python -m mousedroid.cli.rearm --config /etc/mousedroid/jetson_production.yaml --operator <you> --confirm-area-clear`
— **the documented command names the production config** (Ruling 36): without `--config` the CLI
calls `load_settings()` with no path (`src/mousedroid/cli/rearm.py` line 92), loads `default.yaml`,
sees no latch and exits 0 ("nothing to clear"), a silent no-op on the production container; task 4.7
makes it fall back to `MOUSEDROID_CONFIG` and puts the path in the boot hint
(`docs/runbooks/emergency-stop-rearm.md`,
`docs/runbooks/jetson-rover-smoke.md`, `docs/runbooks/jetson-full-bringup.md`).

~~`jetson_lidar_only.yaml` keeps the latch off.~~ **Refuted (round 1, R-7) and fixed:** the shipped
validation stack is `jetson_production.yaml,jetson_lidar_only.yaml` (`scripts/jetson_validate.sh`
line 31), the deep merge inherits `true` from production unless lidar_only declares `false`, and
lidar_only's `lidar_unavailable_policy: "emergency"` with its 0.5 s grace would then latch on one
lost LiDAR acquisition. Revision 2 writes `false` explicitly — and **the AQA loads the stack, not the
file**, so the premise (the stack that would latch) and the conclusion (it does not) are both asserted
on the merged settings.

**Why ON in production.** The latch is the mechanism that keeps an e-stop asserted across ticks and
restarts until an operator clears it; the re-arm CLI and runbook now ship, so the operator-side
precondition for arming it is met (S-5 correction).

**Pins inverted** (task 2.3; module docstring declares the reversal):
`tests/regression/test_estop_latch_backwards_compat.py::test_every_shipped_overlay_still_loads_with_the_latch_off`
(parametrised over `config/*.yaml` minus `baselines.yaml` and `jetson_production.yaml`), with a new
`test_production_overlay_arms_the_latch_by_decision`. **Pins added:**
`tests/regression/test_f059_aqa.py::test_latch_is_armed_on_production_and_explicitly_off_on_the_lidar_only_stack`
(production `True`; stack `lidar_unavailable_policy == "emergency"` **and** latch `False`; raw
lidar_only YAML declares the key) and `::test_runbooks_point_operators_at_the_rearm_cli`.

**Rejected.** Leaving the latch OFF with the software watchdog only (CF-2's alternative). Relying
on inheritance (the exact defect R-7 found).

**Lands in** task 2.3 (PR-D — landed `345b504`; the `--config` clause is Ruling 36 from the PR-D
review, folded in before merge).

## D-6 — Deploy path: fence and retarget `rsync --delete` in one PR; git-fetch is the recorded fallback

*(reconstructed 2026-10-19 — original in the drafting-session zip; rewritten fix round 1 to the
fence PR-E landed — Rulings 18, 23, 24, 26, 27, 37)*

**Decision (CF-4, split per R-2 into two tasks, one PR).**

- **2.4a — the fence.** `scripts/deploy_remote.sh` gains a bash array `RSYNC_PROTECT_ARGS`, used by
  `rsync_code` and printed one-per-line by `--print-rsync-args` (handled before host resolution, so a
  fixture test can run the *script's own* arguments through a real `rsync -n --delete
  --itemize-changes` with no host). Its order is the design: `--filter='+ **/.gitkeep'` **first**,
  then `--filter=':- .gitignore'`, then the nine dev-tool names at any depth (`.git __pycache__ .venv
  *.egg-info node_modules .mypy_cache .pytest_cache .ruff_cache .claude`), then the seven runtime
  roots **anchored to the transfer root and spelt without a trailing slash** — `/venv /weights
  /models /logs /reports /estop_latch /tensorrt_cache`. Anchoring keeps a same-named directory deeper
  in the tree (a `reports` directory under `docs/`, a `models` directory under `src/mousedroid/`) shipping; no trailing slash keeps a root that
  is a symlink (`models` → NVMe) held rather than deleted, because a slash-pattern matches
  directories only and rsync does not count a symlink as one. The include rule precedes the merge
  because the merge drops `!` re-includes; it restores tracked `.gitkeep` sentinels wherever the
  ignore is file-level (`dir/*`) and nothing else. The literal `rsync -avz --delete \` stays
  immediately after `preserve_remote_wip` (pinned by `tests/regression/test_f051_aqa.py`), and the
  string `docker.env` must not appear anywhere in the script
  (`tests/regression/test_f051_backwards_compat.py::test_secrets_are_not_transferred_by_the_sync`).
- **2.4b — the retarget, same PR.** `REMOTE_SRC` defaults to `/opt/mousedroid` (the rover's repo root
  — proposal B-1), every camp-1 surface in B-2 moves with it, camp 2 (the container's package path
  `/opt/mousedroid/src` in `scripts/validate_compose_env.sh::_SOURCE_PATHS_IN_USE`) is **not**
  retargeted. `restart_service` restarts `mousedroid-docker` when enabled (the unit's `ExecStartPre`
  re-syncs the overlay and runs the fail-closed preflight), then the bare-metal unit if that is
  enabled too — and the mode is explicit, `MOUSEDROID_REMOTE_RESTART_MODE=unit|container` (landing in
  PR-E; Ruling 37): `unit` is the systemd restart, whose `ExecStop` is `docker compose down`
  (`scripts/mousedroid-docker.service` line 80), so the container is **recreated** and its writable
  layer — every hot-installed package — discarded; `container` is `docker restart mousedroid` in
  place, where bind-mounted code and the hand-synced overlay take effect and the writable layer
  survives — the right mode after bench step 3.3's hot-installs, and the safe one while a recreate
  can wedge the camera (2026-07-15). `run_health_check` **waits** (`wait_for_container`, polling
  once a second for up to `MOUSEDROID_REMOTE_HEALTH_WAIT_S`, default 30, `0` probes once, a malformed
  value is refused) for the pinned container name `mousedroid` to be listed and then execs the health
  check — a container that never appears is a **WARNING, never a silent skip** (Ruling 27).
  `--code-only` **never runs `deploy_jetson.sh`** (Ruling 26): with no venv it leaves the bare-metal
  installer alone, with neither a venv nor the Docker unit it warns and carries on; only `--full` runs
  the installer, and the header says not to run `--full` against the Docker-spine rover.
  **Code-before-config guard:** `deploy_config` first runs `require_remote_safety_floor`, which greps
  `battery_implausible_below_v` in the remote `src/mousedroid/config/schema/reward_safety.py` and dies
  "deploy code first" when it is absent (D-4).

**R3-3 — the default was a live hazard, not a nesting bug.** ~~(rev 2) The current default nests the
repo root inside the package parent.~~ → **(rev 2, after R3-3)** on a clean rover it `--delete`d
`/opt/mousedroid/src/mousedroid/` before writing the nested tree. ~~(rev 2) The script header and
both runbooks say do not run against the rover with default paths until 2.4b lands.~~ → **(fix round
1, Rulings 18/23)** 2.4a and 2.4b land as **one PR** (PR-E), fence commit first, so no trunk commit
ever carries the fenced script with the old default and there is no window to warn about; the
script header beside `REMOTE_SRC` carries the warning as history ("it used to be the package parent
one level down … `--delete` removed the live package directory beneath it"), the old literal may not
survive anywhere in the script as a fallback (the inverted pin asserts
`MOUSEDROID_REMOTE_SRC:-/opt/mousedroid/src}` is absent), and `config/docker.env.example` joins camp
2 with its clarifying parenthetical.

**R3-4 — negations are dropped, so the pin needs two directions.** ~~(rev 2) Pin 2:
`test_delete_never_touches_runtime_state` — `rsync -n --delete` with the script's own flags never
lists a gitignored path.~~ → **(rev 2, after R3-4)** two directions: the negative one (every runtime
root survives, symlinked or not, and `--delete` is proven still live by a stale tracked file being
listed) and the positive one (over a fixture checkout, every `git ls-files` path outside the fenced
roots appears in `--itemize-changes` as transferred). ~~(this reconstruction) The `.gitkeep` under a
negated `.gitignore` entry does not transfer; that one loss is accepted in writing rather than
papered over with an `--include`.~~ → **(fix round 1, Ruling 24)** false on two counts: the landed
fence *does* ship tracked sentinels (the `+ **/.gitkeep` rule ahead of the merge — the fixture's
`cfg/runtime/.gitkeep` transfers while the generated file beside it stays PC-side), and the loss is
not one file but the documented set — anything tracked under a fenced root (`reports/endurance/.gitkeep`,
the two tracked `reports/*.json`) plus every tracked file `.gitignore` ignores or re-includes with
`!` (`.vscode/*.json`, `.agents/skills/approval-gate-wiring/SKILL.md`,
`docs/3D_printing_files/README.md`, `docs/planning/SMOKE_REPORT_TEMPLATE.md`): **eight** tracked
files on this tree (proposal B-7), accepted in the script header. The bundle follows the landed
code.

**Git-fetch delivery is the recorded fallback — and the bench's primary.** On the rover:
`chown -R <user>:<user> /opt/mousedroid` (ownership drift), preserve WIP with
`bash scripts/rover_wip_guard.sh preserve … --archive -`, record the rollback commit and tag the image,
`git fetch origin && git checkout --detach <SHA>`, prove `import mousedroid.main` in the container,
**then** `scripts/sync_jetson_overlay.sh`, then restart **in place** (`docker restart mousedroid` —
a unit restart would recreate the container and drop the packages hot-installed on the way, Ruling
37). `docs/runbooks/pc-to-jetson-promotion.md` §1 documents it with the bench-step commands.
Replacing rsync entirely was the rejected alternative: the fenced rsync is the PC-side path once PR-E
has merged and the fixture proves it.

**Pins inverted** (2.4b):
`tests/regression/test_f051_backwards_compat.py::test_remote_path_defaults_are_the_previous_literals`
(→ `REMOTE_SRC="${MOUSEDROID_REMOTE_SRC:-/opt/mousedroid}"` and the old literal absent as a fallback;
docstring names R3-3). Re-pointed, not inverted:
`tests/unit/scripts/test_deploy_remote_guard.py::test_a_plain_absolute_path_is_still_accepted`
(payload `/opt/mousedroid`). **Pins added** (2.4a, real rsync, skipped where rsync is absent —
ubuntu-latest ships it): `tests/unit/scripts/test_deploy_remote_guard.py::test_delete_never_touches_runtime_state`,
`::test_a_symlinked_runtime_root_is_held_too`, `::test_every_tracked_file_outside_the_fenced_roots_is_transferred`,
`::test_runtime_root_excludes_are_anchored`, `::test_the_shim_recorded_the_same_rsync_args_the_script_prints`;
(2.4b) `::test_code_only_restarts_the_docker_unit_when_enabled`,
`::test_code_only_skips_the_docker_unit_when_absent`,
`::test_config_only_refuses_a_remote_tree_without_the_safety_floor`,
`::test_config_only_installs_config_when_the_remote_tree_has_the_safety_floor`,
`::test_code_only_without_a_venv_leaves_the_bare_metal_installer_alone`,
`::test_code_only_with_neither_a_venv_nor_the_docker_unit_warns_and_carries_on`,
`::test_the_health_check_waits_for_the_container_to_be_listed`,
`::test_the_health_check_warns_when_the_container_is_never_listed`,
`::test_a_malformed_health_wait_is_refused`, `::test_the_remote_container_is_the_one_compose_pins`;
the restart-mode pins ride with `MOUSEDROID_REMOTE_RESTART_MODE` when it lands.

**Rejected.** Retargeting first, or in a separate later PR (R-2: deletes `venv/`, `weights/`,
`models/` on first deploy; Ruling 18: a trunk window with the fenced-but-mistargeted script).
Trusting `require_safe_remote_path` (syntax, not blast radius — B-4). A per-line `--include` for
every negated `.gitignore` entry (one generic `+ **/.gitkeep` rule ahead of the merge was adopted
instead — it restores sentinels, not arbitrary re-includes, and the rest is documented as not
shipping). Falling through to `deploy_jetson.sh` when `--code-only` finds no venv (the Docker-spine
rover has none; the bare-metal installer is `--full`'s). Probing the container once right after the
restart (compose has not listed it yet; the check would be skipped in silence). A single hard-wired
restart (the unit restart recreates; the in-place restart skips the overlay sync and preflight —
the operator must choose, so the mode is a variable).

**Lands in** tasks 2.4a then 2.4b as one PR — PR-E, landed `3a6a551`, `f76b2c3` (2.4a) and
`1a43e3b`, `7679361` (2.4b) on `claude/f059-pr-e-rsync-fence`; the restart mode is still landing;
ticks at merge.

## D-7 — The evidence record

*(reconstructed 2026-10-19 — original in the drafting-session zip; shape confirmed unchanged by all
three rounds)*

**Decision.** Bench results become tracked artifacts under `reports/esp32_bench/<date>/`: the smoke
run's `SUMMARY.md` and per-stage output copied from the rover's `reports/jetson_smoke/<stamp>/`,
the probe transcripts (3.5/3.6/3.7/3.8), `f008-check-<stamp>.txt` (the `validate.py --check F-008`
output), and `decision.md` — the decision-tree leaf taken, the commands run, the frames seen, and the
firmware provenance ("stock as shipped, unflashed" or the app sha256 written). `.txt`/`.md` only —
`*.log` is gitignored (`.gitignore` line 112). `hw-evidence-auditor` must trace every hardware claim
in `decision.md` to a file in that directory before PR-H merges.

**F-008 closes only on that record.** `status: "done"`, `implemented_in` = the **full hex SHA the
rover ran** (never a branch name — `.claude/skills/feature-closeout/SKILL.md`), `notes` += evidence
path + firmware provenance. Side effect: the `.claude/workforce.yaml` freeze on `src/mousedroid/arm/**`
self-disables. If the ladder ends without a T:1001 frame, F-008 stays `todo` and `notes` points at
`decision.md`; Phases 4 and 5 proceed regardless.

**Pins inverted (PR-H, Ruling 21).** Flipping F-008 to `done` turns two doc-reconciliation pins red:
`tests/regression/test_doc_reconciliation_aqa.py::test_arm_claude_md_path_exemption_lapses_when_f008_is_done`
(once F-008 is `done` it asserts `KEY_FILES_PATH_EXEMPTIONS` is empty) and
`::test_arm_claude_md_path_exemption_is_enumerated_not_a_prefix` (asserts the set equals
`{"src/mousedroid/arm/CLAUDE.md"}`). PR-H therefore empties `tests/_claude_md.py::KEY_FILES_PATH_EXEMPTIONS`
and fixes the Key Files list of `src/mousedroid/arm/CLAUDE.md` so every named path resolves (the
exemption was hiding a real failure — `::test_arm_key_files_exemption_is_not_vacuous` pins that and
returns early once the set is empty), declaring both reversals in the docstrings (the second becomes
"the set is empty and still not a prefix"). The arm doc sits under the freeze: PR-H edits
`features.yaml` first so `freeze_gate` has self-disabled before `src/mousedroid/arm/CLAUDE.md` is
touched, or sets `MOUSEDROID_WORKFORCE_ALLOW_FROZEN=1` and says so in the PR.

**Rejected.** Citing MCP `health_check` / `esp32_diagnostics` / `benchmark_latency` output as
evidence — they are stubs that always report OK until task 4.3. Closing F-008 on a passing
`validate.py --check` alone (a 600 s timeout would hide a real pass; the direct script exit 0 is
recorded instead and the catalog command narrowed in PR-H).

**Lands in** bench steps 3.10f–g and task 4.4 (PR-H).

## D-8 — The bench ladder

*(reconstructed 2026-10-19 — original in the drafting-session zip; ladder confirmed unchanged by
all three rounds, with the R3-2 annotation on checkpoint A; landed as
`docs/runbooks/esp32-bench-ladder.md` at `5b2cf71`, the authoritative step list — `tasks.md` Phase 3
mirrors it)*

**Decision.** Cheapest-first, cold-to-warm, dual-gated. Steps 3.0–3.10 in
`docs/runbooks/esp32-bench-ladder.md` (mirrored in `tasks.md`); the decision tree:

| Leaf | Evidence | Action | Record |
|---|---|---|---|
| L1 answers on USB-C (3.6) | `{"T":1001,…}` | no flash; 3.10a–g | "stock firmware as shipped; F-008 closed at `<sha>`" |
| L2 silent on USB-C, answers on ttyTHS1 (3.7) | 3.7 reply | try the secondary CP2102N bridge by pinning `esp32.serial_port` in the bench overlay; answers → continue 3.10; else production over `/dev/ttyTHS1` needs a compose device + `serial_port` + `usbc_discovery` `required: false` change — a PR, not a bench hack | "ESP32 alive; CP2102N/USB-C path faulty; repair = bridge rework or PCB swap" |
| L3 silent on both, esptool syncs (3.8) | `Chip is ESP32…` | 3.9 (CF-5 re-flash, dry run then `--write`); re-run 3.5–3.6; still silent after a verified flash → UART-path fault → replace PCB | "ROM alive, app absent/corrupt; reflashed `<app sha256>`" or "replace" |
| L4 no sync | `Failed to connect` after the BOOT retry | multimeter: 3.3 V at the module — rail absent → regulator/PSU; rail present → module/PCB dead | "replace General Driver board" |

**Checkpoint A runs against the mock (R3-2).** "Healthy container, no chassis" proves the brain and
the rollback anchor, not the board: the container's own stack keeps `esp32.enabled: false`, so its
health check reads `MockESP32Driver`. ~~(rev 2) The opted-out `power` stage reads the mock (12.0 V)
and passes, recorded as such.~~ → **(fix round 1, Ruling 31)** with the bench overlay explicitly
opted out (`MOUSEDROID_SMOKE_BENCH_ESP32_OVERLAY=`) the `motor` and `power` stages record **SKIP
without running pytest** — a mock PASS would be a green that measures nothing — and with the overlay
stacked a test that skips is a stage **FAIL** (`57d6996`). The checkpoint's substance is the health
check and the `camera` stage. The board is first measured at 3.5/3.6, with the bench overlay
forwarded per command. After step 3.3's git-fetch delivery the container is restarted **in place**
(`docker restart mousedroid`): the unit restart's `ExecStop` is `docker compose down`, which
recreates the container and discards every package hot-installed in that step (Ruling 37).

**Time box.** Session 1 (bench day 1 — the runbook dates its evidence directory
`reports/esp32_bench/<date>/` 2026-10-20) = 3.1–3.9 + the decision; session 2 (the following day) =
one repair attempt + re-run of 3.5–3.10. Hard 3 h stop per session; no 1001 frame by the end of
session 2 → order the replacement board, F-008 stays `todo`. Dates, not weekday labels, everywhere
in this bundle.

**Rejected.** Flashing before the passive and active probes (CF-5's gate: silent on **both** ports
**and** esptool syncs). Running the smoke stages with the container stopped during the bench (not needed —
the container holds the mock; cold mode is a Phase-4 mechanism for after production flips).

**Lands in** Phase 3 (operator + agents; PR-H).

## D-9 — Re-pin ordering: after merge, before branch cleanup, for reachability

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

**Decision.** ~~Re-pin `deployments/jetson-image.json` before the config flips so the config-compat
gate validates them against a schema that knows the keys.~~ **Inverted by R-9:** the pinned
`032942b` schema is plain `BaseModel` without `extra="forbid"`, so `scripts/check_config_compat.py`
*ignores* unknown keys and already passes them — the re-pin is **not** a precondition for the flips.
The re-pin happens **after** the bench SHA merges and **before** any branch cleanup, for
**reachability**: tag it (`bash scripts/repin_tags.sh`, then `--push` with explicit operator
confirmation), then `bash scripts/archive_stale_branches.sh` and the `pin-reachability-audit` skill.
No PR-D commit touches the record (Ruling 29); 4.5 re-pins to **the SHA the rover ran** — the one
PR-H's `implemented_in` names.

**R3-7 — the first strict run.** Trunk's `src/mousedroid/config/schema/_primitives.py`
`ConfigDict(extra="forbid")` post-dates the pin, so re-pinning to a trunk SHA makes the gate strict
for the first time and any stale key in `config/*.yaml` surfaces at the re-pin, far from the change
that introduced it. **Pre-check** before pinning: substitute the candidate SHA into a temporary copy
of the record and run `python scripts/check_config_compat.py --platform jetson --deployments-dir
<tmp> --changed-files config/*.yaml`; record the result in the bundle.

**The record's narrative.** `notes` drops the "NOT an ancestor of the default branch" sentence
(false since at least 2026-10-16); states whether the image was rebuilt or still carries the
`9c31968` layers plus writable-layer packages from bench step 3.3; keeps `model_sha256` /
`ort_provider` null.

**Pins inverted** (task 4.5):
`tests/regression/test_f051_backwards_compat.py::test_the_record_sha_was_not_repinned_in_this_change`
(→ the new SHA; docstring "post-merge trunk commit, ancestor of the default branch").

**Follow-up filed (not this change):** make the gate pin a schema that forbids extras so unknown keys
are caught at the change that introduces them.

**Lands in** task 4.5 (PR-K, after PR-H).

## D-10 — Firmware record and the flash script

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

**Decision (CF-5).** A tracked record `deployments/{esp32-firmware}.json` pins the four-part stock
image set by offset, size and sha256 (proposal table C: bootloader `0x1000`, partitions `0x8000`,
boot_app0 `0xE000`, `General_Driver.ino.bin` `0x10000`, `connect_baud` 115200, the factory esptool
flags, the `WAVE_ROVER_V0.9` alternate, `combine/target.bin` under `never_flash`, and the
post-flash `{"T":900,"main":2,"module":2}` frame). The images themselves stay out of git (`*.bin`
gitignored) and are staged under `firmware/*.bin`; the sha256 is authoritative, the file name a
convenience. On the rover they are staged under `/opt/mousedroid/firmware/` — inside the
bind-mounted checkout, because the container never mounts the home directory and `docker exec`
cannot see `~/firmware` (Ruling 20). `scripts/flash_esp32.sh` is rewritten: `--port <dev> [--bin-dir] [--record] [--baud]
[--write]`; **dry run by default** — verify size + sha256 of all four parts against the record,
print the factory `write_flash` and `verify_flash` argv, exit 0 without touching esptool; `--write`
runs `write_flash` then `verify_flash`; a size or hash mismatch refuses **before** any `esptool.py`
call; `esptool` is never installed automatically. `scripts/jetson_qspi_update.sh` is deleted (only
self-referenced; exits 0 on R36; unreachable tail). `docs/deployment.md` stops instructing a flash
as a bring-up step.

**Why the four-part layout.** The repo script writes one file at `0x0` at 460800 and names a binary
that was never committed (C-2). The factory flashed four parts at 115200 with `verify=True` 1 283
times; the record encodes exactly that recipe so a repair re-flash reproduces the factory state, not
an approximation of it.

**Pins amended** (task 2.5): `tests/regression/test_f051_aqa.py::test_no_parallel_deploy_manifest_was_introduced`
(allows the second per-platform record — one record per platform artifact;
`scripts/check_config_compat.py`, `scripts/repin_tags.sh`, `scripts/archive_stale_branches.sh` read
`deployments/jetson-image.json` by name, so the new file is invisible to them);
`tests/unit/scripts/test_scripts.py::TestFlashEsp32Sh` (re-pinned to the new CLI: `--port is
required`, the record path, 115200, dry-run default). **Pins added:**
`tests/unit/scripts/{test_flash_esp32}.py` (dry run verifies every part and prints the argv without
invoking the esptool shim; hash mismatch refuses before esptool; `--write` makes exactly two shim
calls; the tracked record is the factory image set; `bash -n` + strict mode).

**Rejected.** Committing the binaries (gitignored by policy; hash-pinned instead). Keeping the
single-image script with a new path (wrong layout, wrong baud, no verification). Flashing
`combine/target.bin` (unidentified, smaller than either app).

**Lands in** task 2.5 (PR-G — landed `cee6e9e`, `dad0ac4`, `8866101` on `claude/f059-pr-g-firmware`;
a security fix round is pending).

## D-11 — The bench overlay (EX-1): the only place `esp32.enabled` becomes true

*(reconstructed 2026-10-19 — original in the drafting-session zip; this section did not exist in
revision 2 — round 3 R3-2 required the flip to be named, and EX-1 ratified where)*

**Decision.** ~~`esp32.enabled` flips on `config/jetson_production.yaml` so the smoke stages reach
the board.~~ *(revision 2 never named a flip; R-1 and R-6 presupposed one — R3-2)* → A new overlay
`config/{jetson_bench_esp32}.yaml` carrying **exactly one key**, `esp32.enabled: true`, stacked
**after** production via `MOUSEDROID_JETSON_CONFIGS` by the two smoke stages that must talk to the
board. `scripts/jetson_smoke_test.sh` gains `esp32_stage_configs_csv()`: unset
`MOUSEDROID_SMOKE_BENCH_ESP32_OVERLAY` → append the repo default overlay to the stage's config CSV
(no double append); set-but-empty → the two ESP32 stages record **SKIP without running pytest** (the
no-chassis checkpoint uses this — Ruling 31; a mock PASS would measure nothing); any path → that
file; the singular `MOUSEDROID_JETSON_CONFIG` is the base when the CSV is empty (`4706338`). With the
overlay stacked, a test that skips (board configured off, pytest exit 0) is a stage **FAIL**:
`test_power` requires `test_power_chain_within_budget PASSED` exactly as `test_motor` requires its
own test (`57d6996`). `test_motor` and `test_power` pass the result as `MOUSEDROID_JETSON_CONFIGS`;
`test_pytest` keeps the orchestrator's stack (stacking there would open the port for every module).
The motor reason grep is widened to the new failure texts (`esp32.enabled=false`, `no
FEEDBACK_BASE_INFO`, `serial device … not found`).

**Why an overlay and not the production flip or an env lever.** Production keeps `false` until
F-008 is `done`, so the orchestrator in the running container keeps the mock and holds no port
(which is also why R-5's collision is a post-closeout condition — D-13). Both existing pins
(`test_default_plus_production_overlay_stack_loads` `esp32.enabled is False`; the F-059 AQA) stay
green. An env lever would have to be forwarded through the docker-exec wrapper, and the wrapper
forwards **no `MOUSEDROID_ESP32__*` variable** — so the Pydantic motion gate can only come from YAML,
which is exactly the property the second motion gate relies on (plan review-focus 2). The overlay
therefore **never** sets `smoke_test_allow_motion`; the pin asserts it stays False.

**Pins added** (task 1.1): `tests/regression/test_f059_aqa.py::test_bench_overlay_flips_exactly_esp32_enabled_on_top_of_production`
(raw YAML == `{"esp32": {"enabled": True}}`; `bench.esp32.model_copy(update={"enabled": False}) ==
prod.esp32`; `bench.safety == prod.safety`) and `::test_bench_overlay_leaves_the_motion_gate_closed`;
`tests/unit/scripts/test_jetson_smoke_orchestrator.py::test_esp32_stages_stack_the_bench_overlay`
(~~`test_motor_stage_stacks_the_bench_esp32_overlay`, four CSV cases~~ → the landed name: parametrised
over both stages, `motor` and `power`, × eight CSV cases through a stub python — default-base,
csv-appended, already-stacked, opt-out, singular-base, csv-beats-singular, override-path,
opt-out-empty-csv), `::test_power_stage_fails_unless_its_test_passes` and
`tests/regression/test_f059_aqa.py::test_power_chain_test_skips_when_the_board_is_configured_off`
(`57d6996`). The new overlay is picked up by every `config/*.yaml` glob
(`tests/regression/test_config_overlays_load.py`, `tests/regression/test_lidar_interlock_overlay_aqa.py`,
`tests/regression/test_estop_latch_backwards_compat.py`) and must load with the latch off and legacy
defaults on its own.

**Rejected.** The production flip (inverts a pin for a bench-only need and arms the orchestrator's
port before F-008 is proven). Forwarding env through the wrapper (opens the motion gate from outside
YAML).

**Lands in** task 1.1 (PR-B — landed `bbbda25` and `4706338` on `claude/f059-pr-b-gate-measures-board`;
the SKIP-without-pytest opt-out is in the pending 1.2 fix round). Bench use: `BENCH_CFG=…/jetson_production.yaml,…/jetson_bench_esp32.yaml`
forwarded per `docker exec`; nothing is copied to `/etc/mousedroid`.

## D-12 — Catalog edge and the F-050/F-051 deferral (EX-2, EX-3)

*(reconstructed 2026-10-19 — original in the drafting-session zip; landed `739494e`, `c7f62b1`,
`ab65226`)*

**Decision.** ~~F-059 `depends_on: ["F-008"]`.~~ → **F-059 `depends_on: ["F-001"]`; `"F-059"`
appended to F-008's `depends_on`** (R3-1 / EX-2). F-059's Phases 1–3 are what make F-008 passable, so
F-008 cannot be a prerequisite of F-059; the edge as drafted was semantically inverted and left
F-059 `blocked` in `src/mousedroid/harness/spec.py::select_next`. F-050 and F-051 are set
`status: "deferred"` with the note "Phases 5–6 stood down after the task-2.1 ceiling gate; resume
after F-008" (EX-3).

**R3-1 with the plan's correction.** ~~"`select_next` will return it (critical beats F-050's
high)."~~ → `select_next` returns any `in_progress` feature first as `resume`, choosing by
`PRIORITY` then id; F-059 is registered `in_progress` and `critical`, so **priority, not the
deferral, is the selector lever** — the selector already preferred F-059 over the in-progress F-050
(high) and F-051 (medium) once the edge was fixed. EX-3 is ratified for **catalog honesty** (two
features that are not being worked must not read `in_progress`), not as a selector fix. Output at
`ab65226`: `F-059 … (in_progress — resume)` (proposal §4).

**Pins added** (task 0.1): `tests/regression/test_f059_aqa.py::test_f059_is_reserved_and_critical`,
`::test_f059_edge_points_the_right_way`, `::test_select_next_returns_f059_while_it_is_in_progress`
(skips once F-059 leaves `in_progress` or while another critical feature is in progress — the edge
pin is the durable guarantee); `tests/regression/test_f059_backwards_compat.py::test_f050_f051_are_deferred_with_a_reason_or_resumed`
(deferred with the reason, or legitimately resumed — never silently `todo`/`blocked`). The plan's
`test_no_other_feature_changed_status` (a trunk-SHA catalog snapshot) was **retired** in `c7f62b1`:
it would go red on any later legitimate catalog edit and skip on depth-1 clones.

**Rejected.** Keeping `depends_on: ["F-008"]` (F-059 blocked forever). Leaving F-050/F-051
`in_progress` (the selector would keep saying "resume F-050" the moment F-059 is no longer
`in_progress`).

**Lands in** task 0.1 (PR-0) — landed.

## D-13 — Exclusive serial open; warm and cold smoke-run modes (Phase 4)

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

**Decision.** ~~Task 1.6: open the port with `exclusive=True` in Phase 1.~~ → **Moved to Phase 4
(R-5), coupled with the cold-vs-warm decision for the smoke run.**

- **4.1 — exclusive open everywhere a port is opened.**
  `src/mousedroid/comms/serial_driver.py::_open_serial` passes `exclusive=_SERIAL_EXCLUSIVE`
  (`Final[bool] = True`, beside `_SERIAL_IO_WORKERS`, same "not a config field" rationale: two openers
  must collide loudly; POSIX `flock`, no-op on Windows). The probe (D-3) inherits it through the
  factory driver. `scripts/jetson_smoke_test.sh::test_serial` adds `exclusive=True` to its own
  `serial.Serial(...)` so a held port prints `FAIL: Serial error: Could not exclusively lock port`
  instead of a "no response" PASS — and "no response" itself becomes a FAIL: the serial stage must be
  a probe. S-9 closes here. Until this lands neither the driver nor the preflight probe holds a lock —
  PR-C `48820c4` removed the claim from the probe's docs and tests (Ruling 28) — so the probe is run
  cold.
- **4.2 — the smoke run's two modes.** `scripts/jetson_full_smoke_run.sh` runs its stages via
  `docker exec` in the running container. `MOUSEDROID_SMOKE_ESP32_MODE=warm` (default): if the
  container's own settings have `esp32.enabled` and not `mock_hardware`, the ESP32 stages record FAIL
  "warm smoke refused: the orchestrator owns the ESP32 port; re-run with
  `MOUSEDROID_SMOKE_ESP32_MODE=cold`". `cold`: `docker stop` under an EXIT trap that always `docker
  start`s (the `scripts/jetson_full_validation.sh::restart_container_on_exit` pattern) and a
  `docker compose run --rm --no-deps` wrapper. Both modes forward `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION`
  so the documented lever reaches the container (today the wrapper forwards no `MOUSEDROID_ESP32__*`
  var — the runbook's "zero-velocity by default" sentence is corrected in 4.6).

**Conditional on the production flip (R3-2).** With EX-1 the container keeps the mock through the
bench, so the orchestrator holds no port and the docker-exec'd stages with the bench overlay open it
freely — the collision R-5 describes exists only once production enables the board after closeout.
4.2's warm-mode refusal is what detects that state; 4.1 is what makes it loud instead of silent.
Warm-with-mock is therefore **sanctioned**: it proves the brain, and says so.

**A sibling Phase-4 follow-on (4.3).** The MCP `esp32_diagnostics` stub becomes real and stays
non-actuating: when the orchestrator holds the (now exclusive) port the handler reads through the
**live driver** and never reopens the port; otherwise it runs `collect_esp32_probe`; it never calls
`send_velocity`/`emergency_stop` and stays out of `actuation_tools`.

**Pins inverted** (4.3): `tests/unit/common/test_tool_registry.py::test_dispatch_esp32_diagnostics` —
a bare `create_default_registry()` asserted `status == "esp32_ok"`; it becomes `unconfigured` and the
docstring declares the reversal. **Pins added** (4.1)
`tests/unit/comms/test_serial_driver.py::test_open_serial_requests_exclusive_access`
(`Serial.assert_called_once_with(port, baud, timeout=…, exclusive=True)`); `tests/unit/scripts/test_smoke_harness.py`
pins for the serial-stage FAIL on silence; (4.2) `tests/unit/scripts/test_smoke_harness.py` pins for
`MOUSEDROID_SMOKE_ESP32_MODE`, the forwarded `-e MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION`, and "warm
smoke refused"; (4.3) `tests/unit/common/test_tool_registry.py` (`unconfigured` / `mock` / live-driver
`ok` with `send_velocity`/`emergency_stop` never awaited / probe branch; the registry stays at 11
tools).

**Rejected.** Exclusive open in Phase 1 (R-5: the smoke run would collide with itself before the
mode decision existed). A config field for exclusivity (two openers colliding is never a tunable).
Reopening the port from the MCP tool (defeats the exclusive open).

**Lands in** tasks 4.1–4.3 (PR-I, PR-J).

## D-14 — Hot-loop purity extends to the cloud SDKs (added fix round 1, Ruling 22)

*(added 2026-10-19 — not a revision-2 section; it records a ruling from the in-repo review of this
bundle)*

**Decision.** `tests/regression/test_hot_loop_import_purity.py::_FORBIDDEN_MODULES` gains
`"anthropic"` and `"claude_agent_sdk"` beside the #242 names (`google.adk`, `honcho`, `composio`,
`composio_core`), so importing `mousedroid.orchestrator.autonomous` or `mousedroid.factory.mission`
loads neither cloud SDK (CHARTER §4 invariant 10: the 30 Hz loop stays LLM-free). Today it holds by
construction — `src/mousedroid/llm_gateway/anthropic_gateway.py` imports the SDK lazily inside
`start()`, and WS-B's reviewer lives under `tools/`, never under `src/` — and the widened tuple makes
the property a pin rather than a habit (proposal §2 D-5: the #242 pin never named `anthropic`).

**Pins added** (task 5.4): the widened tuple, plus
`tests/regression/{test_f059_llm_retarget_aqa}.py` asserting both names are present.

**Rejected.** A second purity test for the cloud SDKs (one tuple, one pin). Importing the SDK at
module scope in the gateway "because it is installed on the rover" (it is hot-installed into the
writable layer and gone on a recreate; a lazy import degrades, a top-level one crashes).

**Lands in** task 5.4 (PR-N).
