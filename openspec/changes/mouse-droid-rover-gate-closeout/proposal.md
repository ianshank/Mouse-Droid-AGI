# Change: `mouse-droid-rover-gate-closeout`

**F-number:** F-059 (reserved in `features.yaml` by task 0.1 at `739494e`; F-009–F-014 and F-033
remain burned holes per ADR-013)
**Status:** proposed — revision 2, reconstructed, with the round-3 review and fix round 1
(2026-10-19, after the in-repo adversarial review of `3735c66`) folded in
**Branch:** `claude/f059-bundle` @ `5b2cf71` (fix round 1 on top)
**Compared against:** trunk `f57d99b` — the SHA all three review rounds read — on the repository's
default branch `claude/markdown-implementation-plan-aVJ2l` (there is no `main`).

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

> **Provenance.** Revisions 1 and 2 of this file were drafted in a read-only proxy clone and never
> pushed; the only bundle file that reached this repository is `peer-review.md` (three rounds). This
> document is reconstructed from that record — every table row, decision and criterion below is one
> the review names by identifier — and from the implementation plan the operator approved on
> 2026-10-17 (`create-a-plan-to-moonlit-ember.md`, kept in the operator's plans directory, not
> tracked here). Where the original wording is unrecoverable, the decision is reconstructed from the
> review's disposition column and says so in a parenthetical. Revision-2 wording that round 3
> overturned is struck through in place with the replacement beside it. Every line number below was
> re-read in this worktree on 2026-10-19 and is written outside the backticks so the path checker
> validates the file, not the line.

---

## 1. Problem

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

The project has been blocked on a dead Wave Rover ESP32 since 2026-05-31. Catalog feature F-008
("USB-C rover smoke passes on the physical Jetson") is still `todo`, and the capability freeze
(`.claude/workforce.yaml` `freeze.feature_key: F-008`) parks the arm platform, the T3+ training arc
and every autonomy stream behind it. The bench session is dated 2026-10-20 (the evidence directory in
`docs/runbooks/esp32-bench-ladder.md`). **The gate cannot pass as shipped**,
for reasons that have nothing to do with whether the board is alive:

1. **The motor stage does not measure the chassis.** `scripts/jetson_smoke_test.sh::test_motor`
   runs `tests/hardware/test_esp32_loopback.py`, whose velocity test has no encoder-less branch and
   `pytest.skip`s on near-zero feedback (A-1). Under stock firmware the codec echoes the commanded
   speed into `L`/`R`, so the assertion would pass without a measurement.
2. **Every ESP32 stage runs against the mock.** `config/jetson_production.yaml` ships
   `esp32.enabled: false`; `src/mousedroid/factory/hardware.py::build_esp32_driver` returns
   `MockESP32Driver` whenever the flag is off, regardless of `mock_hardware` (A-6, round 3 R3-2).
   The motor stage is structurally unpassable and the power stage is a green that measures nothing.
3. **Preflight calls a dead or absent board OK** — `_check_esp32` names the driver type and never
   connects (A-3; S-4).
4. **The safety thresholds cannot fire.** Battery thresholds are zeroed on the production overlay
   (A-7; S-3) and the e-stop latch that landed in #232 ships OFF (A-8; S-5).

**The misdiagnosis at 1 Mbaud.** The ESP32 was declared dead with the overlay's `serial_baud:
1000000` and the legacy `{"T":1,"vx","vy","omega"}` command set. Neither factory firmware image
contains that protocol (`"vx"` ×0 in both — table C), so a live stock board and a dead one are
indistinguishable at that setting. The overlay still carries no `command_set` key and the 1 Mbaud
line (A-4), so the heartbeat failsafe is never armed (S-8) and the battery poll is a PID write on
stock firmware (S-11). The first thing the bench must do is retest at the stock dialect and
115200 — which is why the software has to be right first.

**The deploy path is a hazard, not a nesting bug.** `scripts/deploy_remote.sh` with default paths
would delete the live package on a clean rover (B-1, R3-3), and the planned retarget without a fence
would delete the venv, the LLM model and the ONNX weights (B-2b, R-2). The config-compat pin in
`deployments/jetson-image.json` is protected by no tag and ignores unknown keys (B-5, B-6).

**The catalog pointed the wrong way.** F-059 as drafted (`depends_on: ["F-008"]`) was `blocked` in
`select_next`, and the in-progress F-050/F-051 pre-empted it anyway (R3-1; fixed by task 0.1).

Initial-research findings cited by number below *(reconstructed — the S-numbering comes from the
initial research answer, share `c10181d7`; only the items the review and plan cite by number are
recoverable)*:

| S | Finding | Where it lands |
|---|---|---|
| S-3 | battery thresholds zeroed on the production overlay — no voltage can e-stop | CF-1 / D-4 / 2.2 |
| S-4 | preflight `esp32` check reports a dead or absent board OK | D-3 / 1.3–1.4 |
| S-5 | emergency-stop latch — **landed in #232**, ships OFF (§6 correction 1) | CF-2 / D-5 / 2.3 |
| S-8 | heartbeat failsafe never armed under `legacy` | D-2 / 2.1 |
| S-9 | serial port opened non-exclusively — two openers collide silently | D-13 / 4.1 (stays open through Phase 3) |
| S-11 | battery poll under `legacy` is a PID write on stock firmware | D-2 / 2.1 |

**Intended outcome.** Before the bench session every software precondition for a *valid* ESP32 retest
is on trunk with its regression pins, so bench time is bench time. The bench ladder then either flips F-008
to `done` (lifting the arm freeze and the T3+ arc) or produces a hash-pinned repair-vs-replace
decision. In parallel, the API credit funds reviewer and batch-triage work that F-008 keeps off the
bench (§5).

**Posture (CHARTER §3).** No autonomous motion. Every motion step is dual-gated —
`MOUSEDROID_SMOKE_ALLOW_MOTION=1` **and** `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true` — with the
rover on blocks and the operator at the power switch. No cloud agent ever drives the rover; WS-D
forbids actuation tools in every agent toolset.

## 2. Verified findings

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

Every row was verified at `ab65226` in this worktree (tables A–C) or at `5b2cf71` (table D and the
fix-round-1 rewrites, 2026-10-19); the "Review" column records the round that confirmed, refuted or
amended it. Rows the review refuted are kept, struck through, so they are not re-raised. Every
strike-through names the revision its wording belongs to and the round that overturned it —
~~(rev N) …~~ → **(rev M, after R-x)** … — so a struck cell is never read as the current claim
(fix round 1).

### A. The gate

| # | Finding | Evidence (by symbol; line numbers outside backticks) | Review |
|---|---|---|---|
| A-1 | ~~(rev 1) The F-008 motor stage is gated by `chassis_has_wheel_encoders`; an encoder-less chassis skips it.~~ → **(rev 2, after R-1)** the motor stage runs the loopback module, greps two test names, and that module never reads the flag. Only `tests/hardware/test_motor_smoke.py` does, and the motor stage does not run it. The velocity test skips on near-zero feedback; under stock firmware `WaveshareStockCodec.parse_encoders` echoes the commanded speed into `L`/`R`, so it would pass coincidentally — asserting a measurement never made. | `scripts/jetson_smoke_test.sh` line 481 `test_file=…/tests/hardware/test_esp32_loopback.py`; 494 `grep -q "test_send_velocity_moves_encoders PASSED"`; 504 `grep -q "test_emergency_stop_latency PASSED"`; `tests/hardware/test_esp32_loopback.py` 115–135 (`test_vel = max_vel * 0.2`, `pytest.skip` at 133), no `chassis_has_wheel_encoders` in the module | R-1 accepted; holds round 3 |
| A-2 | The F-008 gate is SKIP-green without motion. | `scripts/jetson_smoke_test.sh` line 819 `exit "${FAILURES}"`; `record_skip` (82) increments `SKIPS` only; `test_motor` records SKIP unless `MOUSEDROID_SMOKE_ALLOW_MOTION=1` (468) | confirmed; holds round 3 |
| A-3 | `_check_esp32` is a non-probe (S-4): it builds the driver and returns OK with its type name, never calling `connect`. `PreflightStatus` has `OK/WARN/FAIL/DEGRADED` only, DEGRADED aggregate-only. | `src/mousedroid/validation/preflight.py` 320–331 (`return _ok("esp32", f"driver={type(driver).__name__}", …)`); `PreflightStatus` at 42–48 | confirmed; R-4 (no `SKIP` member) accepted |
| A-4 | The production overlay still speaks `legacy` at 1 Mbaud, so `heartbeat_enabled` is inert (S-8): the heartbeat frame `{"T":136,"cmd":<ms>}` belongs to the stock dialect. | `config/jetson_production.yaml` line 58 `serial_baud: 1000000`, no `command_set` key; `src/mousedroid/comms/command_set.py` 64 (`CMD_HEART_BEAT_SET`); `ESP32Config.heartbeat_enabled` description + `src/mousedroid/validation/preflight.py::_check_esp32_failsafe` docstring | confirmed; holds round 3 |
| A-5 | The legacy battery poll `{"T":2}` is a PID write on stock firmware (S-11); the stock query is a read. 12.0 V mock and 0.0 V both stay clear of e-stop. | `src/mousedroid/comms/command_set.py::LegacyCommandCodec.battery_query` docstring vs `WaveshareStockCodec.battery_query` ("a READ, unlike the legacy PID-write"); `ESP32Config.mock_battery_v = 12.0`; `tests/regression/test_f025_backwards_compat.py::test_battery_implausibility_floor_defaults_on_and_backwards_compatible` pins `battery_implausible_below_v == 1.0`; `src/mousedroid/comms/base_driver.py::get_battery_voltage` returns `0.0` on `None` | confirmed; holds round 3 |
| A-6 | **The shipped production overlay routes every ESP32 smoke stage to the mock** (round 3 R3-2; *the revision-2 row at this number is unrecoverable and is replaced by the finding that supersedes it*). As shipped the motor stage reads mock zeros, skips, and `test_motor` records FAIL "encoder loopback inactive" — unpassable whatever the firmware does; the power stage reads `mock_battery_v=12.0` and passes against the mock; `_check_esp32` reports OK for the mock (this *is* S-4). R-1's "passes coincidentally" and R-6's "fails `power` on missing voltage" both presuppose a real driver. | `config/jetson_production.yaml` line 56 `enabled: false`; `src/mousedroid/factory/hardware.py::build_esp32_driver` returns `MockESP32Driver` whenever `not cfg.esp32.enabled`; `tests/_jetson_hardware.py::load_jetson_runtime_settings` forces only `mock_hardware=False`; `src/mousedroid/comms/mock_driver.py::MockESP32Driver.read_encoders` returns `EncoderReading()` (zeros). The pin that would invert on a production flip: `tests/regression/test_jetson_deploy_prep_backwards_compat.py::test_default_plus_production_overlay_stack_loads` (`esp32.enabled is False`) — EX-1 keeps it green by flipping in a bench overlay instead (D-11) | R3-2 accepted |
| A-7 | Battery thresholds are zeroed on the production overlay (S-3), so the power stage's assertion is vacuous as shipped. | `config/jetson_production.yaml` 13–14 `battery_critical_v: 0.0` / `battery_warn_v: 0.0`; `tests/hardware/test_power_chain_smoke.py` 34–35 guards on `battery_critical_v > 0.0` then asserts `battery_voltage_v >= battery_critical_v` with no lower floor; pinned by `test_default_plus_production_overlay_stack_loads` (`battery_critical_v == 0.0`) | confirmed; R-6 decided (D-4) |
| A-8 | The e-stop latch landed in #232 and ships OFF (S-5). Production carries no `emergency_latch` key; the lidar-only overlay mentions it only in a comment while setting the policy that would latch; the shipped validation stack is the two-file deep merge, which inherits whatever production declares. | `config/jetson_production.yaml` — no `emergency_latch` key; `config/jetson_lidar_only.yaml` line 58 `lidar_unavailable_policy: "emergency"`, 63 comment only; `scripts/jetson_validate.sh` line 31 `REMOTE_CONFIGS_CSV` defaults to `jetson_production.yaml,jetson_lidar_only.yaml` | R-7 accepted (D-5) |
| A-9 | The trend timer runs `cli.preflight` with the container up; its check list is operator-editable and nothing pins `esp32` out of it (R-11). The shipped default is already safe — the gap is the missing pin. | `scripts/mousedroid-trend.service` line 33 `Environment=MOUSEDROID_TREND_CHECKS=config,host_env_keys` with the "non-exclusive checks" comment at 4–6; line 50 runs `mousedroid.cli.preflight … --trend` | R-11 partially accepted (D-3) |

### B. The deploy path

| # | Finding | Evidence | Review |
|---|---|---|---|
| B-1 | **The current default deletes the live package on a clean rover** (R3-3). `REMOTE_SRC` defaults to `/opt/mousedroid/src` and `rsync_code` syncs `${PROJECT_DIR}/` into it with `--delete`. The rover's repo root is `/opt/mousedroid`; `/opt/mousedroid/src/` holds only the `mousedroid/` package and the local repo root has no top-level `mousedroid/`, so `--delete` removes `/opt/mousedroid/src/mousedroid/` before writing the nested tree. Only `scripts/rover_wip_guard.sh` refusing on a dirty rover (exit 3) has prevented it. **The fence (2.4a) and the retarget (2.4b) land as one PR (PR-E; Rulings 18/23), fence commit first, so no trunk commit ever carries the fenced script with this default; the script header beside `REMOTE_SRC` records the hazard as history (landed `1a43e3b`).** | `scripts/deploy_remote.sh` line 38 `REMOTE_SRC="${MOUSEDROID_REMOTE_SRC:-/opt/mousedroid/src}"`, 334 `rsync_code()`, 344 `rsync -avz --delete`; `scripts/mousedroid.service` 25 `WorkingDirectory=/opt/mousedroid`, 31 `ExecStartPre=/opt/mousedroid/scripts/preflight_check.sh`; `docker-compose.jetson.yml` 80 binds `/opt/mousedroid:/opt/mousedroid`; `docs/superpowers/specs/2026-06-02-jetson-claude-pilot-deploy-design.md` records the container importing from `/opt/mousedroid/src` via the editable `.pth`; `smoke-reports/smoke_report.md` recorded the drift ("Plan v2 assumed `/opt/mousedroid/src/scripts/` — actual is `/opt/mousedroid/scripts/`") | R-2 accepted; R3-3 raised the severity |
| B-2 | **Inventory, two camps** (rewritten per R-8; completed per R3-3). **Camp 1 — means the rover repo root; retargeted to `/opt/mousedroid` by 2.4b:** `scripts/deploy_remote.sh::REMOTE_SRC` (38) and its header/help; `scripts/jetson_validate.sh::REMOTE_SRC` (28; its fallback loop at 98–109 stays); `docs/operator/JETSON_SMOKE_RUNBOOK.md` 16–17 (`cd /opt/mousedroid/src`); `docs/runbooks/jetson-claude-pilot-deploy.md` 37–44 (cross-link only — its `/opt/mousedroid/src/mousedroid/...` verification path is the package path and is correct); `docs/runbooks/pc-to-jetson-promotion.md` §1 (gains the git-fetch fallback); `tests/unit/scripts/test_deploy_remote_guard.py::test_a_plain_absolute_path_is_still_accepted` (662; payload re-pointed from `/opt/mousedroid/src` to `/opt/mousedroid` so the "legitimate override" it names is the real one — does not invert); `tests/regression/test_f051_backwards_compat.py::test_remote_path_defaults_are_the_previous_literals` (162; **inverts**). **Camp 2 — means the container's package path `/opt/mousedroid/src`; NOT retargeted:** `scripts/validate_compose_env.sh::_SOURCE_PATHS_IN_USE` (89: `/opt/mousedroid/src /opt/mousedroid/weights /opt/mousedroid/config /opt/mousedroid/models` — the mount targets a cache directory must neither cover nor nest inside); `tests/unit/scripts/test_validate_compose_env.py` (pins that list); `config/docker.env.example` line 140 ("nor covering or nested inside /opt/mousedroid/src, weights, config or models" — gains a clarifying parenthetical, nothing else). ~~(rev 1, this row) `scripts/rover_wip_guard.sh` exits 4 because rsync excludes `.git`.~~ → **(rev 2, after round 1)** refuted and removed from this row: `resolve_toplevel` runs `git -C <target> rev-parse --show-toplevel`, which resolves from any subdirectory of a work tree. | as listed; `scripts/rover_wip_guard.sh::resolve_toplevel` | round 1 refuted the exit-4 sub-claim; R-8 accepted; R3-3 added `config/docker.env.example` |
| B-2b | **Blast radius of retargeting without a fence** (added per R-2; phrasing corrected per R3-6). `rsync_code` has **nine** excludes — `.git __pycache__ .venv *.egg-info node_modules .mypy_cache .pytest_cache .ruff_cache .claude` — none of which protects `venv/` (the service interpreter: `REMOTE_VENV="${REMOTE_INSTALL_DIR}/venv"`), `weights/`, `logs/` or `models/`, all gitignored and so invisible to the WIP guard. `reports/` is gitignored only in sub-trees, not wholesale; the blast radius is the same, the revision-2 phrase "all gitignored" was not. | `scripts/deploy_remote.sh` 345–353 (nine `--exclude` lines), 49 `REMOTE_VENV`; `.gitignore` 42 `venv/`, 85 `weights/`, 113 `logs/`, 261 `models/`, 173 `reports/jetson_validate/`, 176–177 `reports/jetson_smoke/**`, 336 `reports/dead_code/`, 346–347 `reports/endurance/*` + `!reports/endurance/.gitkeep` | R-2 accepted; R3-6 corrected |
| B-3 | The WIP guard is the only thing that has stood between B-1 and a clean rover: `preserve_remote_wip` runs before the sync, and `scripts/rover_wip_guard.sh` refuses a dirty target (exit 3) so `rsync --delete` never runs over unpreserved work; `--confirm-dirty` archives the work off-rover and commits it to `rover/wip-<date>` first. *(row wording reconstructed, fix round 1 — the record says B-3 survived every round and its revision-1 text is unrecoverable; the exit-4 claim this reconstruction first put here belonged to B-2, where it now sits struck)* | `scripts/deploy_remote.sh` header ("A dirty target REFUSES the sync") and `::preserve_remote_wip`; `scripts/rover_wip_guard.sh` exit 3, as R3-3 cites it | survives rounds 1–3 |
| B-4 | `require_safe_remote_path` (landed by F-052) validates syntax — IFS characters, backslashes, globs, shell metacharacters — not blast radius. A syntactically safe path can still be the wrong directory for `--delete`. | `scripts/deploy_remote.sh::require_safe_remote_path`; `tests/unit/scripts/test_deploy_remote_guard.py` payload classes | confirmed |
| B-5 | The config-compat pin ignores unknown keys (R-9) and is about to become strict (R3-7). `deployments/jetson-image.json` pins `032942b50ab71abf285282a4de7333193f208c38`; at that SHA there is a single `schema.py` module under `src/mousedroid/config/` with plain `BaseModel`s and no `extra=`, so `scripts/check_config_compat.py` already passes `command_set`, `emergency_latch` and `battery_implausible_below_v`. Trunk's `src/mousedroid/config/schema/_primitives.py` `ConfigDict(extra="forbid")` post-dates the pin; the first re-pin makes the gate strict for the first time, so any stale key in `config/*.yaml` surfaces there, far from the change that introduced it. *(amended round 1 R-9, round 3 R3-7)* | `deployments/jetson-image.json` line 4; `git show 032942b --stat`; `scripts/check_config_compat.py` docstring describes trunk, not the pin | R-9 accepted (D-9); R3-7 accepted (4.5) |
| B-6 | The pin is reachable only by luck: no tag contains `032942b` (`git tag --contains` → none), so a branch cleanup could orphan the SHA the config-compat gate checks out, and the record's `notes` still claims the SHA is "NOT an ancestor of the default branch" — false since at least 2026-10-16 (`git merge-base --is-ancestor` → true). Task 4.5 re-pins to the SHA the rover ran and tags it (`bash scripts/repin_tags.sh`) before any `bash scripts/archive_stale_branches.sh` run (D-9). *(row wording reconstructed, fix round 1 — the record says B-6 survived round 1; this reconstruction had put round-3 content under the number, now B-7)* | `deployments/jetson-image.json` line 9; `git tag --contains 032942b50ab71abf285282a4de7333193f208c38` → nothing, re-run 2026-10-19; `scripts/repin_tags.sh` | survives rounds 1–3; re-pin ordering per R-9 (D-9) |
| B-7 | *(added round 3, R3-4; rewritten fix round 1 to the fence PR-E landed — Ruling 24)* `--filter=':- .gitignore'` drops every `.gitignore` negation: the `-` merge modifier parses exclude patterns only, `!` is a list-clearing rule with no argument, and rsync has no negation syntax; `.claude/` negations are moot because `.claude` is excluded wholesale. The landed fence (`claude/f059-pr-e-rsync-fence` `f76b2c3`) therefore (1) anchors the seven runtime-root excludes to the transfer root and spells them without a trailing slash — `/venv /weights /models /logs /reports /estop_latch /tensorrt_cache` — so a same-named directory deeper in the tree (a `reports` directory under `docs/`, a `models` directory under `src/mousedroid/`) still ships and a root that is a symlink (`models` → NVMe) is held rather than deleted; (2) places `--filter='+ **/.gitkeep'` **ahead** of the merge, so a tracked sentinel ships wherever the ignore is file-level (`config/prometheus/runtime/*`) while a generated file beside it stays PC-side; (3) documents, in the script header, the tracked files that still do not ship — **eight** on this tree at `5b2cf71`: three under the fenced `reports/` root (`drift_comparison.json`, `spike_step_distillation.json`, `endurance/.gitkeep` — rsync never descends into an excluded directory) and five that `.gitignore` ignores or re-includes with `!` (`.agents/skills/approval-gate-wiring/SKILL.md`, `.vscode/settings.json`, `.vscode/tasks.json`, `docs/3D_printing_files/README.md`, `docs/planning/SMOKE_REPORT_TEMPLATE.md`); the 33 tracked `.claude/**` files never ship at any depth by the dev-tool exclude. Accepted in writing; the positive-direction pin asserts every other tracked path transfers. ~~(this reconstruction, rev 2) `reports/endurance/.gitkeep` is the one negation loss~~ → false: see (3). | rsync(1) FILTER RULES; `.gitignore` 20, 47, 95, 290, 346; the PR-E header comment (FENCED / SHIPS / DOES NOT SHIP); `tests/unit/scripts/test_deploy_remote_guard.py::test_every_tracked_file_outside_the_fenced_roots_is_transferred`, `::test_runtime_root_excludes_are_anchored`, `::test_a_symlinked_runtime_root_is_held_too`; the count reproduced 2026-10-19 by emulating the merge over `git ls-files` with the `!` lines dropped | R3-4 accepted (2.4a); Ruling 24 |

### C. Firmware — the Waveshare factory package (CF-5 evidence)

Read 2026-10-16 from the operator's `WAVE_ROVER_FACTORY-25*` folder: a snapshot of Waveshare's
**factory flashing station**, not an attempt log of ours — `flash_download_tool_3.9.5/` with 1 283
per-board logs dated 2023-07 → 2025-06, all `Download result: PASS`, none from 2026.

| # | File (under `flash_download_tool_3.9.5/bin/`) | Bytes | SHA-256 | Role |
|---|---|---|---|---|
| C-1a | `General_Driver.ino.bin` | 986 928 | `ebedf216d23402381a16f44373721c8f2d90dfe6ef020cb1f33f73c8f95ee4a4` | stock `General_Driver` app — the dialect `WaveshareStockCodec` speaks (T=13/130/136/1001); written in 996 of 1 283 factory flashes |
| C-1b | `WAVE_ROVER_V0.9.ino.bin` | 986 768 | `80db6e52137590ec70bbb075d2de2a67f7ac3d32ffa42667a6649cfcc02fbb29` | older WAVE ROVER firmware (272 flashes); the tool's saved panel config currently selects **this** image |
| C-1c | `General_Driver.ino.bootloader.bin` = `WAVE_ROVER_V0.9.ino.bootloader.bin` | 18 992 | `644de0067047e22380034b8989c39e5d2882f7538c698788866ca5130427322e` | bootloader @ `0x1000` |
| C-1d | `*.partitions.bin` (identical for both) | 3 072 | `148b959cbff1c38aa8e1d5c0ba9d612c54997b945e56a63f41223eef650653a1` | partition table @ `0x8000` |
| C-1e | `boot_app0.bin` | 8 192 | `f94c5d786a7a8fab06ac5d10e33bf37711a6697636dc037559ea19cc410a17f0` | OTA data @ `0xE000` |
| C-1f | `combine/target.bin` | 896 192 | `3d5a7534c7c700f8e94cb7c7fd6188ac9db8c16b8242fd261a80dda75e1f9e00` | unidentified combined image (smaller than either app) — **never flash** |

| # | Finding | Evidence | Review |
|---|---|---|---|
| C-1 | **Which firmware is on the rover.** The factory flashed `WAVE_ROVER_V0.9` from 2024-03 to 2024-05 and switched to `General_Driver` in 2024-07; every flash from 2024-07 through 2025-06 (the newest log) wrote the 986 928-byte image. A rover bought in 2026 almost certainly runs `General_Driver`. Both app images carry identical JSON-key fingerprints (`"T":` ×58, `"L":` ×6, `"R":` ×5, `"X":` ×4, `"Z":` ×1, `heart` ×6, `UGV` ×9) — two builds of the same `ugv_base_general` codebase, so `WaveshareStockCodec` is the right dialect for either. The repo's legacy `{"T":1,"vx","vy","omega"}` protocol appears in **neither** image (`"vx"` ×0, `"lv"` ×0): a stock board has never understood it, which is exactly the F-025 finding. Factory recipe (logs + `configure/esp32/multi_download.conf`): connect at **115200**; esptool `before=default_reset after=no_reset compress=True flash_size=keep flash_mode=keep flash_freq=keep verify=True`; four writes `0x1000` bootloader, `0x8000` partitions, `0xE000` boot_app0, `0x10000` app. | factory logs; strings over both images | confirmed (plan CF-5 evidence) |
| C-2 | `scripts/flash_esp32.sh` writes **one** file at `0x0` at 460800 baud and expects `firmware/waverover_mousedroid.bin`, which was never committed (`*.bin` is gitignored). It must be rewritten for the four-part layout with per-part size + sha256 verification, dry-run by default; the images stay out of git and are pinned by hash in a tracked record (D-10). | `scripts/flash_esp32.sh` line 3 (usage names the missing binary), 28 `--baud 460800 write_flash 0x0`; `.gitignore` 90 `*.bin` | plan 2.5 |
| C-3 | `scripts/jetson_qspi_update.sh` is referenced only by its own usage line; it exits 0 on R36 (the rover's release) and its unconditional `reboot` makes the rest unreachable. Deleting it breaks no test or link. | repo-wide grep: one hit, line 10 of the file itself | confirmed; holds round 3 |
| C-4 | `docs/deployment.md` line 8 instructs flashing as a bring-up step with a `<path-to-firmware>.bin` argument. Flashing is **not** required — the board runs stock firmware; re-flash is repair-only (decision tree leaf L3). | `docs/deployment.md` 8; `README.md` 286–291 | plan 2.5 step 4 |

### D. LLM and credit surfaces (Phase 5 evidence)

*(reconstructed, fix round 1 — the review record names rows D-1…D-7 as surviving round 1 and quotes
only D-5's wording ("#242 uncalled; purity pin lacks `anthropic`"); the other six rows are the Phase-5
evidence the plan relies on, verified on disk at `5b2cf71` and numbered to the record's count. This
table was missing from the first reconstruction.)*

| # | Finding | Evidence | Review |
|---|---|---|---|
| D-1 | The Anthropic gateway always sends a sampling parameter: `_create_message` passes `temperature=self._cfg.temperature` on every call, and the SDK is imported lazily inside `start()`. A bare model-id swap to a generation that rejects non-default sampling parameters would flip `_degraded=True` and serve the fallback while the overlay names the primary — the gap WS-A's `AnthropicRequestConfig` closes (§5). The 400-on-`temperature` behaviour of the 5.5 generation is [Likely]: quoted from the `claude-api` skill's model table (cached 2026-10-06), not re-verified against the live API. | `src/mousedroid/llm_gateway/anthropic_gateway.py` 173 (`import anthropic` inside `start`), 301–327 (`_create_message`, `temperature=self._cfg.temperature`) | survives rounds 1–3; sourced fix round 1 (finding 12) |
| D-2 | There is no cloud-VLM seam for the credit to exercise (§6 correction 2). The only VLM seam is local and mocked: `src/mousedroid/factory/mission.py::build_vlm_progress` returns `None` unless `mission.vlm_progress_enabled` and otherwise wraps `src/mousedroid/reward/vlm_progress.py::MockVLMProgress` in `VLMProgressHead` ("A real VLM backend (HF-hosted, BLIP-2, …) is a separate sprint"); `VLMProgressBackend` is a `Protocol` with that one implementation; the gateway's `_create_message` takes strings, never frames. | `src/mousedroid/factory/mission.py` 42–72; `src/mousedroid/reward/vlm_progress.py` 44 (`VLMProgressBackend`), 65 (`MockVLMProgress`), 103 (`VLMProgressHead`) | survives; symbol cited fix round 1 (finding 8) |
| D-3 | `reports/dead_code/` is gitignored, so WS-C's input chain is local-only: `python scripts/dead_code_audit.py` regenerates the findings file and the triage document records its sha256 and the vulture version (§6 correction 3). | `.gitignore` 336; `scripts/dead_code_audit.py` | survives rounds 1–3 |
| D-4 | Five surfaces name the model id and must move together on a retarget or the golden goes red: both overlays' `model_name`, the metrics sample seed `src/mousedroid/telemetry/metrics/registry.py::generate_metrics_sample`, the golden populate script `tests/regression/_render_prometheus_populate.py` and the golden `tests/regression/fixtures/render_prometheus_golden.txt` — all `claude-haiku-4-5` today. | `config/jetson_production.yaml` 96; `config/jetson_claude_pilot.yaml` 37; `registry.py` 268–273; `_render_prometheus_populate.py` 64–69; golden 174–175, 198 | survives rounds 1–3 |
| D-5 | **#242 uncalled; purity pin lacks `anthropic`.** PR #242 ("external agent integration", `ea1d6b9`) added the cloud-agent seams — `src/mousedroid/factory/agents.py` (`build_mission_decomposer`, `build_memory_mirror`, `build_cloud_tool_adapter`), `src/mousedroid/agents/adk_adapter.py`, `src/mousedroid/agents/composio_adapter.py`, `src/mousedroid/memory/honcho_mirror.py` — and nothing outside `src/mousedroid/factory/` calls the three builders, so the credit cannot exercise them; the purity pin #242 shipped, `tests/regression/test_hot_loop_import_purity.py::_FORBIDDEN_MODULES`, names only `google.adk`, `honcho`, `composio`, `composio_core`, so a hot-loop `import anthropic` (or `claude_agent_sdk`, once WS-B adds it) would pass. **Still live at `5b2cf71`. Ruling 22:** task 5.4 adds `"anthropic"` and `"claude_agent_sdk"` to the tuple (CHARTER §4 invariant 10, hot-loop purity; D-14). | `git show ea1d6b9 --stat`; a repo-wide grep for the three builders outside `src/mousedroid/factory/` → none; `tests/regression/test_hot_loop_import_purity.py` 13–18 | **CONFIRMED round 1; still live; closed by Ruling 22 in 5.4** |
| D-6 | Neither cloud SDK is a declared dependency: `pyproject.toml`'s `llm` extra is `llama-cpp-python` + `huggingface-hub`, and no extra names `anthropic` or `claude-agent-sdk`; the gateway degrades rather than crashes when the SDK is absent (the lazy import), and on the rover the SDK is hot-installed into the writable layer. WS-B adds an `agent_reviewer` extra rather than widening `llm`. | `pyproject.toml` 100–103 (`llm`), no `agent_reviewer` / `claude-agent-sdk` entry; `anthropic_gateway.py` 173 | survives rounds 1–3 |
| D-7 | `tools/llm_latency_probe.py` is argparse-driven and writes no JSON at all, so a latency run leaves no tracked artifact and cannot prove which backend served; WS-A adds `--report-json` with `served: primary\|fallback`. `RegexInjectionFilter` (`src/mousedroid/security/injection_filter.py`) guards NL leaving the rover toward actuation, not reviewer output — which is why WS-B's `policy.py` records why it is not applied there. | `tools/llm_latency_probe.py` 46, 238–256 (argparse; no `json` import); `src/mousedroid/security/injection_filter.py` | survives rounds 1–3 |

## 3. Scope

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

**In scope** (`tasks.md` Phases 0–5; `design.md` D-1…D-13):

- Phase 0 — the catalog edge, the F-050/F-051 deferral, this bundle, the key rotation gate, the
  execution environment.
- Phase 1 — the F-008 gate measures the board: bench overlay (D-11), encoder-less criterion and
  second motion gate (D-1), preflight probe (D-3).
- Phase 2 — the configuration ships: stock dialect (D-2), battery thresholds (D-4), latch (D-5);
  the deploy path is fenced and retargeted in one PR, with a code-before-config guard (D-6); the firmware is
  hash-pinned and the flash script made verify-then-dry-run (D-10).
- Phase 3 — the bench ladder (D-8; landed as `docs/runbooks/esp32-bench-ladder.md`) and the evidence
  record (D-7); F-008 closes on evidence
  only.
- Phase 4 — exclusive serial open and warm/cold smoke modes (D-13), a real non-actuating
  `esp32_diagnostics`, the re-pin (D-9), the narrative sweep.
- Phase 5 — the credit workstreams WS-A…WS-D (§5).

**Out of scope.**

- Autonomous motion of any kind; any cloud agent holding an actuation tool (CHARTER §3).
- The physical repair or replacement of the General Driver board — the ladder produces the
  *decision*; the operator executes it.
- Production over `/dev/ttyTHS1` (decision-tree leaf L2 needs a compose device + `serial_port` +
  `usbc_discovery` change — a PR of its own, not a bench hack).
- The arm replanner's model retarget (`src/mousedroid/arm/planning/llm_replanners/anthropic_backend.py`
  lives under the F-008 freeze; deferred, still served).
- Rebuilding the Jetson image as a goal. Phase 3 rebuilds only if `import mousedroid.main` fails
  after checkout (risk R4); the re-pin records which happened.
- Making the config-compat gate pin a schema that forbids extras (follow-up filed from R-9; the
  pre-check in 4.5 is the mitigation).
- A cloud-VLM seam (§6 correction 2).

## 4. Decisions — CF-1 … CF-5, EX-1 … EX-4

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

Ratified by the operator on 2026-10-17. "Alternative" is recorded so a reversal is a one-line
edit, not a re-plan. Round 2 left CF-1 through CF-5 unratified; round 3 (2026-10-16) still read
them as open — the ratification post-dates the review record and is recorded here and in the
one-line note appended to `peer-review.md`.

| ID | Decision | Ratified | Alternative (not taken) |
|---|---|---|---|
| CF-1 | Un-zero battery thresholds on the production overlay | **Yes** — `battery_warn_v: 10.5`, `battery_critical_v: 9.5` (3S LiPo); `battery_implausible_below_v` stays 1.0; the smoke `power` stage becomes strict (R-6, D-4) | keep 0.0 until a real voltage is observed on the bench |
| CF-2 | E-stop latch | **Yes** — `emergency_latch.enabled: true` in `config/jetson_production.yaml`; explicit `false` in `config/jetson_lidar_only.yaml` (R-7); `python -m mousedroid.cli.rearm --config /etc/mousedroid/jetson_production.yaml …` in the runbooks (D-5; Ruling 36 — without `--config` the CLI calls `load_settings()` with no path, loads `default.yaml`, sees no latch and exits 0) | leave OFF; software watchdog only |
| CF-3 | Rover login | **Taken as routine** — the scripted default `MOUSEDROID_REMOTE_USER` stays `jetson` (invariant 6); runbooks instruct exporting the rover's real login (`smoke-reports/smoke_report.md`) | change the script default (inverts the F-051 literal pin for a per-host fact) |
| CF-4 | Deploy path | **Yes** — fence `rsync --delete` (2.4a) and retarget the `REMOTE_SRC` default to `/opt/mousedroid` (2.4b) in **one PR**, fence commit first (Rulings 18/23); git-fetch delivery is the recorded fallback **and the first delivery method used on the bench** (D-6) | replace rsync with `git fetch` entirely |
| CF-5 | Firmware | **Yes** — stock `General_Driver` (sha256 `ebedf216…`) + shared bootloader/partitions/boot_app0 pinned in a tracked record `deployments/{esp32-firmware}.json`; flash only if silent on USB-C **and** the ttyTHS1 jumper **and** esptool syncs (D-10, leaf L3) | repair/replace the driver PCB without reflash |
| EX-1 | `esp32.enabled` flip (R3-2) | **Yes** — new bench overlay `config/{jetson_bench_esp32}.yaml` carrying only `esp32.enabled: true`, stacked via `MOUSEDROID_JETSON_CONFIGS` by the motor/power stages; production keeps `false` and both pins stay green (D-11) | flip the production overlay; or forward env through the docker-exec wrapper |
| EX-2 | F-059 ↔ F-008 edge (R3-1) | **Taken as a correctness fix** — F-059 `depends_on: ["F-001"]`; `"F-059"` appended to F-008's `depends_on` (D-12; landed `739494e`) | keep `depends_on: ["F-008"]` (F-059 stays `blocked` forever) |
| EX-3 | F-050 / F-051 (in progress, ONNX delivery) | **Yes** — `status: "deferred"` with the note "Phases 5–6 stood down after the task-2.1 ceiling gate; resume after F-008" (landed `739494e`) | leave `in_progress` |
| EX-4 | Execution method | **Subagent-driven** (`superpowers:subagent-driven-development`) with the house agents as gates | workflow fan-out; native single session |

**Catalog state after task 0.1** (R3-1 asked for the selector output to be pasted into the bundle;
run 2026-10-19 at `ab65226` with `python scripts/select_next.py`):

```
F-059  Rover gate closeout: valid ESP32 retest preconditions on trunk  (in_progress — resume)
```

~~"`select_next` will return it (critical beats F-050's high)."~~ **Corrected:** the drafted edge
`depends_on: ["F-008"]` made F-059 `blocked`, not `ready`, so `PRIORITY` never applied to it, and
`src/mousedroid/harness/spec.py::select_next` returns any `in_progress` feature first as `resume`
(lines 424–426). With the edge fixed and F-059 itself `in_progress` and `critical`, the selector
already prefers it over the in-progress F-050 (high) and F-051 (medium) — the pre-emption R3-1
describes bites only while F-059 is `todo`. **EX-3 is therefore ratified for catalog honesty, not as
a selector fix** (plan correction to R3-1; pinned by
`tests/regression/test_f059_aqa.py::test_select_next_returns_f059_while_it_is_in_progress`, which
credits priority and skips rather than asserts once F-059 leaves `in_progress`).

## 5. Credit workstreams — WS-A … WS-D

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

Revised per R-10: `cache_control` is withdrawn (a no-op on a ~70-token system prompt — the
revision-1 task that added it is deleted); WS-B is **costed per agentic run**, not per cached call
(an agentic loop re-sends the context every turn); **model ids are not asserted in the budget table**
— every tool resolves the alias at run time via `client.models.retrieve()` and writes the resolved id
into its artifact and ledger row. Prices are [Likely], quoted from the `claude-api` skill's model
table (cached 2026-10-06; the skill is not a tracked path in this repository) and dated here so a
re-quote is a one-cell edit.

| WS | What | Model alias (resolved at run time) | Cap | Expected | Artifact |
|---|---|---|---|---|---|
| A | Anthropic request profile, Haiku 5.5 retarget, latency probe JSON report | `haiku` | $10 | < $2 total | `reports/llm_latency/<date>/*.json` |
| B | read-only Agent-SDK peer reviewer with a spend ledger | `opus` (top tier) | $60 | **$5–10 per agentic run × ≤ 6 runs** (mid tier $2–3/run); ledger before each run | `reports/agent_reviewer/<date>/wsb-NNNN.md` |
| C | Batch-API vulture triage — keep / delete / verify, **no deletions** | `sonnet` via the Batch API (50 % off) | $15 | < $5 (≈ 40 chunks × 8 K input) | `docs/analysis/dead-code-triage-<date>.md` |
| D | guardrails — lands with WS-B | — | — | — | pinned by tests |
| reserve | distillation spike (`docs/runbooks/jetson-alayaworld-spike.md`) only after F-008 is `done` | — | $115 | decided at the Sprint 2 review | — |

Expected alias resolution per the cached table [Likely]: `haiku` → `claude-haiku-5-5`
($0.10 in / $0.50 out per MTok ≤ 100 K prompt), `sonnet` → `claude-sonnet-5-5` ($2/$10), `opus` →
`claude-opus-5-5` ($4/$20). **A bare id swap would silently degrade the rover** [Likely — the 400
behaviour is quoted from the `claude-api` skill's model table, cached 2026-10-06, and not re-verified
against the live API]: Haiku 5.5 and Sonnet 5.5 return HTTP 400 on any non-default
`temperature`/`top_p`/`top_k`, and
`src/mousedroid/llm_gateway/anthropic_gateway.py::AnthropicLLMGateway._create_message` always sends
`temperature=cfg.temperature` (0.1 on both overlays) — so the gateway would flip `_degraded=True` and
the fallback would serve Phi-3 while the overlay says Haiku. WS-A's `AnthropicRequestConfig`
(`LLMConfig.anthropic`, default `None` ⇒ byte-identical request) exists for that; the probe asserts
`served: primary`.

**Ledger.** `reports/credit/{ledger}.jsonl` (tracked, one JSON object per run) written by
`tools/{credit_ledger}.py` — `charge(ws, estimate_usd)` raises `CapExceeded` **before** any API call;
`record(entry)` appends `{ts, ws, run_id, target, model_requested, model_resolved, resolved_via,
input_tokens, output_tokens, cache_read_tokens, cost_usd, cost_source, cap_usd, cumulative_ws_usd,
artifact, status}`. Caps come from a `credit:` block in `.claude/workforce.yaml`
(`caps_usd: {A: 10, B: 60, C: 15}`, `ledger: reports/credit/ledger.jsonl`), never from code literals.

**WS-D guardrails.** The tool allowlist is pinned by test; no `MCPConfig.actuation_tools` name ever
appears in a cloud agent's allowlist (`set_velocity`, `emergency_stop`, `translate_nl_mission`,
`export_experience`, `tensorrt_compile`, `calibrate_ultrasonic` are denied by name as well); the
prompt builder refuses env/secret files (`*.env`, `docker.env*`, anything matching `.gitleaks.toml`
patterns); keys come from env only, never CLI; every run pre-checks its cap and appends a ledger row;
`max_turns` and caps live in `.claude/workforce.yaml`; the reviewer's MCP runs only with
`MOUSEDROID_MOCK_HARDWARE=true` on the PC, never against the rover. All credit work is gated on the
key rotation (task 0.2): a key appeared in a chat transcript and must be revoked before any spend.

## 6. Corrections — what the initial research and earlier revisions got wrong

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

Recorded so a later pass does not re-raise them.

1. **S-5 is not "add an e-stop latch."** The latch landed in #232 (`SafetyConfig.emergency_latch`,
   `python -m mousedroid.cli.rearm --config /etc/mousedroid/jetson_production.yaml …`,
   `docs/runbooks/emergency-stop-rearm.md`) and ships **OFF**. The
   work is CF-2 — turn it on in production, declare it off explicitly on the lidar-only stack — not a
   new mechanism (D-5).
2. **There is no cloud-VLM seam.** The initial credit allocation assumed a seam the credit could
   exercise by sending camera frames to a cloud vision model. The only VLM seam in the tree is local
   and mocked: `src/mousedroid/factory/mission.py::build_vlm_progress` returns `None` unless
   `mission.vlm_progress_enabled` and otherwise wraps `src/mousedroid/reward/vlm_progress.py::MockVLMProgress`
   in `VLMProgressHead` (its docstring: a real backend "is a separate sprint"); `VLMProgressBackend`
   is a `Protocol` with that one implementation; the Anthropic gateway's `_create_message` takes
   strings, never frames; growth/VLA distillation is off-loop and local. Building such a seam is
   CHARTER-§3-adjacent and out of scope; the credit goes to reviewer and triage work on surfaces that
   exist (§5, §2 D-2). *(reconstructed from the plan's decision register — the review record does not
   carry this correction's text; the symbols above were verified on disk at `5b2cf71`, fix round 1)*
3. **`reports/dead_code/` is gitignored** (`.gitignore` 336). WS-C's input chain is therefore
   declared **local-only**: `python scripts/dead_code_audit.py` regenerates the findings file, and the
   triage document records that file's sha256 and the vulture version instead of a tracked JSON.
4. **The stock flip is necessary, not sufficient.** `command_set: waveshare_stock` on the
   production overlay (D-2) fixes the dialect; the F-008 motor stage *also* needs the bench overlay
   (D-11 — `esp32.enabled: false` routes to the mock), the encoder-less criterion (D-1 — stock echoes
   the commanded speed), both motion gates, and a board that answers. A stock-flipped overlay against
   a dead board is still a FAIL — correctly.
5. **Three review-record inaccuracies**, corrected in this file rather than by rewriting the
   record: `rsync_code` has nine excludes including `.claude`, not eight (B-2b); `reports/` is
   gitignored in sub-trees only (B-2b); "`select_next` will return it" was false for the reasons in
   §4 (R3-6; the in-place cell corrections themselves are task 4.6's narrative sweep).
6. **The deploy default was a live hazard, not a nesting bug** (R3-3): B-1 and the 2.4b header
   warning replace revision 2's "nests the repo root inside the package parent" framing.
7. **Fix round 1 (2026-10-19) corrections to this reconstruction** — the exit-4 claim was revision
   1's B-2 sub-claim, not a B-3 row; B-3 and B-6 are surviving revision-1 rows and the round-3
   negation finding is B-7; "one `.gitkeep` is the only negation loss" was false — the landed fence
   ships sentinels and documents eight non-shipping tracked files (B-7); §2 table D was missing;
   `~/firmware` is unreachable from the container (3.9, Ruling 20); the bench-overlay opt-out
   records SKIP, not a mock PASS (Ruling 31); the preflight probe reports connect errors, not lock
   errors, until 4.1 (Ruling 28); re-arm commands name the production config (Ruling 36); the
   bench restarts the container in place, because a unit restart recreates it (Ruling 37); weekday
   labels replaced by dates.

## 7. Success criteria

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

1. **Silent board, red stage.** With the bench overlay stacked and a dead ESP32,
   `bash scripts/jetson_smoke_test.sh motor` FAILs — not skips, not passes — because the encoder-less
   branch requires a T:1001 frame (`EncoderReading.imu_valid`) within `esp32.max_consecutive_timeouts`
   polled reads, with `emergency_stop()` in `finally` (Ruling 33); any commanded velocity is bounded
   by `esp32.smoke_test_velocity_mps` (default 0.05 m/s; `0.0` locks all smoke motion — Ruling 32);
   the stage ↔ test pairing is pinned by `tests/regression/test_f059_aqa.py`.
2. **Preflight probes.** `python -m mousedroid.cli.preflight --checks esp32`: `enabled=false` → WARN
   (never OK; aggregate DEGRADED); `legacy` → WARN, no probe; silent stock board → FAIL; answering
   board → OK with "FEEDBACK_BASE_INFO" in `detail`; a connect error (device node missing or
   unreadable) → FAIL fast with the message in `detail`. The probe is run cold — the driver takes no
   exclusive lock until 4.1, so a live container that owned the port would contend for the frames
   rather than fail (Ruling 28; PR-C `48820c4`). No new `PreflightStatus` member.
3. **The production overlay ships the decided configuration.**
   - 3a. `command_set: waveshare_stock`, `serial_baud: 115200`, `chassis_has_wheel_encoders: false`,
     `esp32.enabled: false` (the flip lives only in the bench overlay).
   - 3b. `battery_warn_v: 10.5`, `battery_critical_v: 9.5`; `battery_implausible_below_v` 1.0
     unchanged, so the runtime routes 0.0 V to log-only and a missing reading can never e-stop a
     moving rover.
   - 3c. **The smoke `power` stage FAILS on a missing voltage** (R-6): a board that answers the poll
     without `v` reads 0.0 and `battery_voltage_v >= battery_critical_v` fails. The gate is
     deliberately stricter than the runtime; the test gains no floor. The motor stage's
     `test_battery_voltage_in_plausible_range` keeps passing on a missing reading — a declared
     asymmetry (R3-5), motion gate vs telemetry gate.
   - 3d. `emergency_latch.enabled: true`; `config/jetson_lidar_only.yaml` declares `false`
     explicitly and the `production,lidar_only` stack loads with the latch off while its
     `lidar_unavailable_policy` stays `emergency`.
4. **One place for the flip.** `config/{jetson_bench_esp32}.yaml` is the only file where
   `esp32.enabled` becomes `true`; it contains exactly that key; it leaves
   `esp32.smoke_test_allow_motion` False; `jetson_smoke_test.sh` motor/power stack it by default;
   `MOUSEDROID_SMOKE_BENCH_ESP32_OVERLAY=` (set-but-empty) opts out — the two stages then record SKIP
   without running pytest — and with the overlay stacked a test that skips is a stage FAIL (a skip is
   never a measurement; Ruling 31).
5. **The deploy path cannot delete rover state.** `bash scripts/deploy_remote.sh --print-rsync-args`
   lists `--filter='+ **/.gitkeep'` ahead of `--filter=':- .gitignore'`, the nine dev-tool excludes
   and the seven runtime-root excludes anchored and without a trailing slash (`/venv` …
   `/tensorrt_cache`); the fixture proves both directions — no runtime root, symlinked or not, is ever
   listed for deletion, a same-named directory deeper in the tree still ships, every `git ls-files`
   path outside the fenced roots transfers, and a tracked sentinel under a file-level ignore ships
   while the eight documented non-shipping tracked files (B-7) do not; the default `REMOTE_SRC` is
   `/opt/mousedroid`; `--code-only` never runs `deploy_jetson.sh`; the container health check waits a
   bounded `MOUSEDROID_REMOTE_HEALTH_WAIT_S` and WARNs, never skips silently; the restart mode is
   explicit (`MOUSEDROID_REMOTE_RESTART_MODE=unit|container`, Ruling 37); `--config-only` refuses a
   remote tree that lacks `battery_implausible_below_v` ("deploy code first"). Fence and retarget
   land as one PR (Rulings 18/23/24/26/27).
6. **Firmware is pinned and flashing is verify-then-dry-run.** The tracked record pins the four
   parts by offset, size and sha256; `bash scripts/flash_esp32.sh --port <dev>` verifies all four before
   any `esptool.py` call and prints the factory argv; `--write` runs `write_flash` then
   `verify_flash`; a hash mismatch refuses before esptool runs; `scripts/jetson_qspi_update.sh` is
   deleted.
7. **The catalog says what the plan says.** F-059 `critical`/`fast`, `depends_on: ["F-001"]`;
   F-008 `depends_on` includes `"F-059"`; F-050/F-051 `deferred` with the reason in `notes`;
   `python scripts/select_next.py` prints F-059 while F-008 is `todo` — **met at `ab65226`** (§4).
8. **Every inverted pin is declared.** Each regression the flips invert is named by `file::test`
   symbol in `tasks.md` and in its D-section, and its docstring declares the reversal; every new pin
   is proven red-before-green with `bash scripts/prove_pin_fails.sh`.
9. **F-008 closes on evidence only.** Phase 3 either flips F-008 to `done` with `implemented_in` =
   the full hex SHA the rover ran and evidence under `reports/esp32_bench/<date>/`, or yields a
   hash-pinned repair-vs-replace decision in `decision.md`; in both cases `hw-evidence-auditor`
   traces every hardware claim to a file.
10. **Credit stays inside the ledger.** Every ledger row ≤ its cap; the WS-A report shows
    `served: primary`; a WS-B artifact exists for the first Phase-1 PR; the WS-C document proposes
    zero `src/` deletions; no `cache_control`; model ids resolved at run time and written into every
    artifact.
11. **Gates.** `python -m ruff check .`, `python -m ruff format --check .`, `python -m mypy --strict src/`
    clean; the four pytest steps of CI's `test` job green with coverage ≥ 90 %;
    `python scripts/validate.py --check F-059` OK; `--tier fast,slow --strict-git` green after
    closeout; no new `noqa` / `type: ignore` / `hardcoded-ok` marker (all three budgets are at
    ceiling).

## 8. Authoritative counterparts

Per `openspec/project.md`, this tree is documentation-only and the repo-native artifacts win on
disagreement:

- `features.yaml` — F-059 (reserved `739494e`), the F-008 `depends_on` edge, the F-050/F-051
  deferral.
- `scripts/validations/F-059.sh` — the harness-executed proof; `tests/regression/test_f059_aqa.py`
  and `tests/regression/test_f059_backwards_compat.py` — the pins it runs.
- `openspec/changes/mouse-droid-rover-gate-closeout/peer-review.md` — rounds 1–3, the record this
  file is reconstructed from.
- `docs/CHARTER.md` §3/§4 — the motion posture every bench step is checked against.
- `docs/runbooks/esp32-bench-ladder.md` — the Phase-3 ladder as landed (`5b2cf71`); `tasks.md` Phase 3
  mirrors it.
- `deployments/jetson-image.json` — the config-compat pin re-pinned by task 4.5.
