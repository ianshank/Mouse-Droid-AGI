# Peer review — `mouse-droid-rover-gate-closeout` (F-059)

Two rounds. Round 1 was an independent adversarial review of revision 1 by a reviewer
that had not seen the drafting, working from `.claude/agents/peer-reviewer.md` against
trunk `f57d99b`. Round 2 is the author's re-verification of every round-1 finding against
source, with the disposition applied in revision 2. Nothing below was accepted on the
reviewer's word; each row names what was re-read.

## Round 1 — independent review of revision 1

**Verdict: REQUEST_CHANGES.** "The premise is right and most of the evidence holds, but
the plan's first task repairs the wrong file and its fourth would wipe the rover's runtime
state."

### Verdict table

| Claim (revision 1) | Verdict | Re-verified (round 2) |
|---|---|---|
| A-1 / D-1: the F-008 motor stage is gated by `chassis_has_wheel_encoders` | **REFUTED** — `jetson_smoke_test.sh::test_motor` runs `tests/hardware/test_esp32_loopback.py`; the grepped test never reads the flag; only `test_motor_smoke.py` does, and the motor stage does not run it | Confirmed: `test_motor` → `test_file=…/test_esp32_loopback.py`; `grep chassis_has_wheel_encoders test_esp32_loopback.py` → none; `pytest.skip` present. [Certain] |
| CF-4 / D-6: `REMOTE_SRC=/opt/mousedroid` is safe with `--exclude .git` | **REFUTED** — `rsync --delete` exclude list protects `.venv`, not `venv/`; `weights/`, `models/`, `reports/`, `logs/` unprotected | Confirmed: `rsync_code` excludes `.git __pycache__ .venv *.egg-info node_modules .mypy_cache .pytest_cache .ruff_cache` only; `REMOTE_VENV=/opt/mousedroid/venv`. [Certain] |
| B-2: `rover_wip_guard.sh` exits 4 because rsync excludes `.git` | **REFUTED** — `resolve_toplevel` walks up to the enclosing work tree | Confirmed: `rover_wip_guard.sh::resolve_toplevel` runs `git -C <target> rev-parse --show-toplevel`, which resolves from any subdirectory of a work tree. Claim removed from B-2. [Certain] |
| D-5: `jetson_lidar_only.yaml` keeps the latch off | **REFUTED for the shipped stack** — `jetson_validate.sh` loads `production,lidar_only`; deep merge inherits `true` unless lidar_only declares `false` | Confirmed: `REMOTE_CONFIGS_CSV` default is the two-file stack; `grep emergency_latch config/jetson_lidar_only.yaml` → comment only, no key. [Certain] |
| D-1 rationale: `MockESP32Driver` reports velocity | **REFUTED** — `read_encoders` returns `EncoderReading()` (zeros) | Confirmed. The rationale was wrong; the decision (overlay, not default) stands on invariant 6 alone. [Certain] |
| A-2, A-3, A-4, A-5, A-7, A-8, B-1, B-3, D-5 (#242 uncalled; purity pin lacks `anthropic`), D-6 | **CONFIRMED** | — *(annotation, fix round 1, 2026-10-19: D-5 is still live at `5b2cf71` — `tests/regression/test_hot_loop_import_purity.py::_FORBIDDEN_MODULES` names only the #242 SDKs; Ruling 22 closes it in task 5.4 by adding `anthropic` and `claude_agent_sdk`; the row is reconstructed as proposal §2 D-5. Original cell text untouched.)* |
| F-059 is next free; `depends_on: ["F-008"]` passes `spec.py::check_dag` | **CONFIRMED** — dangling + cycle checks only; `features.schema.json` has no status rule | Confirmed; task 0.1's fallback clause deleted as dead text. [Certain] |
| D-4: legacy `{"T":2}` is a stock PID write; 12.0 V mock and 0.0 V both stay clear of e-stop | **CONFIRMED** | — |
| Deleting `jetson_qspi_update.sh` breaks a test or link | **CLEAN** — only self-reference | — |

### Defects (severity-ordered) and dispositions

| ID | Sev | Finding | Disposition in revision 2 |
|---|---|---|---|
| R-1 | **Critical** [Certain] | The plan's "first task" does not touch the gate it claims to repair. The F-008 `validation_command` motor stage greps `test_send_velocity_moves_encoders PASSED` from `test_esp32_loopback.py`, which `pytest.skip`s on near-zero feedback and has no `chassis_has_wheel_encoders` branch. Under stock firmware `WaveshareStockCodec.parse_encoders` echoes commanded speed into `L`/`R`, so it passes coincidentally — asserting a measurement never made. | **Accepted.** D-1 is now two halves: (1) `test_esp32_loopback.py` gains the encoder-less branch and loses the skip on that branch; a new AQA reads `jetson_smoke_test.sh` and pins the file ↔ test-name pairing; (2) the overlay flag. Task 1.1 / 1.1b. Proposal §1 and A-1 rewritten with the revision-1 claim struck through in place. |
| R-2 | **Critical** [Certain] | CF-4 as written deletes the rover's venv, LLM model and ONNX weights on first deploy. `rsync -avz --delete` retargeted at `/opt/mousedroid` removes `venv/` (the service interpreter), `weights/`, `models/`, `reports/`, `logs/` — all gitignored, so the WIP guard reports clean. `require_safe_remote_path` validates syntax, not blast radius. The *current* default is also broken (it nests the repo root inside the package parent). | **Accepted.** CF-4 split into 2.4a (fence: `--filter=':- .gitignore'` + explicit excludes, pinned by an `rsync -n --delete` fixture test that must land first) and 2.4b (retarget, plus `jetson_validate.sh`, `validate_compose_env.sh`, two runbooks). Git-fetch delivery recorded as the fallback. B-2b added to the proposal. |
| R-3 | High [Certain] | Four pinned regressions the plan did not name: `test_f025_backwards_compat.py::test_shipped_overlays_keep_legacy_baud_without_the_env_lever` (D-2), `test_jetson_deploy_prep_backwards_compat.py::test_default_plus_production_overlay_stack_loads` (D-4), `test_f051_backwards_compat.py::test_remote_path_defaults_are_the_previous_literals` (CF-4), `test_preflight_real_branches.py::TestEsp32RealBranch::test_driver_ok` (D-3). | **Accepted.** Each is named by symbol in the task that inverts it (1.2, 1.4, 2.1, 2.4b) and in the matching D-section, with the docstring requirement that the reversal be declared. |
| R-4 | High [Certain] | D-3 specified a `SKIP` outcome; `PreflightStatus` has `OK/WARN/FAIL/DEGRADED` only. | **Accepted — and resolved by not adding the member.** `enabled=false` → `WARN` with the reason in `detail`; aggregate `DEGRADED` is the documented meaning. Spec scenario and task 1.4 updated. |
| R-5 | High [Likely] | Task 1.6 (`exclusive=True`) collides with 3.7: `jetson_full_smoke_run.sh` runs stages via `docker exec` in the running container whose orchestrator holds the port. | **Accepted.** 1.6 moved to 4.5, coupled with a cold-vs-warm decision for the smoke-run motor stages. S-9 stays open and is recorded as such; the spec requirement is kept but now lands in Phase 4. |
| R-6 | High [Certain] | CF-1 + missing telemetry = red `power` gate: `test_power_chain_smoke.py` asserts `battery_voltage_v >= battery_critical_v` with no floor; `get_battery_voltage` returns `0.0` on no reading. Possibly desired; the plan did not say. | **Decided in writing:** the smoke gate is allowed to be stricter than the runtime. A board that answers the poll without `v` fails `power`; the runtime keeps its 1.0 V floor so a missing reading cannot e-stop a moving rover. Recorded in D-4, success criterion 3c, spec scenario, task 2.1. |
| R-7 | Med [Certain] | D-5 contradicted its own rationale: lidar_only declares no `emergency_latch`, so the validation stack inherits `true` and `lidar_unavailable_policy: emergency` latches on a >0.5 s dropout. | **Accepted.** `jetson_lidar_only.yaml` writes `enabled: false` explicitly; the AQA loads the stack, not the file. D-5, spec, task 2.2. |
| R-8 | Med [Certain] | B-2's inventory missed `jetson_validate.sh::REMOTE_SRC`, `validate_compose_env.sh::_SOURCE_PATHS_IN_USE`, `test_validate_compose_env.py`, `jetson-claude-pilot-deploy.md`, `docs/operator/JETSON_SMOKE_RUNBOOK.md`. | **Accepted.** B-2 rewritten as two camps with every member listed; task 2.4b covers each. |
| R-9 | Med [Likely] | D-9's ordering rationale inverted: the pinned `032942b` schema is plain `BaseModel` without `extra="forbid"`, so `check_config_compat.py` ignores unknown keys and already passes keys the pin does not know. | **Accepted.** D-9 and task 2.10 corrected: re-pin after merge, before branch cleanup, for reachability — not as a precondition for the flips. Follow-up filed to make the gate pin a schema that forbids extras. |
| R-10 | Med [Likely] | WS-A `cache_control` is a no-op on a ~70-token system prompt. WS-B "$1.50/review" assumed one cached call; an agentic loop re-sends per turn. Model ids asserted from memory. | **Accepted on all three.** `cache_control` withdrawn (task 5.2 deleted). WS-B re-costed per agentic run at $5–10 top tier / $2–3 mid tier; cap unchanged, run count reduced; ledger before each run. Model ids removed from the budget table; the pricing figures are cited as [Likely] with their source and date, and task 5.1/5.4 resolve the id at run time. |
| R-11 | Low [Certain] | `mousedroid-trend.service` runs `cli.preflight` on a timer with the container up; `MOUSEDROID_TREND_CHECKS` is operator-editable and nothing pins `esp32` out of it. | **Partially accepted.** Re-read: the unit ships `MOUSEDROID_TREND_CHECKS=config,host_env_keys` with a "non-exclusive checks" comment, so the default is already safe. The gap is that nothing *pins* it. Task 1.9 adds the regression test; D-3 records both fences. |

### What survives review unchanged (round 1)

- §2 tables A-2–A-5, A-7–A-9, B-1, B-3–B-6, C-1, D-1–D-7; §6 corrections. The F-008 gate *is* SKIP-green without motion; `_check_esp32` *is* a non-probe; `heartbeat_enabled` *is* inert under `legacy`.
- D-2's wire frames and the derived 115200 (`ESP32Config._apply_command_set_coupling`).
- D-4's safety argument (`battery_implausible_below_v=1.0` routes `0.0` to log-only; `mock_battery_v=12.0` clears both thresholds).
- D-7 record shape; D-8 bench ladder; CHARTER §3 posture — every motion step is dual-gated, on blocks, operator-present; WS-D forbids actuation tools.
- F-059 numbering and DAG validity; `select_next` will return it (critical beats F-050's high).

## Round 2 — author's re-verification

Every REFUTED row above was re-read from source before the disposition was applied; the
"Re-verified" column records what was read. R-3's `test_preflight_real_branches.py::TestEsp32RealBranch::test_driver_ok`
was also re-read: it monkeypatches `factory.build_esp32_driver` to return `SimpleNamespace()`,
which has no `connect`, so D-3 breaks it exactly as stated. [Certain] All round-1 refutations
are now [Certain].

**What round 2 did not find that round 1 missed.** Nothing new. Round 2 was a
verification pass, not a second search; a fresh reviewer against revision 2 (task 2.9 /
5.5) is the next adversarial step, not this section.

## Load-bearing pins for revision 2 (carried from round 1, all adopted)

1. `tests/regression/test_f059_aqa.py` reads `scripts/jetson_smoke_test.sh` and pins the
   motor stage ↔ `test_esp32_loopback.py::test_send_velocity_moves_encoders` pairing.
2. `tests/unit/scripts/test_deploy_remote_guard.py::test_delete_never_touches_runtime_state`
   — `rsync -n --delete` with the script's own flags never lists a gitignored path.
3. Every regression the flips invert is named by symbol in `tasks.md` (1.2, 1.4, 2.1, 2.4b).
4. `enabled=false` ESP32 preflight is `WARN`, never `OK`; no new `PreflightStatus` member.
5. `jetson_lidar_only.yaml` sets `emergency_latch.enabled: false` explicitly; the AQA loads
   the `production,lidar_only` stack.
6. R-6 decided: `power` fails on missing voltage; the smoke test gains no floor.
7. Exclusive serial open lands in Phase 4 with the cold-vs-warm smoke-run decision, not in
   Phase 1.
8. No `cache_control`; WS-B costed per agentic run; model ids resolved at run time and
   written into every artifact.

## Verdict (round 2, on revision 2)

Revision 2 addresses every round-1 finding. It has **not** been re-reviewed by an
independent party; task 2.9 runs the house reviewer (or WS-B once it exists) against the
first PR cut from Phase 1 before anything merges. Until then the bundle's status is
**proposed**, and CF-1 through CF-5 remain unratified.

## Round 3 — independent re-verification against the canonical checkout (2026-10-16)

Rounds 1 and 2 ran in a read-only proxy clone. Round 3 ran in the canonical repository
(worktree `openspec-peer-review-0b49b8`, trunk `f57d99b`, clean) by a reviewer that had
seen only this file — not `proposal.md`, `design.md`, `specs/` or `tasks.md`, which exist
only in the drafting session's zip and Project upload and are not in any branch, stash,
worktree or artifact of this clone. Every row below was re-read from source here; the
bundle itself could not be reviewed and is marked NEEDS_CONTEXT at the end.

### Verdict table — rounds 1–2 claims re-read in the canonical tree

| Claim | Round 3 | Evidence (by symbol) |
|---|---|---|
| A-1 / D-1 refutation | **HOLDS** [Certain] | `scripts/jetson_smoke_test.sh::test_motor` sets `test_file=…/tests/hardware/test_esp32_loopback.py`, greps `test_send_velocity_moves_encoders PASSED` and `test_emergency_stop_latency PASSED`; the loopback module never reads `chassis_has_wheel_encoders`; `pytest.skip("encoder loopback inactive…")` present |
| CF-4 / D-6 refutation | **HOLDS, one inaccuracy** [Certain] | `scripts/deploy_remote.sh::rsync_code` has **nine** excludes, not eight — the review omits `.claude`. `venv/` (`.gitignore:42`), `weights/` (`:85`), `logs/` (`:113`), `models/` (`:261`) are gitignored and unprotected. `reports/` is **not** gitignored wholesale — only sub-trees (`reports/jetson_smoke/**`, `reports/jetson_validate/`, `reports/dead_code/`, `reports/endurance/*` with a negated `.gitkeep`); the blast radius is the same, the phrasing "all gitignored" is not |
| B-2 refutation (`resolve_toplevel`) | **HOLDS** [Certain] | `scripts/rover_wip_guard.sh::resolve_toplevel` → `git -C "${target}" rev-parse --show-toplevel` |
| D-5 refutation (stack inherits the latch) | **HOLDS** [Certain] | `scripts/jetson_validate.sh::REMOTE_CONFIGS_CSV` defaults to `jetson_production.yaml,jetson_lidar_only.yaml`; `config/jetson_lidar_only.yaml` mentions `safety.emergency_latch.enabled` only in a comment; `config/jetson_production.yaml` carries no `emergency_latch` key today either |
| D-1 rationale refutation (mock zeros) | **HOLDS** [Certain] | `comms/mock_driver.py::MockESP32Driver.read_encoders` returns `EncoderReading()` |
| F-059 next free; `check_dag` dangling + cycle only; schema has no status rule | **HOLDS** [Certain] | `features.yaml` ends at F-058; `harness/spec.py::check_dag` checks malformed ids, dangling `depends_on`, cycles; `features.schema.json` `allOf` has one rule (`status: done` ⇒ `validation_command` + `implemented_in` required) |
| **`select_next` will return F-059 (critical beats F-050's high)** | **REFUTED** [Certain] | See R3-1 |
| D-4: `{"T":2}` is a stock PID write; 12.0 V mock; 0.0 V clears e-stop | **HOLDS** [Certain] | `comms/command_set.py::LegacyCommandCodec.battery_query` docstring + `WaveshareStockCodec.battery_query` ("a READ, unlike the legacy PID-write"); `ESP32Config.mock_battery_v = 12.0`; `test_f025_backwards_compat.py::test_battery_implausibility_floor_defaults_on_and_backwards_compatible` pins `battery_implausible_below_v == 1.0`; `comms/base_driver.py::get_battery_voltage` returns `0.0` on `None` |
| `jetson_qspi_update.sh` self-reference only | **HOLDS** [Certain] | repo-wide grep: one hit, its own usage line |
| R-3 four pins exist; `test_driver_ok` uses `SimpleNamespace()` | **HOLDS, incomplete** [Certain] | All four symbols exist. `test_jetson_deploy_prep_backwards_compat.py::test_default_plus_production_overlay_stack_loads` pins **three** production-overlay values — `battery_critical_v == 0.0`, `esp32.enabled is False`, `min_valid_sensors == 1` — the review names it for the first only (see R3-2) |
| R-4 `PreflightStatus` members | **HOLDS** [Certain] | `validation/preflight.py::PreflightStatus` = `OK/WARN/FAIL/DEGRADED` (DEGRADED aggregate-only) |
| R-5 smoke run executes in the live container | **HOLDS, conditional** [Certain] | `scripts/jetson_full_smoke_run.sh` builds a `docker exec … ${CONTAINER} python3` wrapper and runs `motor`, `power` and `tests/hardware/test_motor_smoke.py` through it. But with the shipped overlay the orchestrator in that container holds a `MockESP32Driver` and no serial port (see R3-2); the collision exists only after `esp32.enabled` flips |
| R-6 `power` fails on 0.0 V after CF-1 | **HOLDS, with an asymmetry** [Certain] | `tests/hardware/test_power_chain_smoke.py` guards on `battery_critical_v > 0.0` and asserts `battery_voltage_v >= battery_critical_v` with no lower floor. But `test_esp32_loopback.py::test_battery_voltage_in_plausible_range` — in the module the **motor** stage runs — guards with `if v > 0.0`, so the motor stage stays green on the same 0.0 V (R3-5) |
| R-7 | **HOLDS** [Certain] | as D-5 above |
| R-8 B-2 inventory | **STILL INCOMPLETE** [Certain] | See R3-3 |
| R-9 pinned schema ignores extras | **HOLDS** [Certain] | `git show 032942b` has a single `src/mousedroid/config/schema.py`; `SafetyConfig(BaseModel)` and no `extra=` anywhere in the file; trunk's `config/schema/_primitives.py` `ConfigDict(extra="forbid")` post-dates the pin. The `check_config_compat.py` docstring ("`extra="forbid"` is the default on most config models") describes trunk, not the pinned SHA |
| R-10 | **NOT RE-VERIFIED** | external pricing; outside the tree |
| R-11 | **HOLDS** [Certain] | `scripts/mousedroid-trend.service` ships `Environment=MOUSEDROID_TREND_CHECKS=config,host_env_keys` with the "non-exclusive checks" comment; nothing pins it |
| "F-008 gate is SKIP-green without motion" | **HOLDS** [Certain] | `jetson_smoke_test.sh::main` exits `"${FAILURES}"`; `record_skip` increments `SKIPS` only; `test_motor` records SKIP unless `MOUSEDROID_SMOKE_ALLOW_MOTION=1` |
| `_check_esp32` is a non-probe; `heartbeat_enabled` inert under `legacy`; 115200 derived | **HOLDS** [Certain] | `validation/preflight.py::_check_esp32` builds the driver and returns OK with its type name, never calls `connect`; `ESP32Config.heartbeat_enabled` description + `_check_esp32_failsafe` docstring; `ESP32Config._apply_command_set_coupling` sets `serial_baud = WAVESHARE_STOCK_BAUD` (115200) when still at the legacy default |

### New findings (severity-ordered)

| ID | Sev | Finding | Required change |
|---|---|---|---|
| R3-1 | **High** [Certain] | **"`select_next` will return F-059" is false on two counts.** `harness/spec.py::select_next` returns any `in_progress` feature first as `resume` — F-050 and F-051 are `in_progress` today — and only then considers `todo` features whose `depends_on` are all `done`. With `depends_on: ["F-008"]` and F-008 `todo`, F-059 is **`blocked`**, not `ready`; `PRIORITY` ordering never applies to it. The edge is also semantically inverted: F-059's Phases 1–3 are what make F-008 pass, so F-008 cannot be a prerequisite of F-059. | Task 0.1: `depends_on: ["F-001"]` (or `[]`) on F-059, and add `"F-059"` to **F-008's** `depends_on` so the catalog says what the plan says — the hardware gate waits on the closeout. Note in the proposal that `select_next` will still say `resume F-050` until F-050/F-051 are closed or deferred; that is the correct answer, not a defect. Re-run `scripts/select_next.py` and paste the output into the bundle. |
| R3-2 | **High** [Certain on mechanism; Guessing on whether revision 2 covers it] | **The shipped production overlay routes every ESP32 smoke stage to the mock.** `config/jetson_production.yaml` sets `esp32.enabled: false`; `factory/hardware.py::build_esp32_driver` returns `MockESP32Driver` whenever `not cfg.esp32.enabled`, regardless of `mock_hardware`; `tests/_jetson_hardware.py::load_jetson_runtime_settings` only forces `mock_hardware=False`. So, as shipped: the F-008 **motor** stage reads mock zeros, skips, and `test_motor` records FAIL "encoder loopback inactive" — unpassable whatever the firmware does; the **power** stage reads `mock_battery_v=12.0` and passes against the mock — a green that measures nothing; `_check_esp32` reports OK `driver=…` for the mock (this *is* S-4). R-1's "passes coincidentally under stock firmware" and R-6's "fails `power` on missing voltage" both presuppose a real driver, i.e. a flip of `esp32.enabled` that this review never names. The pin that inverts is `test_default_plus_production_overlay_stack_loads` (`stacked.esp32.enabled is False`), named in R-3 for D-4 only. | The bundle must state where `esp32.enabled` becomes `true` (overlay, or `MOUSEDROID_ESP32__ENABLED` lever in `jetson_validate.sh` / `jetson_full_smoke_run.sh`), in which Phase, and name the inverted assertion by symbol. Until that flip, success criterion "motor stage PASS" is unreachable and "power stage PASS" is meaningless; the bench ladder's "healthy container with no rover attached" checkpoint must say that it runs against the mock. R-5's collision and task 4.5 become conditional on this flip. |
| R3-3 | **High** [Certain] | **R-2 understates the current default: it deletes the live package on a clean rover.** The rover's repo root is `/opt/mousedroid` (`scripts/mousedroid.service`: `WorkingDirectory=/opt/mousedroid`, `ExecStartPre=/opt/mousedroid/scripts/preflight_check.sh`; `docker-compose.jetson.yml` binds `/opt/mousedroid:/opt/mousedroid`; `docs/superpowers/specs/2026-06-02-jetson-claude-pilot-deploy-design.md` records "the container imports from `/opt/mousedroid/src`" via the editable `.pth`). `rsync_code` syncs `${PROJECT_DIR}/` into `/opt/mousedroid/src/`, whose only content is the `mousedroid/` package; the local repo root has no top-level `mousedroid/`, so `--delete` removes `/opt/mousedroid/src/mousedroid/` before writing the nested tree. The only thing that has prevented this is `rover_wip_guard.sh` refusing on a dirty rover (exit 3). `smoke-reports/smoke_report.md` already recorded the drift ("Plan v2 assumed `/opt/mousedroid/src/scripts/` — actual is `/opt/mousedroid/scripts/`"). B-2 also still misses a live surface: `config/docker.env.example` (the "nor covering or nested inside /opt/mousedroid/src" comment). `tests/unit/scripts/test_deploy_remote_guard.py::test_a_plain_absolute_path_is_still_accepted` passes `/opt/mousedroid/src` as a payload — it does not invert, but 2.4b should re-point it so the "legitimate override" it names is the real one. | Proposal: mark `scripts/deploy_remote.sh` with default paths as **do not run against the rover** until 2.4b lands, in the runbooks and in the script's own header. 2.4a's fixture test should include the current default as a negative case (the fence must refuse, or the test must prove `src/mousedroid` survives). Add `config/docker.env.example` to B-2 camp 2 and task 2.4b. |
| R3-4 | Med [Certain] | **`--filter=':- .gitignore'` drops every `.gitignore` negation.** rsync's `-` merge modifier means "the file should consist of only exclude patterns, with no other rule-parsing except for in-file comments"; `!` is a list-clearing rule that takes no argument, and rsync has no negation pattern syntax. `!reports/endurance/.gitkeep` becomes an exclude for a path literally beginning with `!` and the tracked `.gitkeep` is excluded by `reports/endurance/*`; `.claude/` negations are moot because `rsync_code` excludes `.claude` wholesale. Pin 2 (`test_delete_never_touches_runtime_state`) checks only the negative direction. | Task 2.4a: add the positive direction — over a fixture checkout, every `git ls-files` path appears in `rsync -n --itemize-changes` output (a tracked file under an ignored-with-negation directory in the fixture). Either accept the `.gitkeep` loss in writing or add an explicit `--include` for the negated tracked paths. |
| R3-5 | Low [Certain] | **Two tests in the same gate disagree on whether 0.0 V is a failure.** `test_power_chain_smoke.py` fails on `battery_voltage_v < battery_critical_v` with no floor (R-6's decision); `test_esp32_loopback.py::test_battery_voltage_in_plausible_range`, run by the motor stage, guards with `if v > 0.0` and passes on a missing reading. | D-4 / task 2.1: state the asymmetry and either keep it (motor stage is a motion gate, not a telemetry gate) or remove the loopback guard so both stages agree. One sentence; no new test required. |
| R3-6 | Low [Certain] | Round-2 "Re-verified" column inaccuracies: the exclude list omits `.claude`; "`reports/` … all gitignored" is only true of sub-trees. Neither changes a disposition. | Correct the two cells above in place. |
| R3-7 | Info [Likely] | Re-pinning `deployments/jetson-image.json` to a trunk SHA that carries `_primitives.py`'s `extra="forbid"` makes `check_config_compat.py` strict for the first time, so any stale key already in `config/*.yaml` surfaces at that re-pin, far from the change that introduced it. | Task 2.10: run `scripts/check_config_compat.py --changed-files config/*.yaml` against the candidate SHA before pinning, and record the result in the bundle. |

### What survives round 3 unchanged

- Every round-1 refutation (A-1/D-1, CF-4/D-6, B-2, D-5, D-1 rationale) and every
  round-1 confirmation except the `select_next` sentence.
- R-1 through R-11 as findings; their dispositions are coherent with the tree as read here.
  R-5 and R-6 are correct *given* a real driver — which is exactly what R3-2 says the plan
  must make true explicitly.
- The eight load-bearing pins, with pin 2 widened by R3-4.
- F-059 numbering; DAG validity (no dangling edge, no cycle) — only the selection claim and
  the edge direction fall.
- D-2's wire frames and derived baud; D-4's runtime safety argument; the SKIP-green reading
  of the F-008 gate; `_check_esp32` as a non-probe; `heartbeat_enabled` inert under `legacy`.

### Verdict (round 3)

- **On this review record (rounds 1–2): REQUEST_CHANGES.** One "survives" claim is false
  (R3-1), one inverted pin was named for the wrong reason and the flip it implies was never
  stated (R3-2), and the current deploy default is a live hazard rather than a nesting bug
  (R3-3).
- **On revision 2 of the bundle: NEEDS_CONTEXT.** `proposal.md`, `design.md`, `specs/` and
  `tasks.md` are not in this repository. Unpack the drafting session's zip at
  `openspec/changes/mouse-droid-rover-gate-closeout/` (keeping this file, which is a
  superset of the zip's `peer-review.md`), then re-run the house reviewer against the four
  files with R3-1 through R3-4 as the opening questions. CF-1 through CF-5 remain unratified.

Revision 2 reconstructed in-repo on 2026-10-19 (Task 0.0); CF-1/2/4/5 and EX-1/2/3 ratified by the operator on 2026-10-17 — see the Decision register in the implementation plan.
