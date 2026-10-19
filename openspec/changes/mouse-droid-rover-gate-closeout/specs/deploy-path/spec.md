# Spec — deploy path

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

Delta for `scripts/deploy_remote.sh`, its sibling `scripts/jetson_validate.sh`, the firmware record
and the config-compat pin. The fence (2.4a) and the retarget (2.4b) land as **one PR**, fence commit
first (Rulings 18/23); git-fetch delivery is the recorded fallback and the bench's primary (CF-4,
D-6). The scenarios below describe the fence PR-E landed (`claude/f059-pr-e-rsync-fence`, rebased onto trunk `f57d99b`, tip `4fa0014`, open as PR #260)
— the bundle follows the landed code (Ruling 24).

## Requirement: `rsync --delete` MUST never remove rover runtime state

### Scenario: no runtime root is ever deleted (negative direction)
- **GIVEN** a fixture checkout whose `.gitignore` anchors every runtime root
  (`/venv/ /weights/ /models/ /logs/ /estop_latch/ /tensorrt_cache/`, `reports/endurance/*` with a
  negated `.gitkeep`) and a rover tree carrying a file under each root — `models` as a symlink to
  NVMe — plus the fixture's stale tracked module `src/mousedroid/{removed_module}.py`
- **WHEN** `rsync -n -a --delete --itemize-changes` runs with exactly the args
  `bash scripts/deploy_remote.sh --print-rsync-args` prints
- **THEN** the stale tracked file is listed for deletion (premise: `--delete` is live)
- **AND** no path whose first segment is a runtime root is listed, the symlinked root included
  (`::test_delete_never_touches_runtime_state`, `::test_a_symlinked_runtime_root_is_held_too`)
- **AND** the sentinel inside a fenced root is neither shipped nor deleted

### Scenario: the runtime-root excludes are anchored
- **GIVEN** the same fixture, which also tracks files under same-named directories deeper in the
  tree (a `weights` or `models` directory inside `src/mousedroid/`, a `reports` directory inside
  `docs/`)
- **WHEN** the dry run completes
- **THEN** every such nested same-named file transfers
  (`::test_runtime_root_excludes_are_anchored`) — the fence is `/reports`, not `reports`

### Scenario: every tracked file outside the fenced roots is transferred (positive direction, R3-4)
- **GIVEN** the same fixture, whose `.gitignore` also carries a file-level ignore with a negated
  sentinel (`cfg/runtime/*` + `!cfg/runtime/.gitkeep`) and a generated file beside that sentinel
- **WHEN** the dry run completes
- **THEN** every `git ls-files` path whose first segment is not a fenced root appears in the
  transfer list (`::test_every_tracked_file_outside_the_fenced_roots_is_transferred`)
- **AND** the tracked sentinel under the file-level ignore transfers — the `+ **/.gitkeep` rule
  precedes the `:- .gitignore` merge, which drops `!` re-includes — while the generated file beside
  it does not
- **AND** `reports/endurance/.gitkeep` is tracked yet NOT transferred: it sits under a fenced root
  and rsync never descends into an excluded directory — the documented loss

### Scenario: the non-shipping tracked files are documented, not discovered
- **GIVEN** the real tree at `5b2cf71` and the fence's rules
- **WHEN** the merge is emulated over `git ls-files` with the `!` lines dropped
- **THEN** exactly eight tracked files do not ship — three under `reports/` (two tracked JSON
  reports and the `endurance/.gitkeep` sentinel) and five that `.gitignore` ignores or re-includes
  (`.vscode/settings.json`, `.vscode/tasks.json`, `.agents/skills/approval-gate-wiring/SKILL.md`,
  `docs/3D_printing_files/README.md`, `docs/planning/SMOKE_REPORT_TEMPLATE.md`) — plus the 33
  tracked `.claude/**` files the dev-tool exclude drops at any depth
- **AND** the script header's DOES-NOT-SHIP note names the patterns; a new tracked file under any of
  them is a header edit, not a surprise on the rover

### Scenario: one definition of the args
- **GIVEN** the ssh/rsync shim harness in `tests/unit/scripts/test_deploy_remote_guard.py`
- **WHEN** `--code-only` runs
- **THEN** the args the shim recorded are a superset of the `--print-rsync-args` output and
  include `--delete`
- **AND** `rsync -avz --delete \` still immediately follows `preserve_remote_wip`, and the string
  `docker.env` appears nowhere in the script (the existing F-051 pins stay green)

### Scenario: the old default cannot come back (R3-3)
- **GIVEN** the fence and the retarget in one PR
- **WHEN** `scripts/deploy_remote.sh` is read
- **THEN** the old literal survives nowhere, not even as a fallback — the inverted pin asserts
  `MOUSEDROID_REMOTE_SRC:-/opt/mousedroid/src}` is absent from the script
- **AND** the transfer pin proves the package ships to the root (`src/mousedroid/main.py` is in the
  transfer list) while the only `src/` deletion in the dry run is the fixture's stale module — the
  layout the old default produced no longer exists to be tested against

## Requirement: the default remote path MUST be the rover's repo root

### Scenario: the retarget
- **GIVEN** `scripts/deploy_remote.sh` and `scripts/jetson_validate.sh`
- **WHEN** read
- **THEN** both carry `REMOTE_SRC="${MOUSEDROID_REMOTE_SRC:-/opt/mousedroid}"`
- **AND** `tests/regression/test_f051_backwards_compat.py::test_remote_path_defaults_are_the_previous_literals`
  is inverted with the R3-3 reason in its docstring, and
  `tests/unit/scripts/test_deploy_remote_guard.py::test_a_plain_absolute_path_is_still_accepted`
  names `/opt/mousedroid`

### Scenario: fence and retarget land together; the script header carries the warning
- **GIVEN** PR-E/F (`3311d28`, `e02f71e` fence; `3de8c3f`, `19d9a24`, `b3e7732`, `30c6c6a`, `4fa0014`
  retarget and follow-ups; PR #260, base trunk `f57d99b`, closes #253)
- **WHEN** its commits are read in order
- **THEN** the fence precedes the retarget, and no trunk commit ever carries the fenced script with
  the old default — there is no window in which an operator needs a "do not run" notice
  (Rulings 18/23)
- **AND** the script header beside `REMOTE_SRC` records the hazard as history: the old default
  nested the repo under the package parent and `--delete` removed the live package beneath it (R3-3)
- **AND** `docs/runbooks/pc-to-jetson-promotion.md` §1 documents git-fetch delivery with the bench
  commands

### Scenario: the container package path is not retargeted
- **GIVEN** `scripts/validate_compose_env.sh::_SOURCE_PATHS_IN_USE`,
  `tests/unit/scripts/test_validate_compose_env.py` and `config/docker.env.example`
- **WHEN** 2.4b lands
- **THEN** `/opt/mousedroid/src` stays in all three — it is the container's package path and
  correctly a protected mount target
- **AND** `config/docker.env.example` gains only a clarifying parenthetical

### Scenario: the Docker spine restarts through the fail-closed path
- **GIVEN** `systemctl is-enabled mousedroid-docker` succeeds on the rover
- **WHEN** `--code-only` finishes syncing
- **THEN** in `unit` mode `systemctl restart mousedroid-docker` runs — its `ExecStartPre` re-syncs
  the overlay and runs `preflight_check.sh` — and the health check is
  `docker exec -w /opt/mousedroid mousedroid python3 -m mousedroid.main --health-check`
- **AND** when the unit is absent, no restart is attempted and the run still exits 0

### Scenario: the restart mode is explicit, and the hazardous combination is refused (Rulings 37/39/40)
- **GIVEN** `scripts/mousedroid-docker.service`: `ExecStart` is `docker compose up
  --abort-on-container-exit`, attached (line 79), `ExecStop` is `docker compose down` (line 80),
  `Restart=on-failure` and `RestartSec=10` (lines 83–84), and the `ExecStartPre` lines run the
  overlay sync and the preflight
- **WHEN** the unit is enabled, or `systemctl show -p ActiveState --value mousedroid-docker` prints
  anything but `inactive` (`active`, `activating`, `failed`, or the state cannot be read)
- **THEN** `MOUSEDROID_REMOTE_RESTART_MODE=unit` (the default; `b3e7732`) restarts through the unit —
  the container is **recreated**, the writable layer and every hot-installed package discarded, the
  overlay re-synced and the preflight run — and that recreate is accepted
- **AND** because `unit` is the default, a bare `bash scripts/deploy_remote.sh` after bench step
  3.3's hot-installs recreates the container and discards them unless the image was rebuilt
- **AND** `MOUSEDROID_REMOTE_RESTART_MODE=container` is **refused**, fail closed, before any sync
  and again at restart time (`read_unit_state`; `30c6c6a`, `4fa0014`): an in-place `docker restart`
  of a container the attached unit manages makes compose abort and the unit fail, after which
  systemd's on-failure retry runs `ExecStartPre` + `compose up` over whatever state the container is
  in — `ExecStop=` is skipped when the main process exits on its own; unverified on hardware, never
  raced — and the `RestartSec` auto-restart window is inside the refusal
- **AND** the unit-mode "runs under compose only" hint uses the same predicate, so it is never given
  where `container` would refuse
- **AND** pinned by `tests/unit/scripts/test_deploy_remote_guard.py::test_the_default_restart_mode_restarts_the_unit_and_not_the_container`,
  `::test_the_container_restart_mode_refuses_when_a_unit_is_enabled_or_running`,
  `::test_the_container_restart_mode_refuses_when_the_unit_state_cannot_be_read`,
  `::test_the_restart_time_check_uses_the_same_predicate`,
  `::test_the_compose_only_hint_is_not_given_when_container_mode_would_refuse`

### Scenario: with no unit enabled and the unit inactive, the in-place restart is the restart
- **GIVEN** a compose-only rover (`restart: unless-stopped`; no `mousedroid-docker` unit enabled
  and its `ActiveState` `inactive`)
- **WHEN** `MOUSEDROID_REMOTE_RESTART_MODE=container` runs
- **THEN** `docker restart mousedroid` restarts in place: it keeps the image, the compose config and
  the `env_file` the container was created with — none of which `docker restart` re-reads — and
  skips the unit's fail-closed `ExecStartPre` preflight; bind-mounted code and the hand-synced
  overlay take effect, the writable layer survives
- **AND** because the image is kept, a `docker compose build` is picked up only by the unit path
  (or `docker compose … up -d`), never by `docker restart`
  (`::test_the_container_restart_mode_restarts_in_place_when_no_unit_is_enabled_or_running`,
  `::test_the_container_restart_mode_warns_when_there_is_nothing_to_restart`)
- **AND** the runbook (3.3) applies the same rule: unit enabled or not inactive → through the unit,
  accepting the recreate (re-run the import probe; more than a few hot-installed packages → rebuild
  the image; the camera may need a host reboot, 2026-07-15); neither → in place

### Scenario: the health check waits after a unit restart this run made, then warns (Rulings 27/35)
- **GIVEN** a run whose `restart_service` restarted `mousedroid-docker` (`DOCKER_UNIT_RESTARTED=1`),
  which returns before compose has the container listed
- **WHEN** `run_health_check` runs
- **THEN** `wait_for_container` polls once a second, wall-clock-bounded by
  `MOUSEDROID_REMOTE_HEALTH_WAIT_S` (default 30; `0` probes once; an ssh 255 ends the wait at once),
  and execs the health check as soon as `mousedroid` is listed
- **AND** a container that never appears yields `WARNING: container mousedroid not running after
  <N>s — container health check skipped`, and a malformed wait value is refused before any remote
  command (`::test_the_health_check_waits_for_the_container_to_be_listed`,
  `::test_the_health_check_warns_when_the_container_is_never_listed`,
  `::test_a_malformed_health_wait_is_refused`)

### Scenario: without a unit restart, an unlisted container is information, not a warning (Ruling 35)
- **GIVEN** a run that did not restart the Docker unit (`container` mode, or no unit enabled)
- **WHEN** `run_health_check` runs
- **THEN** the container is probed once, with no wait
- **AND** an unlisted container is logged as an info line ("not running and this run did not restart
  the Docker unit — container health check skipped"), never as a WARNING (`b3e7732`)

### Scenario: `--code-only` never runs the bare-metal installer (Ruling 26)
- **GIVEN** a Docker-spine rover with no `/opt/mousedroid/venv`
- **WHEN** `bash scripts/deploy_remote.sh --code-only` runs
- **THEN** `deploy_jetson.sh` is not invoked — with the Docker unit enabled it logs that the bind
  mount serves the package; with neither a venv nor the unit it warns and carries on
- **AND** only `--full` runs `deploy_jetson.sh`, and the header says not to run `--full` against the
  Docker-spine rover (`::test_code_only_without_a_venv_leaves_the_bare_metal_installer_alone`,
  `::test_code_only_with_neither_a_venv_nor_the_docker_unit_warns_and_carries_on`)

## Requirement: configuration MUST NOT be installed onto a tree that cannot interpret it safely (code before config)

A rover still running the `032942b` image has no 1.0 V implausibility floor; installing CF-1
thresholds there trips `battery_critical` on a missing reading (review focus 3).

### Scenario: `--config-only` refuses an old tree
- **GIVEN** a remote tree whose `src/mousedroid/config/schema/reward_safety.py` lacks
  `battery_implausible_below_v`
- **WHEN** `bash scripts/deploy_remote.sh --config-only` runs
- **THEN** it exits non-zero with "deploy code first" on stderr and installs nothing

### Scenario: an unreachable rover is not a stale tree (Ruling 38)
- **GIVEN** a rover whose ssh link drops at the floor probe (ssh exit 255)
- **WHEN** `bash scripts/deploy_remote.sh --config-only` runs
- **THEN** `require_remote_safety_floor` dies "cannot reach the rover at `<host>`; nothing installed"
  — never "deploy code first" — and installs nothing
- **AND** `tests/unit/scripts/test_deploy_remote_guard.py::test_config_only_says_so_when_the_rover_drops_out_at_the_floor_probe`
  pins the branch (`b3e7732`)

### Scenario: the probed path cannot drift away from the tree (Ruling 38)
- **GIVEN** the guard's `grep -q <marker> ${REMOTE_SRC_Q}/<path>` line
- **WHEN** `::test_the_safety_floor_probe_reads_a_marker_this_tree_carries` runs
- **THEN** it regexes `<path>` and `<marker>` out of the script and asserts `<path>` exists in this
  repository and contains `<marker>` — the field already moved once, when #191 split the single
  `schema.py` module under `src/mousedroid/config/` into the `src/mousedroid/config/schema/` package,
  and a later move would otherwise make the guard refuse every `--config-only` after new code was on
  disk
- **AND** `::test_the_safety_floor_probe_pin_goes_red_on_a_scratch_copy` proves the pin is live

### Scenario: the bench is ordered the same way
- **GIVEN** bench step 3.3
- **WHEN** trunk is delivered by git-fetch
- **THEN** `sudo bash scripts/sync_jetson_overlay.sh` runs only after `import mousedroid.main`
  succeeds on the new tree
- **AND** the restart that follows obeys the rule: through the unit when `systemctl is-active
  mousedroid-docker` prints `active` — accepting the recreate, re-running the import probe, rebuilding
  the image when more than a few packages were hot-installed — and in place only when no unit is
  active (Ruling 39; ~~Ruling 37's always-in-place wording~~)

## Requirement: firmware MUST be hash-pinned in a tracked record and flashed only by a verify-then-dry-run script

### Scenario: the record
- **GIVEN** `deployments/{esp32-firmware}.json`
- **WHEN** parsed
- **THEN** `parts` lists exactly four entries at offsets `0x1000, 0x8000, 0xE000, 0x10000`, each
  with a 64-hex sha256 and a size; `connect_baud == 115200`; the app sha256 starts
  `ebedf216d23402381a16f44373721c8f`; `never_flash` names `combine/target.bin`

### Scenario: dry run is the default
- **GIVEN** four staged parts matching the record
- **WHEN** `bash scripts/flash_esp32.sh --port <dev> --bin-dir <dir> --record <record>` runs
- **THEN** it prints `verified <part> @ <offset>` four times, the write and verify argv, and
  `DRY RUN`
- **AND** `esptool.py` is never invoked

### Scenario: a hash mismatch refuses before esptool
- **GIVEN** one part with the right size and the wrong bytes
- **WHEN** the script runs with `--write`
- **THEN** it exits non-zero with `sha256 mismatch for <part>` on stderr
- **AND** `esptool.py` is never invoked

### Scenario: `--write` flashes then verifies
- **GIVEN** matching parts and `--write`
- **WHEN** the script runs
- **THEN** `esptool.py` is invoked exactly twice: `write_flash` with
  `--flash_mode keep --flash_freq keep --flash_size keep --compress`, then `verify_flash`

### Scenario: one record per platform artifact
- **GIVEN** `tests/regression/test_f051_aqa.py::test_no_parallel_deploy_manifest_was_introduced`
- **WHEN** the firmware record exists
- **THEN** the allowed set is `["esp32-firmware.json", "jetson-image.json"]`
- **AND** `scripts/check_config_compat.py`, `scripts/repin_tags.sh` and
  `scripts/archive_stale_branches.sh` still read `jetson-image.json` by name, so the new record is
  invisible to them

### Scenario: the dead script is gone
- **GIVEN** `scripts/jetson_qspi_update.sh`
- **WHEN** 2.5 lands
- **THEN** it is deleted and no tracked file references it

### Scenario: the images are staged where the container can see them (Ruling 20)
- **GIVEN** bench step 3.9 and a container that bind-mounts `/opt/mousedroid` and never the home
  directory
- **WHEN** the firmware is copied to the rover
- **THEN** it lands under `/opt/mousedroid/firmware/` (gitignored `*.bin` inside the checkout) and
  `docker exec … bash scripts/flash_esp32.sh --bin-dir /opt/mousedroid/firmware` can read it
- **AND** nothing is staged under a home directory `docker exec` cannot reach

## Requirement: the config-compat pin MUST be re-pinned after merge and before branch cleanup, with its new strictness pre-checked

### Scenario: pre-check before the re-pin (R3-7)
- **GIVEN** the candidate trunk SHA
- **WHEN** `python scripts/check_config_compat.py --platform jetson --deployments-dir <tmp> --changed-files config/*.yaml`
  runs against a copy of the record pointing at that SHA
- **THEN** every overlay passes under `extra="forbid"` before the tracked record is edited

### Scenario: ordering
- **GIVEN** PR-H merged
- **WHEN** 4.5 runs
- **THEN** `deployments/jetson-image.json` is re-pinned to the SHA the rover ran (PR-H's
  `implemented_in`) and its false "NOT an ancestor of the default branch" sentence is dropped —
  no PR-D commit touched the record (Ruling 29)
- **AND** `bash scripts/repin_tags.sh` tags the SHA (the push is an operator action) before any
  `bash scripts/archive_stale_branches.sh` run
- **AND** `tests/regression/test_f051_backwards_compat.py::test_the_record_sha_was_not_repinned_in_this_change`
  is inverted with a "post-merge trunk commit, ancestor of the default branch" docstring
