# Spec — deploy path

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

Delta for `scripts/deploy_remote.sh`, its sibling `scripts/jetson_validate.sh`, the firmware record
and the config-compat pin. Order is binding: the fence (2.4a) lands and is proven before the retarget
(2.4b); git-fetch delivery is the recorded fallback and Monday's primary (CF-4, D-6).

## Requirement: `rsync --delete` MUST never remove rover runtime state

### Scenario: no gitignored runtime root is ever deleted (negative direction)
- **GIVEN** a fixture checkout whose `.gitignore` ignores
  `venv/ weights/ models/ logs/ reports/endurance/* estop_latch/ tensorrt_cache/` and a rover tree
  carrying a file under each, plus one stale tracked file
- **WHEN** `rsync -n -a --delete --itemize-changes` runs with exactly the args
  `bash scripts/deploy_remote.sh --print-rsync-args` prints
- **THEN** the stale tracked file is listed for deletion (premise: `--delete` is live)
- **AND** no path whose first segment is a runtime root is listed

### Scenario: every tracked file outside the fenced roots is transferred (positive direction, R3-4)
- **GIVEN** the same fixture
- **WHEN** the dry run completes
- **THEN** every `git ls-files` path whose first segment is not a fenced root appears in the
  transfer list
- **AND** `reports/endurance/.gitkeep` is tracked yet NOT transferred — the accepted, written-down
  loss of `.gitignore` negations under `--filter=':- .gitignore'`

### Scenario: one definition of the args
- **GIVEN** the ssh/rsync shim harness in `tests/unit/scripts/test_deploy_remote_guard.py`
- **WHEN** `--code-only` runs
- **THEN** the args the shim recorded are a superset of the `--print-rsync-args` output and
  include `--delete`
- **AND** `rsync -avz --delete \` still immediately follows `preserve_remote_wip`, and the string
  `docker.env` appears nowhere in the script (the existing F-051 pins stay green)

### Scenario: the old default is a negative case (R3-3)
- **GIVEN** the fixture rover tree with `src/mousedroid/` present and a sync target of
  `<root>/src` — the layout the old default produced
- **WHEN** the fence is exercised against it
- **THEN** either the fence refuses, or the test proves `src/mousedroid` survives — the hazard
  cannot recur silently

## Requirement: the default remote path MUST be the rover's repo root

### Scenario: the retarget
- **GIVEN** `scripts/deploy_remote.sh` and `scripts/jetson_validate.sh`
- **WHEN** read
- **THEN** both carry `REMOTE_SRC="${MOUSEDROID_REMOTE_SRC:-/opt/mousedroid}"`
- **AND** `tests/regression/test_f051_backwards_compat.py::test_remote_path_defaults_are_the_previous_literals`
  is inverted with the R3-3 reason in its docstring, and
  `tests/unit/scripts/test_deploy_remote_guard.py::test_a_plain_absolute_path_is_still_accepted`
  names `/opt/mousedroid`

### Scenario: until then, the script says do not run me
- **GIVEN** the window between 2.4a and 2.4b
- **WHEN** an operator opens the script header, `docs/operator/JETSON_SMOKE_RUNBOOK.md` or
  `docs/runbooks/pc-to-jetson-promotion.md`
- **THEN** each says: do not run `deploy_remote.sh` with default paths against the rover; deliver
  by git-fetch

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
- **THEN** `systemctl restart mousedroid-docker` runs — its `ExecStartPre` re-syncs the overlay and
  runs `preflight_check.sh` — and the health check is
  `docker exec -w /opt/mousedroid mousedroid python3 -m mousedroid.main --health-check`
- **AND** when the unit is absent, no restart is attempted and the run still exits 0

## Requirement: configuration MUST NOT be installed onto a tree that cannot interpret it safely (code before config)

A rover still running the `032942b` image has no 1.0 V implausibility floor; installing CF-1
thresholds there trips `battery_critical` on a missing reading (review focus 3).

### Scenario: `--config-only` refuses an old tree
- **GIVEN** a remote tree whose `src/mousedroid/config/schema/reward_safety.py` lacks
  `battery_implausible_below_v`
- **WHEN** `bash scripts/deploy_remote.sh --config-only` runs
- **THEN** it exits non-zero with "deploy code first" on stderr and installs nothing

### Scenario: the bench is ordered the same way
- **GIVEN** bench step 3.3
- **WHEN** trunk is delivered by git-fetch
- **THEN** `sudo bash scripts/sync_jetson_overlay.sh` runs only after `import mousedroid.main`
  succeeds on the new tree

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

## Requirement: the config-compat pin MUST be re-pinned after merge and before branch cleanup, with its new strictness pre-checked

### Scenario: pre-check before the re-pin (R3-7)
- **GIVEN** the candidate trunk SHA
- **WHEN** `python scripts/check_config_compat.py --platform jetson --deployments-dir <tmp> --changed-files config/*.yaml`
  runs against a copy of the record pointing at that SHA
- **THEN** every overlay passes under `extra="forbid"` before the tracked record is edited

### Scenario: ordering
- **GIVEN** PR-H merged
- **WHEN** 4.5 runs
- **THEN** `deployments/jetson-image.json` is re-pinned to the deployed trunk SHA and its false
  "NOT an ancestor of the default branch" sentence is dropped
- **AND** `bash scripts/repin_tags.sh` tags the SHA (the push is an operator action) before any
  `bash scripts/archive_stale_branches.sh` run
- **AND** `tests/regression/test_f051_backwards_compat.py::test_the_record_sha_was_not_repinned_in_this_change`
  is inverted with a "post-merge trunk commit, ancestor of the default branch" docstring
