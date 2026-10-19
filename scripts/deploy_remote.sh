#!/bin/bash
# Remote deployment orchestrator for MouseDroid Jetson Nano.
# Runs from the local machine (WSL2/Linux/macOS) to deploy code to the Jetson.
#
# Usage: bash scripts/deploy_remote.sh [jetson-host] [--full|--code-only|--config-only]
#                                     [--confirm-dirty]
#        bash scripts/deploy_remote.sh --print-rsync-args
#
# Modes:
#   --code-only    (default) Sync the repo to the rover's checkout root, refresh the package in
#                  the bare-metal venv if there is one, then restart the Docker unit (and the
#                  bare-metal one, if that is enabled). The unit restart RECREATES the container
#                  (see MOUSEDROID_REMOTE_RESTART_MODE). It never runs deploy_jetson.sh (that is
#                  --full) and does NOT rebuild the image: the Docker spine serves the package
#                  from the bind mount, and a dependency change needs 'docker compose build'
#                  on the rover.
#   --config-only  Install config/*.yaml and restart. Refuses a remote tree that predates the
#                  battery plausibility floor (F-025): deploy code first, config second.
#   --full         The bare-metal venv path: it runs deploy_jetson.sh WITHOUT --container.
#                  Do not run it against the Docker-spine rover.
#
# Host resolution order:
#   1. First positional argument
#   2. ~/.mousedroid/jetson_host file
#   3. mDNS/avahi discovery via jetson_discover.sh
#
# Rover-local work (F-051): the sync below is `rsync --delete`, so it is fenced
# by scripts/rover_wip_guard.sh. A dirty target REFUSES the sync. With
# --confirm-dirty the operator's uncommitted work is first archived off the
# rover and committed to a rover/wip-<date> branch, and only then synced over.
# `git clean` is never run, and rsync-delete never runs over unpreserved work.
#
# Environment variables (all optional):
#   MOUSEDROID_REMOTE_USER            SSH user (default: jetson)
#   MOUSEDROID_REMOTE_SRC             Rsync destination on the rover: the repo root, the
#                                     checkout the compose bind mount and both systemd units
#                                     run from (default: /opt/mousedroid)
#   MOUSEDROID_REMOTE_RESTART_MODE    How the Docker spine is restarted: unit (default) or
#                                     container. The rule: where the mousedroid-docker unit is
#                                     enabled, the unit restarts it; with no unit, container
#                                     restarts it in place. unit runs systemctl restart
#                                     mousedroid-docker, which RECREATES the container (the unit's
#                                     ExecStop is docker compose down): anything hot-installed
#                                     into its writable layer is lost, and the camera may need a
#                                     host reboot after a recreate. Its ExecStartPre re-syncs the
#                                     overlay and runs the preflight, fail-closed. A rover that
#                                     runs the container under compose alone (no unit) is left
#                                     running in unit mode, and the log says so. container runs
#                                     docker restart in place, which keeps the writable layer, and
#                                     is REFUSED while the unit is active: the unit is attached to
#                                     the container (docker compose up --abort-on-container-exit,
#                                     Restart=on-failure), so an in-place restart would abort it
#                                     and systemd would recreate the container ten seconds later.
#                                     container also stays on the image the container was created
#                                     from, which only a recreate replaces: not after a docker
#                                     compose build.
#   MOUSEDROID_REMOTE_HEALTH_WAIT_S   Seconds of wall-clock time to wait, polling once a second,
#                                     for the container to be listed after THIS run restarted the
#                                     Docker unit (default: 30; 0 probes once). Never listed is a
#                                     WARNING. A run that restarted no unit probes once, no wait.
#   MOUSEDROID_CONFIG_DIR             Remote config dir (default: /etc/mousedroid)
#   MOUSEDROID_DEPLOY_CONFIRM_DIRTY   1/true = same as --confirm-dirty
#   MOUSEDROID_DEPLOY_ARCHIVE_DIR     Where rover WIP archives land on THIS
#                                     machine (default: ~/.mousedroid/rover-wip)
#   MOUSEDROID_ROVER_WIP_BRANCH       Forwarded to the guard (branch override)
#   MOUSEDROID_ROVER_WIP_DATE         Forwarded to the guard (date stamp)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
REMOTE_USER="${MOUSEDROID_REMOTE_USER:-jetson}"
# Env-overridable, same idiom as docker_deploy.sh's MOUSEDROID_INSTALL_DIR /
# MOUSEDROID_CONFIG_DIR. Making them knobs is what lets the WIP guard below be
# exercised against a throwaway checkout in
# tests/unit/scripts/test_deploy_remote_guard.py instead of only on a rover.
# REMOTE_SRC defaults to the rover's repo ROOT, the checkout that the compose bind mount
# and both systemd units run from (F-059, peer-review R3-3). It used to be the package
# parent one level down, /opt/mousedroid/src: syncing the whole repo there nested a second
# copy inside the first, and --delete removed the live package directory beneath it.
REMOTE_SRC="${MOUSEDROID_REMOTE_SRC:-/opt/mousedroid}"
REMOTE_CONFIG="${MOUSEDROID_CONFIG_DIR:-/etc/mousedroid}"
# How long run_health_check waits, in wall-clock seconds and polling once a second, for the
# container to be listed. Only when THIS run restarted the Docker unit: that restart returns
# before compose has the container up, so a probe made at once finds nothing. When it did not,
# nothing is bringing a container up and the check looks once. Validated below, because shell
# arithmetic reads it.
REMOTE_HEALTH_WAIT_S="${MOUSEDROID_REMOTE_HEALTH_WAIT_S:-30}"
# How restart_service restarts the Docker spine: unit or container (validated below). unit
# restarts mousedroid-docker where that unit is enabled, which RECREATES the container;
# container restarts it in place, for a rover with no unit, and refuses while the unit is active.
REMOTE_RESTART_MODE="${MOUSEDROID_REMOTE_RESTART_MODE:-unit}"
# The rover's install root, and the venv deploy_jetson.sh creates under it. One
# constant instead of the literal repeated at each use. It is deliberately NOT a
# knob: the install that actually runs is fixed by scripts/mousedroid.service
# (WorkingDirectory and ExecStart name /opt/mousedroid and its venv, and
# deploy_jetson.sh installs that unit as-is) and by the compose bind mount. A
# knob that moved only the venv would reinstall and health-check one tree while
# restart_service restarted the other, and report success. Pinned to the
# unit's ExecStart by tests/unit/scripts/test_deploy_remote_guard.py.
REMOTE_INSTALL_DIR=/opt/mousedroid
REMOTE_VENV="${REMOTE_INSTALL_DIR}/venv"
# The container the Docker spine runs. Not a knob either: docker-compose.jetson.yml pins
# container_name, and docker exec needs the real name. Pinned to the compose file by
# tests/unit/scripts/test_deploy_remote_guard.py.
REMOTE_CONTAINER=mousedroid

# F-059 (R-2/R3-3/R3-4): `rsync --delete` must never remove rover runtime state.
# The explicit excludes are the fence: the `:- .gitignore` merge alone protects nothing
# on a first deploy (the rover has no .gitignore yet) and afterwards only what the
# rover's copy of it names.
#   FENCED: the seven runtime roots, anchored to the transfer root and spelt without a
#     trailing slash (/venv ... /tensorrt_cache): never shipped, never deleted. A trailing
#     slash would make a pattern match directories only, and a root that is a symlink
#     (models/ pointing at NVMe) would then be deleted. A same-named directory deeper in
#     the tree (docs/reports, src/mousedroid/models) is source and ships.
#   SHIPS: every working-tree file `.gitignore` does not ignore, plus every .gitkeep
#     sentinel, except the nine dev-tool names excluded below (.git, __pycache__, .venv,
#     ...), which never ship at any depth. The merge drops `!` re-includes, so the
#     first rule restores sentinels; it must precede the merge, and it wins only where the
#     ignore is file-level (`dir/*`) because rsync never descends into an ignored directory.
#   DOES NOT SHIP (accepted): anything tracked under a root (reports/*.json and the
#     reports/endurance/.gitkeep sentinel) and any other tracked file `.gitignore`
#     ignores or re-includes with `!` (.vscode/*.json, docs/planning/SMOKE_REPORT_TEMPLATE.md).
# Pinned in tests/unit/scripts/test_deploy_remote_guard.py.
RSYNC_PROTECT_ARGS=(
    --filter='+ **/.gitkeep'
    --filter=':- .gitignore'
    --exclude '.git' --exclude '__pycache__' --exclude '.venv' --exclude '*.egg-info'
    --exclude 'node_modules' --exclude '.mypy_cache' --exclude '.pytest_cache'
    --exclude '.ruff_cache' --exclude '.claude'
    --exclude '/venv' --exclude '/weights' --exclude '/models' --exclude '/logs'
    --exclude '/reports' --exclude '/estop_latch' --exclude '/tensorrt_cache'
)

DEPLOY_MODE="code-only"
HOST=""
# Set to 1 by restart_service when it restarts the Docker UNIT, the restart that recreates the
# container and returns before compose has it up. run_health_check waits for the container only
# then.
DOCKER_UNIT_RESTARTED=0
# Set to true by the Docker-spine helpers in restart_service when they restarted something or said
# why they did not, so that restart_service does not also print its "skipping restart" line.
SPINE_HANDLED=false
# What restart_service found when it looked at the container and then did nothing that changes
# it: "" if it did not look (or restarted the container, so it must be looked at again), else
# probe_container's status. run_health_check reuses it rather than asking the rover twice.
CONTAINER_LOOK_RC=""
WIP_GUARD="${SCRIPT_DIR}/rover_wip_guard.sh"
ARCHIVE_DIR="${MOUSEDROID_DEPLOY_ARCHIVE_DIR:-${HOME}/.mousedroid/rover-wip}"

# Guard exit codes — mirrored from rover_wip_guard.sh's documented contract.
GUARD_EXIT_DIRTY=3

case "${MOUSEDROID_DEPLOY_CONFIRM_DIRTY:-}" in
    1|true|TRUE|yes|YES) CONFIRM_DIRTY=true ;;
    *)                   CONFIRM_DIRTY=false ;;
esac

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ts() {
    date "+%Y-%m-%d %H:%M:%S"
}

log_section() {
    echo ""
    echo "=== $1 === [$(ts)]"
}

log_step() {
    echo "--- $1 ---"
}

die() {
    echo "ERROR: $1" >&2
    exit 1
}

# Refuse a remote path that could break out of the command string it lands in.
#
# REMOTE_SRC and REMOTE_CONFIG became environment-overridable so the WIP guard
# could be exercised against a throwaway checkout instead of only on a rover.
# They are also interpolated into remote command STRINGS that run under sudo
# (`remote_sudo bash -c "mkdir -p ..."`, `remote_cmd "test -x .../..."`), so an
# override like `/opt/mousedroid; rm -rf /` would execute on the rover as root.
# Before this override existed both were literals and the interpolation was
# safe; it is the override that made them untrusted input.
#
# Two defences, because either alone is thin: this validator rejects anything
# that is not a plain absolute path, and every remote-command boundary below
# quotes with `printf %q` (the idiom `remote_guard` already uses for its env
# values). Fails closed — an unusable path stops the deploy rather than
# silently deploying somewhere else.
require_safe_remote_path() {
    local name="$1" value="$2"
    [[ -n "${value}" ]] || die "${name} must not be empty"
    [[ "${value}" == /* ]] || die "${name} must be an absolute path, got: ${value}"
    case "${value}" in
        *[\;\&\|\$\`\(\)\<\>\!$'\n'$'\r'$'\t'$'\v'$'\f']*|*'"'*|*"'"*|*' '*)
            die "${name} contains characters unsafe for a remote command: ${value}"
            ;;
    esac
    # Separate arm: the previous single-arm form spelt this as *'\\'*, and a
    # QUOTED backslash pair in a case pattern matches TWO literal backslashes,
    # so a lone backslash passed. Unquoted \\ matches exactly one.
    case "${value}" in
        *\\*) die "${name} must not contain a backslash: ${value}" ;;
    esac
    # Glob metacharacters are not shell metacharacters, so the arm above never
    # caught them -- but the REMOTE shell pathname-expands an unquoted value,
    # turning `mkdir -p /etc/mousedroid*` into whatever happens to match.
    case "${value}" in
        *[*?]*|*'['*|*']'*)
            die "${name} must not contain glob characters (* ? [ ]): ${value}"
            ;;
    esac
    case "${value}" in
        */../*|*/..) die "${name} must not contain a '..' segment: ${value}" ;;
    esac
}

require_safe_remote_path "MOUSEDROID_REMOTE_SRC" "${REMOTE_SRC}"
require_safe_remote_path "MOUSEDROID_CONFIG_DIR" "${REMOTE_CONFIG}"

# A short run of digits only: the wait is read by shell arithmetic, where a padded 08 is a
# bad octal literal. Forcing base 10 below also drops the padding from the message.
[[ "${REMOTE_HEALTH_WAIT_S}" =~ ^[0-9]{1,4}$ ]] \
    || die "MOUSEDROID_REMOTE_HEALTH_WAIT_S must be a whole number of seconds (0-9999), got: ${REMOTE_HEALTH_WAIT_S}"
REMOTE_HEALTH_WAIT_S=$((10#${REMOTE_HEALTH_WAIT_S}))

case "${REMOTE_RESTART_MODE}" in
    unit|container) ;;
    *) die "MOUSEDROID_REMOTE_RESTART_MODE must be 'unit' or 'container', got: ${REMOTE_RESTART_MODE}" ;;
esac

# Pre-quoted forms for interpolation into remote command strings. Use these,
# not the raw variables, anywhere the value crosses into a shell on the rover.
REMOTE_SRC_Q="$(printf '%q' "${REMOTE_SRC}")"
REMOTE_CONFIG_Q="$(printf '%q' "${REMOTE_CONFIG}")"
REMOTE_VENV_Q="$(printf '%q' "${REMOTE_VENV}")"
REMOTE_INSTALL_DIR_Q="$(printf '%q' "${REMOTE_INSTALL_DIR}")"

remote_cmd() {
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "${REMOTE_USER}@${HOST}" "$@"
}

remote_sudo() {
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "${REMOTE_USER}@${HOST}" sudo -- "$@"
}

# Run scripts/rover_wip_guard.sh ON THE ROVER by piping it over ssh.
#
# Piped rather than invoked from ${REMOTE_SRC}: the guard has to run BEFORE the
# sync that would put it there, so it cannot be assumed present, and a stale
# copy already on the rover is exactly the wrong thing to trust with this
# decision. The local copy is the one that ships.
#
# The remote arg list is %q-quoted because ssh flattens its arguments into one
# string that the remote shell re-parses; passing "$@" raw would split on any
# space in a path. The guard's stdout is left alone (it carries the archive in
# --archive - mode); its messages are on stderr.
remote_guard() {
    local env_prefix=""
    if [[ -n "${MOUSEDROID_ROVER_WIP_BRANCH:-}" ]]; then
        env_prefix+="MOUSEDROID_ROVER_WIP_BRANCH=$(printf '%q' "${MOUSEDROID_ROVER_WIP_BRANCH}") "
    fi
    if [[ -n "${MOUSEDROID_ROVER_WIP_DATE:-}" ]]; then
        env_prefix+="MOUSEDROID_ROVER_WIP_DATE=$(printf '%q' "${MOUSEDROID_ROVER_WIP_DATE}") "
    fi
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "${REMOTE_USER}@${HOST}" \
        "${env_prefix}bash -s -- $(printf '%q ' "$@")" < "${WIP_GUARD}"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --full)
                DEPLOY_MODE="full"
                shift
                ;;
            --code-only)
                DEPLOY_MODE="code-only"
                shift
                ;;
            --config-only)
                DEPLOY_MODE="config-only"
                shift
                ;;
            --confirm-dirty)
                CONFIRM_DIRTY=true
                shift
                ;;
            --print-rsync-args)
                # One argument per line, so a test or an operator can read the exact
                # fence with no host and no network. Handled here, ahead of host
                # resolution, which is why it works on a machine that cannot reach a rover.
                printf '%s\n' "${RSYNC_PROTECT_ARGS[@]}"
                exit 0
                ;;
            --help|-h)
                echo "Usage: bash scripts/deploy_remote.sh [jetson-host] [--full|--code-only|--config-only]"
                echo "                                    [--confirm-dirty]"
                echo ""
                echo "Modes:"
                echo "  --full         Bare-metal venv path (deploy_jetson.sh WITHOUT --container):"
                echo "                 system + hardware + rsync + deploy + restart."
                echo "                 Do not run it against the Docker-spine rover."
                echo "  --code-only    (default) Rsync code to the checkout root, pip reinstall into the"
                echo "                 venv if there is one, restart the Docker unit (which recreates the"
                echo "                 container, see below). Never runs deploy_jetson.sh (that is --full)."
                echo "                 Does NOT rebuild the image: a dependency change needs"
                echo "                 'docker compose build' on the rover."
                echo "  --config-only  Update config files + restart service. Refuses a remote tree"
                echo "                 that predates the battery plausibility floor: deploy code first."
                echo ""
                echo "Restart (MOUSEDROID_REMOTE_RESTART_MODE):"
                echo "  unit (default)  systemctl restart mousedroid-docker, where that unit is enabled."
                echo "                  This RECREATES the container (the unit's ExecStop is docker compose"
                echo "                  down): packages hot-installed into it are lost, and the camera may"
                echo "                  need a host reboot. Its ExecStartPre syncs the overlay and runs the"
                echo "                  preflight, fail-closed. A rover that runs the container under compose"
                echo "                  alone (no unit) is left running, and the log says so."
                echo "  container       docker restart in place, for that compose-only rover: keeps the"
                echo "                  container's writable layer. REFUSED while the unit is active: it is"
                echo "                  attached to the container, so an in-place restart would abort it and"
                echo "                  systemd would recreate the container. It stays on the image the"
                echo "                  container was created from, so not after a docker compose build."
                echo ""
                echo "Rover-local work:"
                echo "  A dirty sync target is REFUSED (rsync --delete would destroy it)."
                echo "  --confirm-dirty  Archive the rover's uncommitted work off-device and"
                echo "                   commit it to rover/wip-<date>, then sync."
                echo ""
                echo "Inspection:"
                echo "  --print-rsync-args  Print the rsync fence (what --delete is kept off), one"
                echo "                      argument per line, and exit. Needs no host."
                exit 0
                ;;
            -*)
                die "Unknown option: $1"
                ;;
            *)
                if [[ -z "${HOST}" ]]; then
                    HOST="$1"
                else
                    die "Unexpected argument: $1"
                fi
                shift
                ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Host resolution
# ---------------------------------------------------------------------------

resolve_host() {
    if [[ -n "${HOST}" ]]; then
        log_step "Using host from argument: ${HOST}"
        return
    fi

    local host_file="${HOME}/.mousedroid/jetson_host"
    if [[ -f "${host_file}" ]]; then
        HOST="$(cat "${host_file}" | tr -d '[:space:]')"
        if [[ -n "${HOST}" ]]; then
            log_step "Using host from ${host_file}: ${HOST}"
            return
        fi
    fi

    if [[ -x "${SCRIPT_DIR}/jetson_discover.sh" ]]; then
        log_step "Attempting mDNS/avahi discovery..."
        HOST="$("${SCRIPT_DIR}/jetson_discover.sh" 2>/dev/null || true)"
        if [[ -n "${HOST}" ]]; then
            log_step "Discovered Jetson at: ${HOST}"
            mkdir -p "${HOME}/.mousedroid"
            echo "${HOST}" > "${host_file}"
            return
        fi
    fi

    die "Cannot determine Jetson host. Provide as argument or save to ~/.mousedroid/jetson_host"
}

# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------

check_connectivity() {
    log_step "Checking SSH connectivity to ${REMOTE_USER}@${HOST}..."
    if ! remote_cmd "echo ok" &>/dev/null; then
        die "Cannot SSH to ${REMOTE_USER}@${HOST}. Check connectivity and SSH keys."
    fi
    log_step "SSH connection verified"
}

# ---------------------------------------------------------------------------
# Rsync project code
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rover WIP preservation (F-051 / campaign step B1) — runs BEFORE any rsync
# ---------------------------------------------------------------------------
# Two artifacts, not one. The rover/wip-<date> branch keeps the operator
# working; the off-device archive is what survives the rover. A microSD that
# dies takes the branch with it, so a branch alone is not preservation.
#
# Fail-closed everywhere: an unreachable guard, an unreadable target, an
# unverifiable archive and an unconfirmed dirty target all stop the run before
# rsync --delete can touch anything.
preserve_remote_wip() {
    log_step "Checking the sync target for uncommitted rover-local work"
    if [[ ! -f "${WIP_GUARD}" ]]; then
        die "WIP guard missing: ${WIP_GUARD} — refusing to rsync --delete unguarded"
    fi

    local guard_status=0
    remote_guard inspect "${REMOTE_SRC}" || guard_status=$?

    if [[ "${guard_status}" -eq 0 ]]; then
        log_step "Sync target is clean — proceeding"
        return 0
    fi

    if [[ "${guard_status}" -ne "${GUARD_EXIT_DIRTY}" ]]; then
        die "WIP guard could not clear the sync target (exit ${guard_status}). \
Refusing to rsync --delete. Fix the reported cause on the rover \
(ownership drift is the usual one) and re-run; do not bypass this."
    fi

    if [[ "${CONFIRM_DIRTY}" != true ]]; then
        die "REFUSING: ${REMOTE_USER}@${HOST}:${REMOTE_SRC} has uncommitted work \
and rsync --delete would destroy it. Re-run with --confirm-dirty (or \
MOUSEDROID_DEPLOY_CONFIRM_DIRTY=1) to archive it off the rover and commit it \
to a rover/wip-<date> branch first."
    fi

    mkdir -p "${ARCHIVE_DIR}"
    local archive
    archive="${ARCHIVE_DIR}/rover-wip-$(date -u '+%Y%m%dT%H%M%SZ').tar.gz"
    log_step "Archiving rover-local work off-device -> ${archive}"
    if ! remote_guard preserve "${REMOTE_SRC}" --archive - > "${archive}"; then
        rm -f "${archive}"
        die "Preserving rover-local work FAILED — nothing has been synced."
    fi

    # Verify the transfer landed intact before anything destructive runs. A
    # truncated stream is a plausible ssh failure mode and a truncated archive
    # is not a backup.
    if [[ ! -s "${archive}" ]] || ! tar -tzf "${archive}" >/dev/null 2>&1; then
        die "Rover WIP archive is empty or corrupt: ${archive} — refusing to sync."
    fi
    local member
    for member in ./status.txt ./head.txt ./diff-ignore-whitespace.patch ./untracked.tar; do
        if ! tar -tzf "${archive}" | grep -qx -- "${member}"; then
            die "Rover WIP archive is missing ${member}: ${archive} — refusing to sync."
        fi
    done
    log_step "Rover WIP archived and verified: ${archive}"
}

rsync_code() {
    log_section "Syncing project code"
    # MUST stay the first statement here: both --full and --code-only reach the
    # rsync below through this function.
    preserve_remote_wip

    log_step "Ensuring remote directory exists"
    remote_sudo bash -c "mkdir -p ${REMOTE_SRC_Q} && chown -R ${REMOTE_USER}:${REMOTE_USER} ${REMOTE_SRC_Q}"

    log_step "Rsyncing ${PROJECT_DIR} -> ${REMOTE_USER}@${HOST}:${REMOTE_SRC}/"
    rsync -avz --delete \
        "${RSYNC_PROTECT_ARGS[@]}" \
        "${PROJECT_DIR}/" "${REMOTE_USER}@${HOST}:${REMOTE_SRC_Q}/"
    log_step "Rsync complete"
}

# ---------------------------------------------------------------------------
# Deploy config overlays
# ---------------------------------------------------------------------------

# Config is written against the newest schema, so it must never land ahead of the code that
# understands it. safety.battery_implausible_below_v (the F-025 plausibility floor) is the
# marker: a tree that predates it reads a missing battery voltage as 0.0 V, so an overlay
# that arms the battery thresholds would latch an e-stop on it. Refuse to install YAML onto
# such a tree. After the sync (--code-only, --full) the probe passes by construction; for
# --config-only it is the only protection, and the remedy it names is: deploy code first.
# ssh's own exit status 255 (cannot connect) is not a stale tree and gets its own message. The
# probe's path is pinned to this tree by tests/unit/scripts/test_deploy_remote_guard.py.
require_remote_safety_floor() {
    local rc=0
    remote_cmd "grep -q battery_implausible_below_v ${REMOTE_SRC_Q}/src/mousedroid/config/schema/reward_safety.py" || rc=$?
    case "${rc}" in
        0) ;;
        255) die "cannot reach the rover at ${HOST}; nothing installed" ;;
        *) die "remote source at ${REMOTE_SRC} predates safety.battery_implausible_below_v — deploy code first (--code-only or git fetch), then config" ;;
    esac
}

deploy_config() {
    require_remote_safety_floor
    log_section "Deploying configuration"
    remote_sudo bash -c "mkdir -p ${REMOTE_CONFIG_Q}"

    log_step "Copying config files to ${REMOTE_CONFIG}/"
    local config_dir="${PROJECT_DIR}/config"
    if [[ -d "${config_dir}" ]]; then
        for cfg_file in "${config_dir}"/*.yaml; do
            [[ -f "${cfg_file}" ]] || continue
            local basename basename_q
            basename="$(basename "${cfg_file}")"
            log_step "  -> ${basename}"
            scp -o ConnectTimeout=10 "${cfg_file}" "${REMOTE_USER}@${HOST}:/tmp/${basename}"
            basename_q="$(printf '%q' "${basename}")"
            remote_sudo bash -c "cp -n /tmp/${basename_q} ${REMOTE_CONFIG_Q}/${basename_q}"
            remote_cmd rm -f "/tmp/${basename}"
        done
    fi
    log_step "Config deployment complete"
}

# ---------------------------------------------------------------------------
# Remote system + hardware setup (full mode only)
# ---------------------------------------------------------------------------

run_system_setup() {
    log_section "Running Jetson system setup"
    if remote_cmd "test -x ${REMOTE_SRC_Q}/scripts/jetson_system_setup.sh"; then
        remote_sudo bash "${REMOTE_SRC_Q}/scripts/jetson_system_setup.sh"
    else
        log_step "SKIP: jetson_system_setup.sh not found on remote"
    fi
}

run_hardware_setup() {
    log_section "Running Jetson hardware setup"
    if remote_cmd "test -x ${REMOTE_SRC_Q}/scripts/jetson_hardware_setup.sh" 2>/dev/null || \
       remote_cmd "test -f ${REMOTE_SRC_Q}/scripts/jetson_hardware_setup.sh" 2>/dev/null; then
        remote_sudo bash "${REMOTE_SRC_Q}/scripts/jetson_hardware_setup.sh"
    else
        log_step "SKIP: jetson_hardware_setup.sh not found on remote"
    fi
}

# ---------------------------------------------------------------------------
# Remote deploy (pip install + systemd)
# ---------------------------------------------------------------------------

# The bare-metal installer (no --container). Only --full calls it; --code-only never does.
run_deploy() {
    log_section "Running deploy_jetson.sh on remote"
    remote_sudo bash "${REMOTE_SRC_Q}/scripts/deploy_jetson.sh"
}

# ---------------------------------------------------------------------------
# Pip reinstall (code-only mode)
# ---------------------------------------------------------------------------
# --code-only never runs the bare-metal installer. deploy_jetson.sh without --container is
# --full's (apt, a venv, PyTorch, a systemd unit, a health check against live hardware), and
# building one beside the container that already serves the rover would leave two runtimes for
# one set of hardware. So a missing venv is not a cue to install one:
#   venv present  -> refresh the editable install (the bare-metal rover).
#   Docker unit   -> nothing to reinstall. The container imports the package from the bind
#                    mount, so the synced tree is already what runs. The image is NOT rebuilt:
#                    a dependency change needs 'docker compose build' on the rover.
#   neither       -> warn and carry on; there is nothing here to reinstall into.

pip_reinstall() {
    log_section "Reinstalling mousedroid package"
    if remote_cmd "test -d ${REMOTE_VENV_Q}"; then
        local pip_target_q
        pip_target_q="$(printf '%q' "${REMOTE_SRC}[hardware,jetson]")"
        remote_sudo "${REMOTE_VENV_Q}/bin/pip" install --quiet -e "${pip_target_q}"
    elif remote_sudo systemctl is-enabled mousedroid-docker &>/dev/null; then
        log_step "No venv: the Docker spine serves the package from the bind mount (image not rebuilt; dependency changes need 'docker compose build' on the rover)"
    else
        echo "WARNING: no venv and no mousedroid-docker unit on the rover — nothing to reinstall (the bare-metal path is --full)"
    fi
}

# ---------------------------------------------------------------------------
# Service restart
# ---------------------------------------------------------------------------

# The Docker unit is attached to the container for as long as it is active: it runs
# docker compose up --abort-on-container-exit in the foreground (Type=exec) with
# Restart=on-failure. An in-place docker restart stops the container compose is attached to, so
# compose aborts, the unit fails, and systemd runs it again ten seconds later through compose
# down: a recreate by detour, with two supervisors fighting over the container in between. So
# container mode refuses while the unit is active, and main asks before anything is synced.
require_no_attached_unit() {
    local rc=0
    remote_sudo systemctl is-active mousedroid-docker &>/dev/null || rc=$?
    case "${rc}" in
        0) die "the mousedroid-docker unit is active and attached to the container; an in-place restart would abort it and trigger a recreate — use MOUSEDROID_REMOTE_RESTART_MODE=unit" ;;
        255) die "cannot reach the rover at ${HOST}; nothing restarted" ;;
    esac
}

# unit mode. Where the unit is enabled, restart it: that RECREATES the container (its ExecStop is
# docker compose down), and its ExecStartPre re-syncs the overlay and runs preflight_check.sh, so
# the synced tree goes live through the fail-closed path. With no unit but the container
# running, the container is under compose alone: leave it, and say how to restart it. These
# helpers are called as plain statements, never in an if or && list, so a failed restart still
# stops the deploy (errexit is off inside anything called from a condition).
restart_docker_unit() {
    if remote_sudo systemctl is-enabled mousedroid-docker &>/dev/null; then
        remote_sudo systemctl restart mousedroid-docker
        DOCKER_UNIT_RESTARTED=1
        SPINE_HANDLED=true
        log_step "mousedroid-docker: $(remote_sudo systemctl is-active mousedroid-docker 2>/dev/null || true)"
    else
        CONTAINER_LOOK_RC=0
        probe_container || CONTAINER_LOOK_RC=$?
        if [[ "${CONTAINER_LOOK_RC}" -eq 0 ]]; then
            SPINE_HANDLED=true
            log_step "Container ${REMOTE_CONTAINER} runs under compose only (no mousedroid-docker unit is enabled), so it was not restarted. MOUSEDROID_REMOTE_RESTART_MODE=container restarts it in place and keeps its writable layer"
        fi
    fi
}

# container mode: docker restart in place, for the rover that runs its container under compose
# alone (restart: unless-stopped). The writable layer survives, so packages hot-installed into
# it are kept, and so does the image it was created from (only a recreate picks up a rebuilt
# one). No ExecStartPre runs: the deploy has run the floor guard and goes on to run the health
# check. Not gated on the unit being enabled, but refused while it is active.
restart_container_in_place() {
    require_no_attached_unit
    local rc=0
    probe_container || rc=$?
    case "${rc}" in
        0)
            remote_cmd "docker restart ${REMOTE_CONTAINER}"
            CONTAINER_LOOK_RC=""
            log_step "Container ${REMOTE_CONTAINER} restarted in place (its writable layer is kept)"
            ;;
        255)
            die "cannot reach the rover at ${HOST}; nothing restarted"
            ;;
        *)
            CONTAINER_LOOK_RC="${rc}"
            echo "WARNING: container ${REMOTE_CONTAINER} is not running — nothing to restart in place"
            ;;
    esac
    SPINE_HANDLED=true
}

restart_service() {
    log_section "Restarting mousedroid service(s)"
    remote_sudo systemctl daemon-reload
    SPINE_HANDLED=false
    if [[ "${REMOTE_RESTART_MODE}" == container ]]; then
        restart_container_in_place
    else
        restart_docker_unit
    fi
    local restarted="${SPINE_HANDLED}"
    if remote_sudo systemctl is-enabled mousedroid &>/dev/null; then
        remote_sudo systemctl restart mousedroid; restarted=true; sleep 2
        log_step "mousedroid: $(remote_sudo systemctl is-active mousedroid 2>/dev/null || true)"
    fi
    [[ "${restarted}" == true ]] || log_step "No mousedroid unit enabled — skipping restart"
}

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

# One probe: is REMOTE_CONTAINER listed? 0 yes, 1 no, 255 the rover cannot be reached (ssh's own).
probe_container() {
    remote_cmd "docker ps --filter name=^/${REMOTE_CONTAINER}\$ --format '{{.Names}}' | grep -qx ${REMOTE_CONTAINER}"
}

# Probe until the container is listed or REMOTE_HEALTH_WAIT_S seconds of wall-clock time have
# passed (SECONDS), once a second. The bound is time, not a poll count, so slow probes cannot
# stretch it; 0 probes once and never sleeps. It gives up at once on 255: an unreachable rover
# does not become reachable by waiting here. Returns the last probe's status (0 listed, 1 not).
wait_for_container() {
    local deadline=$((SECONDS + REMOTE_HEALTH_WAIT_S)) rc=0
    while true; do
        probe_container && return 0
        rc=$?
        (( rc != 255 )) || return 255
        (( SECONDS < deadline )) || return "${rc}"
        sleep 1
    done
}

# The wait applies only when this run restarted the Docker unit (DOCKER_UNIT_RESTARTED): that
# restart returns before compose has the container up. Otherwise nothing is bringing one up, so
# the container is looked at once (or the look restart_service just made is reused) and an
# unlisted one is an info line, not a warning. A container that is listed gets its health check
# either way.
run_health_check() {
    log_section "Running remote health check"
    local probe_rc=0
    if [[ "${DOCKER_UNIT_RESTARTED}" == 1 ]]; then
        wait_for_container || probe_rc=$?
    elif [[ -n "${CONTAINER_LOOK_RC}" ]]; then
        probe_rc="${CONTAINER_LOOK_RC}"
    else
        probe_container || probe_rc=$?
    fi
    case "${probe_rc}" in
        0)
            remote_cmd "docker exec -w ${REMOTE_INSTALL_DIR_Q} ${REMOTE_CONTAINER} python3 -m mousedroid.main --health-check" \
                || echo "WARNING: container health check returned non-zero exit code"
            ;;
        255)
            echo "WARNING: cannot reach the rover at ${HOST} — container health check skipped"
            ;;
        *)
            if [[ "${DOCKER_UNIT_RESTARTED}" == 1 ]]; then
                echo "WARNING: container ${REMOTE_CONTAINER} not running after ${REMOTE_HEALTH_WAIT_S}s — container health check skipped"
            else
                log_step "Container ${REMOTE_CONTAINER} is not running and this run did not restart the Docker unit — container health check skipped"
            fi
            ;;
    esac
    if remote_cmd "test -d ${REMOTE_VENV_Q}"; then
        remote_cmd "MOUSEDROID_MOCK_HARDWARE=false ${REMOTE_VENV_Q}/bin/python -m mousedroid.main --health-check" \
            || echo "WARNING: Health check returned non-zero exit code"
    fi
}

# ---------------------------------------------------------------------------
# Deployment summary
# ---------------------------------------------------------------------------

print_summary() {
    log_section "Deployment Summary"
    echo "  Host:       ${REMOTE_USER}@${HOST}"
    echo "  Mode:       ${DEPLOY_MODE}"
    echo "  Source:      ${PROJECT_DIR}"
    echo "  Remote src:  ${REMOTE_SRC}"
    echo "  Remote cfg:  ${REMOTE_CONFIG}"
    echo "  Dirty-sync:  confirm=${CONFIRM_DIRTY} (archives -> ${ARCHIVE_DIR})"
    echo "  Started:     ${DEPLOY_START}"
    echo "  Finished:    $(ts)"
    echo ""
    echo "=== Deployment complete ==="
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    parse_args "$@"

    log_section "MouseDroid Remote Deployment"
    DEPLOY_START="$(ts)"

    resolve_host
    check_connectivity
    # Container mode refuses a rover whose unit is attached to the container, and it should
    # refuse before the deploy has changed anything on it.
    if [[ "${REMOTE_RESTART_MODE}" == container ]]; then
        require_no_attached_unit
    fi

    case "${DEPLOY_MODE}" in
        full)
            rsync_code
            run_system_setup
            run_hardware_setup
            run_deploy
            deploy_config
            restart_service
            run_health_check
            ;;
        code-only)
            rsync_code
            pip_reinstall
            deploy_config
            restart_service
            run_health_check
            ;;
        config-only)
            deploy_config
            restart_service
            run_health_check
            ;;
        *)
            die "Unknown deploy mode: ${DEPLOY_MODE}"
            ;;
    esac

    print_summary
}

main "$@"
