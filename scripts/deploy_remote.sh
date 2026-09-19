#!/bin/bash
# Remote deployment orchestrator for MouseDroid Jetson Nano.
# Runs from the local machine (WSL2/Linux/macOS) to deploy code to the Jetson.
#
# Usage: bash scripts/deploy_remote.sh [jetson-host] [--full|--code-only|--config-only]
#                                     [--confirm-dirty]
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
#   MOUSEDROID_REMOTE_SRC             Rsync destination on the rover
#                                     (default: /opt/mousedroid/src)
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
# MOUSEDROID_CONFIG_DIR. Defaults are unchanged; making them knobs is what lets
# the WIP guard below be exercised against a throwaway checkout in
# tests/unit/scripts/test_deploy_remote_guard.py instead of only on a rover.
REMOTE_SRC="${MOUSEDROID_REMOTE_SRC:-/opt/mousedroid/src}"
REMOTE_CONFIG="${MOUSEDROID_CONFIG_DIR:-/etc/mousedroid}"
DEPLOY_MODE="code-only"
HOST=""
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
            --help|-h)
                echo "Usage: bash scripts/deploy_remote.sh [jetson-host] [--full|--code-only|--config-only]"
                echo "                                    [--confirm-dirty]"
                echo ""
                echo "Modes:"
                echo "  --full         Full setup: system + hardware + rsync + deploy + restart"
                echo "  --code-only    (default) Rsync code + pip reinstall + restart service"
                echo "  --config-only  Update config files + restart service"
                echo ""
                echo "Rover-local work:"
                echo "  A dirty sync target is REFUSED (rsync --delete would destroy it)."
                echo "  --confirm-dirty  Archive the rover's uncommitted work off-device and"
                echo "                   commit it to rover/wip-<date>, then sync."
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
    remote_sudo bash -c "mkdir -p ${REMOTE_SRC} && chown -R ${REMOTE_USER}:${REMOTE_USER} ${REMOTE_SRC}"

    log_step "Rsyncing ${PROJECT_DIR} -> ${REMOTE_USER}@${HOST}:${REMOTE_SRC}/"
    rsync -avz --delete \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '.venv' \
        --exclude '*.egg-info' \
        --exclude 'node_modules' \
        --exclude '.mypy_cache' \
        --exclude '.pytest_cache' \
        --exclude '.ruff_cache' \
        --exclude '.claude' \
        "${PROJECT_DIR}/" "${REMOTE_USER}@${HOST}:${REMOTE_SRC}/"
    log_step "Rsync complete"
}

# ---------------------------------------------------------------------------
# Deploy config overlays
# ---------------------------------------------------------------------------

deploy_config() {
    log_section "Deploying configuration"
    remote_sudo mkdir -p "${REMOTE_CONFIG}"

    log_step "Copying config files to ${REMOTE_CONFIG}/"
    local config_dir="${PROJECT_DIR}/config"
    if [[ -d "${config_dir}" ]]; then
        for cfg_file in "${config_dir}"/*.yaml; do
            [[ -f "${cfg_file}" ]] || continue
            local basename
            basename="$(basename "${cfg_file}")"
            log_step "  -> ${basename}"
            scp -o ConnectTimeout=10 "${cfg_file}" "${REMOTE_USER}@${HOST}:/tmp/${basename}"
            remote_sudo cp -n "/tmp/${basename}" "${REMOTE_CONFIG}/${basename}"
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
    if remote_cmd "test -x ${REMOTE_SRC}/scripts/jetson_system_setup.sh"; then
        remote_sudo bash "${REMOTE_SRC}/scripts/jetson_system_setup.sh"
    else
        log_step "SKIP: jetson_system_setup.sh not found on remote"
    fi
}

run_hardware_setup() {
    log_section "Running Jetson hardware setup"
    if remote_cmd "test -x ${REMOTE_SRC}/scripts/jetson_hardware_setup.sh" 2>/dev/null || \
       remote_cmd "test -f ${REMOTE_SRC}/scripts/jetson_hardware_setup.sh" 2>/dev/null; then
        remote_sudo bash "${REMOTE_SRC}/scripts/jetson_hardware_setup.sh"
    else
        log_step "SKIP: jetson_hardware_setup.sh not found on remote"
    fi
}

# ---------------------------------------------------------------------------
# Remote deploy (pip install + systemd)
# ---------------------------------------------------------------------------

run_deploy() {
    log_section "Running deploy_jetson.sh on remote"
    remote_sudo bash "${REMOTE_SRC}/scripts/deploy_jetson.sh"
}

# ---------------------------------------------------------------------------
# Pip reinstall (code-only mode)
# ---------------------------------------------------------------------------

pip_reinstall() {
    log_section "Reinstalling mousedroid package"
    local venv="/opt/mousedroid/venv"
    if remote_cmd "test -d ${venv}"; then
        remote_sudo "${venv}/bin/pip" install --quiet -e "${REMOTE_SRC}[hardware,jetson]"
    else
        log_step "Venv not found — running full deploy_jetson.sh"
        run_deploy
    fi
}

# ---------------------------------------------------------------------------
# Service restart
# ---------------------------------------------------------------------------

restart_service() {
    log_section "Restarting mousedroid service"
    remote_sudo systemctl daemon-reload

    if remote_sudo systemctl is-enabled mousedroid &>/dev/null; then
        remote_sudo systemctl restart mousedroid
        log_step "Service restarted"
        sleep 2
        local status
        status="$(remote_sudo systemctl is-active mousedroid 2>/dev/null || true)"
        log_step "Service status: ${status}"
    else
        log_step "Service not enabled — skipping restart"
    fi
}

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

run_health_check() {
    log_section "Running remote health check"
    local venv="/opt/mousedroid/venv"
    if remote_cmd "test -d ${venv}"; then
        remote_cmd "MOUSEDROID_MOCK_HARDWARE=false ${venv}/bin/python -m mousedroid.main --health-check" || {
            echo "WARNING: Health check returned non-zero exit code"
        }
    else
        log_step "SKIP: venv not found, cannot run health check"
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
