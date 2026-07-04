#!/usr/bin/env bash
# host_bootstrap.sh — durable per-host setup for the Jetson rover (F-017).
#
# Makes the host-side runtime surface survive a reflash/rover swap:
#   1. Seeds ${MOUSEDROID_CONFIG_DIR}/docker.env from the committed template
#      (config/docker.env.example) — ONLY if absent; --force overwrites after
#      writing a timestamped backup; --rollback restores the newest backup.
#   2. Installs scripts/mousedroid-docker.service into /etc/systemd/system
#      (the unit header's manual-cp recipe, automated) + daemon-reload.
#   3. Optional --with-trend-timer: installs mousedroid-trend.{service,timer}
#      and enables the timer (F-018 continuous degradation monitor).
#
# Safety contract:
#   * --dry-run prints the full plan and touches NOTHING (exit 0) — the
#     jetson-runner-install.sh convention.
#   * Every mutation goes through run(), which echoes instead of executing
#     in dry-run mode.
#   * No hardcoded paths: MOUSEDROID_INSTALL_DIR / MOUSEDROID_CONFIG_DIR /
#     SYSTEMD_UNIT_DIR env overrides mirror the docker.env.example keys.
#   * Secrets: this script never reads or prints env-file VALUES; the seeded
#     template ships placeholder keys only (operator fills ANTHROPIC_API_KEY
#     by hand — see docs/runbooks/secret-scanning.md).
#
# Usage:
#   sudo bash scripts/host_bootstrap.sh [--dry-run] [--force] [--rollback]
#                                       [--with-trend-timer]
set -euo pipefail

INSTALL_DIR="${MOUSEDROID_INSTALL_DIR:-/opt/mousedroid}"
CONFIG_DIR="${MOUSEDROID_CONFIG_DIR:-/etc/mousedroid}"
SYSTEMD_UNIT_DIR="${SYSTEMD_UNIT_DIR:-/etc/systemd/system}"

TEMPLATE="${INSTALL_DIR}/config/docker.env.example"
ENV_FILE="${CONFIG_DIR}/docker.env"

DRY_RUN=0
FORCE=0
ROLLBACK=0
WITH_TREND_TIMER=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        --rollback) ROLLBACK=1 ;;
        --with-trend-timer) WITH_TREND_TIMER=1 ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \{0,1\}//' | head -30
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

log() { echo "[host_bootstrap] $*"; }

run() {
    # Single mutation gate: echo the command in dry-run mode, execute otherwise.
    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY-RUN: $*"
    else
        "$@"
    fi
}

if [[ "$ROLLBACK" == "1" ]]; then
    latest_backup="$(ls -1t "${ENV_FILE}.bak."* 2>/dev/null | head -1 || true)"
    if [[ -z "$latest_backup" ]]; then
        log "ERROR: no ${ENV_FILE}.bak.* backup found - nothing to roll back"
        exit 1
    fi
    log "rolling back ${ENV_FILE} from ${latest_backup}"
    run cp -p "$latest_backup" "$ENV_FILE"
    log "rollback complete"
    exit 0
fi

log "install_dir=${INSTALL_DIR} config_dir=${CONFIG_DIR} dry_run=${DRY_RUN}"

# --- 1. Seed / refresh the per-host env file --------------------------------
if [[ ! -f "$TEMPLATE" ]]; then
    log "ERROR: template not found: ${TEMPLATE} (is the repo checked out at ${INSTALL_DIR}?)"
    exit 1
fi

run mkdir -p "$CONFIG_DIR"
if [[ -f "$ENV_FILE" && "$FORCE" != "1" ]]; then
    log "env file exists, leaving untouched (use --force to overwrite): ${ENV_FILE}"
elif [[ -f "$ENV_FILE" && "$FORCE" == "1" ]]; then
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    log "backing up ${ENV_FILE} -> ${ENV_FILE}.bak.${stamp}"
    run cp -p "$ENV_FILE" "${ENV_FILE}.bak.${stamp}"
    log "overwriting ${ENV_FILE} from template"
    run cp "$TEMPLATE" "$ENV_FILE"
    # Holds ANTHROPIC_API_KEY once filled in - never leave it umask-wide.
    # Backups (cp -p) inherit this mode from the tightened original.
    run chmod 600 "$ENV_FILE"
else
    log "seeding ${ENV_FILE} from template"
    run cp "$TEMPLATE" "$ENV_FILE"
    run chmod 600 "$ENV_FILE"
fi

# --- 2. Install the docker service unit -------------------------------------
DOCKER_UNIT="${INSTALL_DIR}/scripts/mousedroid-docker.service"
if [[ -f "$DOCKER_UNIT" ]]; then
    log "installing $(basename "$DOCKER_UNIT") -> ${SYSTEMD_UNIT_DIR}"
    run cp "$DOCKER_UNIT" "${SYSTEMD_UNIT_DIR}/"
    run systemctl daemon-reload
else
    log "WARN: ${DOCKER_UNIT} not found - skipping service install"
fi

# --- 3. Optional trend timer (F-018) -----------------------------------------
if [[ "$WITH_TREND_TIMER" == "1" ]]; then
    for unit in mousedroid-trend.service mousedroid-trend.timer; do
        src="${INSTALL_DIR}/scripts/${unit}"
        if [[ ! -f "$src" ]]; then
            log "ERROR: ${src} not found - cannot install trend timer"
            exit 1
        fi
        log "installing ${unit} -> ${SYSTEMD_UNIT_DIR}"
        run cp "$src" "${SYSTEMD_UNIT_DIR}/"
    done
    run systemctl daemon-reload
    log "enabling mousedroid-trend.timer"
    run systemctl enable --now mousedroid-trend.timer
fi

log "bootstrap complete (dry_run=${DRY_RUN})"
