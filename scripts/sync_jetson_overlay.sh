#!/bin/bash
# =============================================================================
# sync_jetson_overlay.sh — sync Jetson production config overlay.
#
# Copies the repo's jetson_production.yaml into the host-side config directory
# so the Docker container bind-mount always has the latest version without
# manual intervention after a git pull.
#
# Invoked by scripts/mousedroid-docker.service as a non-fatal ExecStartPre
# step (the unit uses a leading dash so a failure here never blocks startup).
#
# All paths are driven by environment variables with safe defaults:
#   MOUSEDROID_INSTALL_DIR   — repo root on host (default: /opt/mousedroid)
#   MOUSEDROID_OVERLAY_DST   — destination file  (default: /etc/mousedroid/jetson_production.yaml)
#
# Exit codes:
#   0 — overlay synced (or source was absent, which is a no-op not an error)
#   1 — unexpected error (copy failed, destination dir unwritable, etc.)
# =============================================================================

set -euo pipefail

INSTALL_DIR="${MOUSEDROID_INSTALL_DIR:-/opt/mousedroid}"
OVERLAY_DST="${MOUSEDROID_OVERLAY_DST:-/etc/mousedroid/jetson_production.yaml}"
OVERLAY_SRC="${INSTALL_DIR}/config/jetson_production.yaml"

log() {
    echo "[sync_jetson_overlay] $*" >&2
}

# Source absent → no-op; don't fail service start.
if [[ ! -f "${OVERLAY_SRC}" ]]; then
    log "SOURCE NOT FOUND: ${OVERLAY_SRC} — skipping overlay sync (not a fatal error)"
    exit 0
fi

DST_DIR="$(dirname "${OVERLAY_DST}")"
if [[ ! -d "${DST_DIR}" ]]; then
    log "Creating destination directory: ${DST_DIR}"
    mkdir -p "${DST_DIR}"
fi

# Atomic copy using a temp file in the same directory so rename is on one FS.
OVERLAY_TMP="${OVERLAY_DST}.tmp.$$"
if cp -f "${OVERLAY_SRC}" "${OVERLAY_TMP}"; then
    mv -f "${OVERLAY_TMP}" "${OVERLAY_DST}"
    log "SYNCED: ${OVERLAY_SRC} -> ${OVERLAY_DST}"
else
    rm -f "${OVERLAY_TMP}" 2>/dev/null || true
    log "ERROR: failed to copy ${OVERLAY_SRC} to ${OVERLAY_DST}"
    exit 1
fi
