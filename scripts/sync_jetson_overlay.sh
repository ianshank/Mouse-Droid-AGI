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
# Operators can flip ``Environment=MOUSEDROID_OVERLAY_STRICT=1`` to make
# the unit fatal on overlay drift.
#
# Two modes:
#   - Default (no args):  copy repo overlay over the deployed copy.
#   - ``--verify``:        compare repo overlay to the deployed copy and exit
#                          non-zero on drift. Used by the operator runbook
#                          after every git pull. Never modifies state.
#
# F-013 (smoke-stability sprint) added: hash-compare branch that logs
# ``overlay_sync_replaced`` audibly when the deployed copy differs from the
# repo, ``--verify`` for runbook smoke, WARN on missing source instead of
# silent skip.
#
# All paths are driven by environment variables with safe defaults:
#   MOUSEDROID_INSTALL_DIR   — repo root on host (default: /opt/mousedroid)
#   MOUSEDROID_OVERLAY_DST   — destination file  (default: /etc/mousedroid/jetson_production.yaml)
#
# Exit codes:
#   0 — overlay synced (or source was absent in non-verify mode)
#   1 — copy failed, destination dir unwritable, or (--verify only) overlay drift
# =============================================================================

set -euo pipefail

INSTALL_DIR="${MOUSEDROID_INSTALL_DIR:-/opt/mousedroid}"
OVERLAY_DST="${MOUSEDROID_OVERLAY_DST:-/etc/mousedroid/jetson_production.yaml}"
OVERLAY_SRC="${INSTALL_DIR}/config/jetson_production.yaml"

VERIFY_ONLY=0
if [[ "${1:-}" == "--verify" ]]; then
    VERIFY_ONLY=1
fi

log() {
    echo "[sync_jetson_overlay] $*" >&2
}

_sha256() {
    # Portable sha256 — sha256sum on Linux, shasum -a 256 fallback for older hosts.
    local f="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$f" | awk '{print $1}'
    else
        shasum -a 256 "$f" | awk '{print $1}'
    fi
}

# Source absent → WARN; in verify mode that's a hard fail (operator ran the
# command, expects a real source). In normal mode it's still a no-op (matches
# the pre-F-013 behaviour) so the systemd unit doesn't fail on a partial
# repo checkout.
if [[ ! -f "${OVERLAY_SRC}" ]]; then
    log "WARN overlay_sync_source_missing src=${OVERLAY_SRC} (skipping)"
    if [[ "${VERIFY_ONLY}" == "1" ]]; then
        exit 1
    fi
    exit 0
fi

SRC_HASH="$(_sha256 "${OVERLAY_SRC}")"

# --verify path: never mutate state.
if [[ "${VERIFY_ONLY}" == "1" ]]; then
    if [[ ! -f "${OVERLAY_DST}" ]]; then
        log "FAIL overlay_sync_dst_missing dst=${OVERLAY_DST} src_sha256=${SRC_HASH}"
        exit 1
    fi
    DST_HASH="$(_sha256 "${OVERLAY_DST}")"
    if [[ "${SRC_HASH}" == "${DST_HASH}" ]]; then
        log "OK overlay_sync_match src=${OVERLAY_SRC} dst=${OVERLAY_DST} sha256=${SRC_HASH}"
        exit 0
    fi
    log "FAIL overlay_sync_drift src=${OVERLAY_SRC} src_sha256=${SRC_HASH} dst=${OVERLAY_DST} dst_sha256=${DST_HASH}"
    exit 1
fi

DST_DIR="$(dirname "${OVERLAY_DST}")"
if [[ ! -d "${DST_DIR}" ]]; then
    log "Creating destination directory: ${DST_DIR}"
    mkdir -p "${DST_DIR}"
fi

# If destination exists and matches source, skip the copy + log loudly so
# operators can confirm the overlay is current without trawling the systemd
# journal for "synced" lines.
if [[ -f "${OVERLAY_DST}" ]]; then
    DST_HASH="$(_sha256 "${OVERLAY_DST}")"
    if [[ "${SRC_HASH}" == "${DST_HASH}" ]]; then
        log "OK overlay_sync_match src=${OVERLAY_SRC} dst=${OVERLAY_DST} sha256=${SRC_HASH}"
        exit 0
    fi
    log "overlay_sync_replacing src_sha256=${SRC_HASH} dst_sha256=${DST_HASH}"
fi

# Atomic copy using a temp file in the same directory so rename is on one FS.
OVERLAY_TMP="$(mktemp "${DST_DIR}/.overlay_sync.XXXXXX")"
trap 'rm -f "${OVERLAY_TMP}"' EXIT INT TERM

if cp -f "${OVERLAY_SRC}" "${OVERLAY_TMP}"; then
    mv -f "${OVERLAY_TMP}" "${OVERLAY_DST}"
    log "OK overlay_sync_replaced src=${OVERLAY_SRC} dst=${OVERLAY_DST} sha256=${SRC_HASH}"
else
    log "FAIL overlay_sync_copy_failed src=${OVERLAY_SRC} dst=${OVERLAY_DST}"
    exit 1
fi
