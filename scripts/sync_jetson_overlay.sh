#!/bin/bash
# =============================================================================
# sync_jetson_overlay.sh — sync Jetson production config overlays.
#
# Copies repo-side yaml overlays into the host-side config directory so the
# Docker container bind-mount always has the latest version without manual
# intervention after a git pull.
#
# Invoked by scripts/mousedroid-docker.service as a non-fatal ExecStartPre
# step (the unit uses a leading dash so a failure here never blocks startup).
# Default mode auto-repairs drift; operators wanting a fail-on-drift gate
# should add a separate ExecStartPre that runs ``sync_jetson_overlay.sh
# --verify`` (without the leading dash) — that path is strict and exits
# non-zero on drift / missing destination.
#
# Two modes:
#   - Default (no args):  copy each repo overlay over its deployed copy.
#   - ``--verify``:        compare each pair, exit non-zero on any drift.
#                          Never modifies state. Used by the operator runbook
#                          after every git pull.
#
# F-013 (smoke-stability sprint, PR #101) added: hash-compare branch that logs
# ``overlay_sync_replaced`` audibly when the deployed copy differs from the
# repo, ``--verify`` for runbook smoke, WARN on missing source instead of
# silent skip.
#
# F-006 remote-LLM sprint generalised the single-pair flow to an N-pair loop
# so operator-opt-in overlays (jetson_production_remote_llm.yaml, future
# jetson_production_hailo.yaml, etc.) sync alongside the canonical
# jetson_production.yaml without per-overlay script edits.
#
# Configuration env vars (every path defaulted; nothing hardcoded):
#   MOUSEDROID_INSTALL_DIR    — repo root on host (default: /opt/mousedroid)
#   MOUSEDROID_OVERLAY_DST    — primary destination file
#                               (default: /etc/mousedroid/jetson_production.yaml)
#   MOUSEDROID_EXTRA_OVERLAYS — space-separated list of additional
#                               ``src:dst`` pairs to sync. Each src/dst is an
#                               absolute path. Empty/unset → script behaviour
#                               is functionally equivalent to the pre-F-006
#                               single-pair flow (same files synced, same exit
#                               codes); the only difference is log lines now
#                               carry a ``pair_index=0`` annotation. Example:
#                                 MOUSEDROID_EXTRA_OVERLAYS=\
#                                   "/opt/mousedroid/config/jetson_production_remote_llm.yaml:\
#                                    /etc/mousedroid/jetson_production_remote_llm.yaml"
#
# Exit codes:
#   0 — every pair synced (or default-mode skip on absent source)
#   1 — any copy failed, any dst unwritable, or (--verify only) any drift
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

# Sync one src→dst pair. Echoes "FAIL" → caller increments fail counter.
# Receives pair_index so structured-log lines can be correlated when
# multiple pairs are processed in one invocation.
sync_pair() {
    local src="$1"
    local dst="$2"
    local pair_index="$3"

    # Source absent → WARN; in verify mode that's a hard fail (operator ran
    # the command, expects a real source). In normal mode it's still a no-op
    # so the systemd unit doesn't fail on a partial repo checkout.
    if [[ ! -f "${src}" ]]; then
        log "WARN overlay_sync_source_missing pair_index=${pair_index} src=${src} (skipping)"
        if [[ "${VERIFY_ONLY}" == "1" ]]; then
            return 1
        fi
        return 0
    fi

    local src_hash
    src_hash="$(_sha256 "${src}")"

    if [[ "${VERIFY_ONLY}" == "1" ]]; then
        if [[ ! -f "${dst}" ]]; then
            log "FAIL overlay_sync_dst_missing pair_index=${pair_index} dst=${dst} src_sha256=${src_hash}"
            return 1
        fi
        local dst_hash
        dst_hash="$(_sha256 "${dst}")"
        if [[ "${src_hash}" == "${dst_hash}" ]]; then
            log "OK overlay_sync_match pair_index=${pair_index} src=${src} dst=${dst} sha256=${src_hash}"
            return 0
        fi
        log "FAIL overlay_sync_drift pair_index=${pair_index} src=${src} src_sha256=${src_hash} dst=${dst} dst_sha256=${dst_hash}"
        return 1
    fi

    local dst_dir
    dst_dir="$(dirname "${dst}")"
    if [[ ! -d "${dst_dir}" ]]; then
        log "Creating destination directory: ${dst_dir}"
        mkdir -p "${dst_dir}"
    fi

    if [[ -f "${dst}" ]]; then
        local dst_hash
        dst_hash="$(_sha256 "${dst}")"
        if [[ "${src_hash}" == "${dst_hash}" ]]; then
            log "OK overlay_sync_match pair_index=${pair_index} src=${src} dst=${dst} sha256=${src_hash}"
            return 0
        fi
        log "overlay_sync_replacing pair_index=${pair_index} src_sha256=${src_hash} dst_sha256=${dst_hash}"
    fi

    # Atomic copy using a temp file in the same directory so rename is on one FS.
    local tmp
    tmp="$(mktemp "${dst_dir}/.overlay_sync.XXXXXX")"
    trap 'rm -f "${tmp}"' RETURN

    # cp + mv are checked explicitly rather than relying on ``set -e``: under
    # ``set -e`` a failing ``mv`` after a successful ``cp`` aborts the whole
    # script mid-loop with no diagnostic, leaving the temp file orphaned (the
    # RETURN trap above only fires on a normal function return, not on a
    # set -e abort) and skipping any remaining overlay pairs. Checking each
    # step keeps the per-pair FAIL line + RETURN cleanup intact so the
    # operator gets a recognizable structured-ish error and the loop can
    # report all pairs.
    if ! cp -f "${src}" "${tmp}"; then
        log "FAIL overlay_sync_copy_failed pair_index=${pair_index} src=${src} dst=${dst}"
        return 1
    fi
    if ! mv -f "${tmp}" "${dst}"; then
        log "FAIL overlay_sync_move_failed pair_index=${pair_index} src=${src} dst=${dst} tmp=${tmp} (dst likely read-only or on a different fs)"
        return 1
    fi
    trap - RETURN  # success → don't try to clean up the moved file
    log "OK overlay_sync_replaced pair_index=${pair_index} src=${src} dst=${dst} sha256=${src_hash}"
    return 0
}

# Build the list of pairs: always start with the primary pair, then append
# any operator-supplied extras. The primary pair's src/dst still come from
# the legacy env vars so deployments that never set MOUSEDROID_EXTRA_OVERLAYS
# see byte-identical behaviour vs the pre-F-006 single-pair script.
PAIRS=("${OVERLAY_SRC}:${OVERLAY_DST}")
if [[ -n "${MOUSEDROID_EXTRA_OVERLAYS:-}" ]]; then
    # Space-separated list. Each token is "src:dst" with absolute paths.
    # shellcheck disable=SC2206 — intentional word-splitting on the env var
    EXTRA_PAIRS=(${MOUSEDROID_EXTRA_OVERLAYS})
    PAIRS+=("${EXTRA_PAIRS[@]}")
fi

OVERALL_RC=0
for i in "${!PAIRS[@]}"; do
    pair="${PAIRS[$i]}"
    src="${pair%%:*}"
    dst="${pair#*:}"
    if [[ -z "${src}" || -z "${dst}" || "${src}" == "${pair}" ]]; then
        log "FAIL overlay_sync_malformed_pair pair_index=${i} value=${pair} (expected src:dst)"
        OVERALL_RC=1
        continue
    fi
    if ! sync_pair "${src}" "${dst}" "${i}"; then
        OVERALL_RC=1
    fi
done

exit "${OVERALL_RC}"
