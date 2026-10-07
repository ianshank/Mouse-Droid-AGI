#!/usr/bin/env bash
# =============================================================================
# MouseDroid — validate env values that compose interpolates into STRUCTURE
# =============================================================================
# docker-compose.jetson.yml interpolates some env values into structural
# positions, not just into the container's environment. A bad value there is
# not a wrong setting: it changes what compose builds.
#
#   MOUSEDROID_JETSON__TENSORRT_CACHE_DIR is the container-side TARGET of the
#   named volume `mousedroid_tensorrt_cache`:
#       - mousedroid_tensorrt_cache:${MOUSEDROID_JETSON__TENSORRT_CACHE_DIR:-...}
#   A value containing `:` injects a third mount field (compose short syntax is
#   VOLUME:TARGET[:MODE]), and a value such as `/etc` mounts the volume over an
#   image directory and shadows it. The cache holds torch2trt engines that are
#   unpickled at load time, so where it lands is not a cosmetic choice.
#
# This runs BEFORE compose parses the file, from both of the paths that start
# compose: scripts/docker_deploy.sh, and scripts/preflight_check.sh, which the
# mousedroid-docker systemd unit runs as a fatal ExecStartPre. One rule, two
# callers, so the boot path cannot drift from the deploy path.
#
# The value is checked only when set. Unset, compose uses its own default,
# which already satisfies every rule below.
#
# Exit codes:
#   0 - every checked value is safe (or unset)
#   1 - at least one value was refused; each refusal names the key and reason
#
# Usage:
#   bash scripts/validate_compose_env.sh
# =============================================================================
set -euo pipefail

# Trees whose every subpath belongs to the OS image. Mounting the cache volume
# anywhere inside one shadows image content. Mixed-ownership trees (/opt, /var,
# /srv, /mnt, /home, /data) are allowed below their top level -- that is where a
# relocated cache legitimately goes -- but never AS their top level.
_OS_OWNED_TREES=(bin boot dev etc lib lib32 lib64 libx32 proc root run sbin sys usr)

_refuse() {
    printf '[ERROR] %s=%s refused: %s\n' "$1" "$2" "$3" >&2
}

# Validate one container-side mount target. Prints a reason and returns 1 on
# the first rule it breaks.
require_mount_target() {
    local name="$1" value="$2"

    if [[ "${value}" != /* ]]; then
        _refuse "${name}" "${value}" "must be an absolute path"
        return 1
    fi
    if [ "${value}" = "/" ]; then
        _refuse "${name}" "${value}" "must not be the root or a top-level directory (it would shadow an image tree)"
        return 1
    fi
    # The charset excludes `:` (mount-field injection), whitespace, quotes, `$`
    # and every other character compose or a shell would treat specially.
    if [[ ! "${value}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
        _refuse "${name}" "${value}" "only [A-Za-z0-9._/-] is allowed (no ':', spaces or quotes)"
        return 1
    fi
    if [[ "${value}" == */ ]]; then
        _refuse "${name}" "${value}" "must not end in '/'"
        return 1
    fi

    local -a parts=()
    local part
    IFS=/ read -r -a parts <<< "${value#/}"
    for part in "${parts[@]}"; do
        case "${part}" in
            '' | . | ..)
                _refuse "${name}" "${value}" "must not contain empty, '.' or '..' components"
                return 1
                ;;
        esac
    done
    if (( ${#parts[@]} < 2 )); then
        _refuse "${name}" "${value}" "must not be the root or a top-level directory (it would shadow an image tree)"
        return 1
    fi
    for part in "${_OS_OWNED_TREES[@]}"; do
        if [ "${parts[0]}" = "${part}" ]; then
            _refuse "${name}" "${value}" "must not be inside /${part}, which belongs to the OS image"
            return 1
        fi
    done
    return 0
}

main() {
    local status=0
    if [ -n "${MOUSEDROID_JETSON__TENSORRT_CACHE_DIR:-}" ]; then
        require_mount_target MOUSEDROID_JETSON__TENSORRT_CACHE_DIR \
            "${MOUSEDROID_JETSON__TENSORRT_CACHE_DIR}" || status=1
    fi
    return "${status}"
}

main "$@"
