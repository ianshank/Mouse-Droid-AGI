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
# A refusal names the key and the rule, never the value: on the boot path the
# value comes from systemd's EnvironmentFile parser, which joins the next line
# onto a value whose quote is left open, and in docker.env.example the next
# lines down hold ANTHROPIC_API_KEY and MOUSEDROID_TELEMETRY_TOKEN. Stderr from
# an ExecStartPre goes to the journal.
#
# Not covered, and said so rather than implied: the other values compose
# splices into mount and device entries (GCP_CREDENTIALS_FILE as a bind-mount
# SOURCE, and the four MOUSEDROID_*_DEV device paths), and values compose reads
# from a project `.env` file or COMPOSE_ENV_FILES rather than the environment.
#
# Exit codes:
#   0 - every checked value is safe (or unset)
#   1 - at least one value was refused; each refusal names the key and reason
#
# Usage:
#   bash scripts/validate_compose_env.sh
# =============================================================================
set -euo pipefail

# Byte semantics for every length and every bracket range below: [A-Za-z] is
# then exactly those 52 bytes, not whatever the caller's locale collates
# between A and z, and ${#part} counts bytes, which is what NAME_MAX limits.
export LC_ALL=C

# Trees whose every subpath belongs to the OS image. Mounting the cache volume
# anywhere inside one shadows image content. Mixed-ownership trees (/opt, /var,
# /srv, /mnt, /home, /data) are allowed below their top level -- that is where a
# relocated cache legitimately goes -- but never AS their top level.
_OS_OWNED_TREES=(bin boot dev etc lib lib32 lib64 libx32 proc root run sbin sys usr)

# Symlinks INTO an OS-owned tree: /var/run -> /run and /var/lock -> /run/lock.
# The runtime resolves the target inside the container, so these are /run.
_OS_TREE_ALIASES=(var/run var/lock)

# Container-side targets of the service's other volumes. The cache must not
# cover one (be it or an ancestor of it) or nest inside one -- except the source
# bind mount, inside which the default cache is nested by design (see the
# compose file). Pinned to docker-compose.jetson.yml by
# tests/unit/scripts/test_validate_compose_env.py.
_COMPOSE_MOUNT_TARGETS=(
    /opt/mousedroid
    /etc/mousedroid
    /tmp/argus_socket
    /dev/serial
    /home/jetson/mousedroid_experience
    /sys/devices/virtual/thermal
    /sys/devices/platform
    /var/lib/promtail
    /etc/gcp/credentials.json
)
_SOURCE_BIND_MOUNT=/opt/mousedroid

# Inside that bind mount, what the container runs from: the editable install's
# package root, and the default ONNX weights cache (which, per the compose
# file, a volume would silently orphan). Neither may be covered or nested into.
_SOURCE_PATHS_IN_USE=(/opt/mousedroid/src /opt/mousedroid/weights)

# Linux NAME_MAX and PATH_MAX (less the terminating NUL), in bytes.
_NAME_MAX=255
_PATH_MAX=4095

_refuse() {
    printf '[ERROR] %s refused: %s\n' "$1" "$2" >&2
}

# Succeed if path $1 is $2 or lies below it. Component-wise, because both are
# normalised by the time this runs (no '//', '.', '..' or trailing '/').
_is_within() {
    [ "$1" = "$2" ] || [[ "$1" == "$2"/* ]]
}

# Validate one container-side mount target. Prints a reason and returns 1 on
# the first rule it breaks.
require_mount_target() {
    local name="$1" value="$2"

    if [[ "${value}" != /* ]]; then
        _refuse "${name}" "must be an absolute path"
        return 1
    fi
    if [ "${value}" = "/" ]; then
        _refuse "${name}" "must not be the root or a top-level directory (it would shadow an image tree)"
        return 1
    fi
    # The charset excludes `:` (mount-field injection), whitespace, quotes, `$`
    # and every other character compose or a shell would treat specially.
    if [[ ! "${value}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
        _refuse "${name}" "only [A-Za-z0-9._/-] is allowed (no ':', spaces or quotes)"
        return 1
    fi
    if (( ${#value} > _PATH_MAX )); then
        _refuse "${name}" "must not be longer than ${_PATH_MAX} bytes"
        return 1
    fi
    # One trailing '/' names the same directory, and refusing it would stop a
    # rover from booting on a value that worked before this check existed.
    value="${value%/}"
    if [[ "${value}" == *//* || "${value}" == */ ]]; then
        _refuse "${name}" "must not contain empty, '.' or '..' components"
        return 1
    fi

    local -a parts=()
    local part target
    IFS=/ read -r -a parts <<< "${value#/}"
    for part in "${parts[@]}"; do
        case "${part}" in
            '' | . | ..)
                _refuse "${name}" "must not contain empty, '.' or '..' components"
                return 1
                ;;
        esac
        if (( ${#part} > _NAME_MAX )); then
            _refuse "${name}" "must not have a component longer than ${_NAME_MAX} bytes"
            return 1
        fi
    done
    if (( ${#parts[@]} < 2 )); then
        _refuse "${name}" "must not be the root or a top-level directory (it would shadow an image tree)"
        return 1
    fi
    for part in "${_OS_OWNED_TREES[@]}"; do
        if [ "${parts[0]}" = "${part}" ]; then
            _refuse "${name}" "must not be inside /${part}, which belongs to the OS image"
            return 1
        fi
    done
    for part in "${_OS_TREE_ALIASES[@]}"; do
        if _is_within "${value}" "/${part}"; then
            _refuse "${name}" "must not be inside /${part}, a link into an OS-image tree"
            return 1
        fi
    done
    for target in "${_COMPOSE_MOUNT_TARGETS[@]}"; do
        if _is_within "${target}" "${value}"; then
            _refuse "${name}" "would cover ${target}, which compose also mounts"
            return 1
        fi
        if [ "${target}" != "${_SOURCE_BIND_MOUNT}" ] && _is_within "${value}" "${target}"; then
            _refuse "${name}" "must not be inside ${target}, which compose also mounts"
            return 1
        fi
    done
    for target in "${_SOURCE_PATHS_IN_USE[@]}"; do
        if _is_within "${target}" "${value}" || _is_within "${value}" "${target}"; then
            _refuse "${name}" "would shadow ${target}, which the container runs from"
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
