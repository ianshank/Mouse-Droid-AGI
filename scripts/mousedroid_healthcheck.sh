#!/bin/sh
# MouseDroid Docker HEALTHCHECK CMD.
#
# Reads heartbeat path + staleness threshold from env vars written by
# the entrypoint (``mousedroid_entrypoint.sh``). Falls back to safe
# defaults when the env file is absent so old images without the
# entrypoint hook still produce a usable signal.
#
# Exit codes follow Docker convention:
#   0 — healthy
#   1 — unhealthy
set -u

ENV_FILE="${MOUSEDROID_HEALTHCHECK_ENV_FILE:-/run/mousedroid.env}"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$ENV_FILE"
fi

HEARTBEAT_PATH="${MOUSEDROID_HEARTBEAT_PATH:-/tmp/mousedroid_heartbeat}"
HEARTBEAT_STALE_S="${MOUSEDROID_HEARTBEAT_STALE_S:-30}"
START_GRACE_S="${MOUSEDROID_START_GRACE_S:-60}"
START_GRACE_FILE="${MOUSEDROID_START_GRACE_FILE:-/run/mousedroid.start}"

now=$(date +%s)

# Grace window: if we're still within START_GRACE_S of process start
# AND the heartbeat file doesn't exist yet, return healthy. The
# orchestrator may not have produced its first heartbeat yet.
if [ ! -e "$HEARTBEAT_PATH" ] && [ -e "$START_GRACE_FILE" ]; then
    start_age=$(( now - $(stat -c %Y "$START_GRACE_FILE") ))
    # ``awk`` for float-tolerant comparison — POSIX shell arithmetic is
    # integer-only and START_GRACE_S may be fractional.
    in_grace=$(awk -v a="$start_age" -v g="$START_GRACE_S" \
        'BEGIN { print (a < g) ? 1 : 0 }')
    [ "$in_grace" = "1" ] && exit 0
fi

# Heartbeat file missing past grace window → unhealthy.
[ -e "$HEARTBEAT_PATH" ] || exit 1

age=$(( now - $(stat -c %Y "$HEARTBEAT_PATH") ))
fresh=$(awk -v a="$age" -v t="$HEARTBEAT_STALE_S" \
    'BEGIN { print (a < t) ? 1 : 0 }')
[ "$fresh" = "1" ] && exit 0
exit 1
