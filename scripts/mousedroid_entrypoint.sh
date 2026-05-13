#!/bin/sh
# MouseDroid container entrypoint.
#
# Resolves the runtime Settings via ``print_healthcheck_env`` and writes
# the derived env vars to the file the healthcheck script reads. Touches
# the start-grace marker so the healthcheck has an anchor for its grace
# window. Then execs the main process.
#
# Reusable across compose / Dockerfile / systemd. Drives no policy of
# its own — every value comes from the Python config via
# ``print_healthcheck_env`` (single source of truth, no defaults
# duplicated in this script).
set -eu

# Only the env-file location itself is a deployment knob (defaults to
# /run/mousedroid.env). Everything else — heartbeat path, start-grace
# path, thresholds — is sourced from the env file written below.
ENV_FILE="${MOUSEDROID_HEALTHCHECK_ENV_FILE:-/run/mousedroid.env}"
mkdir -p "$(dirname "$ENV_FILE")"

python3 -m mousedroid.tools.print_healthcheck_env "$@" > "$ENV_FILE"

# Source the env file we just wrote to learn the start-grace file path
# (a Settings field, not a hardcoded fallback). Touch the file so the
# healthcheck script's grace logic has an anchor.
# shellcheck disable=SC1090
. "$ENV_FILE"
mkdir -p "$(dirname "$MOUSEDROID_START_GRACE_FILE")"
: > "$MOUSEDROID_START_GRACE_FILE"

# ``exec`` replaces this shell with the Python process so signals
# (SIGTERM from ``docker stop``) are delivered directly to mousedroid.
exec python3 -m mousedroid.main "$@"
