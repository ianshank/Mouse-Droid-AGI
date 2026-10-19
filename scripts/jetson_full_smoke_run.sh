#!/bin/bash
# Full Jetson smoke run wrapper (container-first).
#
# All Python execution is delegated to `docker exec mousedroid python3` via a
# transient wrapper script exported as MOUSEDROID_SMOKE_PYTHON, so
# scripts/jetson_smoke_test.sh runs unmodified but executes inside the
# container which already has `mousedroid`, pytest, torch, tensorrt and the
# device passthroughs. Stage blocking defaults follow the Phase 1 roadmap
# (LiDAR -> Camera -> OLED -> Motors); per-stage overrides are exposed via
# MOUSEDROID_SMOKE_BLOCKING_<STAGE> env vars (yes|no).
#
# Usage (run on the Jetson host, repo at /opt/mousedroid):
#   bash scripts/jetson_full_smoke_run.sh
#
# Optional overrides:
#   MOUSEDROID_SMOKE_REPORT_ROOT  -- defaults to <repo>/reports/jetson_smoke
#   MOUSEDROID_SMOKE_BUS          -- I2C bus for OLED stage (default 7)
#   MOUSEDROID_SMOKE_CONTAINER    -- container name (default mousedroid)
#   MOUSEDROID_JETSON_CONFIGS     -- forwarded to jetson_smoke_test.sh + container
#   MOUSEDROID_SMOKE_ESP32_MODE   -- warm (default) | cold; see "ESP32 stages" below
#   MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION
#                                 -- the Pydantic motion gate, forwarded by name (see "Container
#                                    python wrapper"): exported here it reaches the stages; unset
#                                    here, a warm stage runs without it (docker exec removes it),
#                                    so the gate falls back to the YAML default, while a cold
#                                    stage keeps the service's own value (compose run, docker.env)
#   MOUSEDROID_SMOKE_HEALTH_RETRIES, MOUSEDROID_SMOKE_HEALTH_INTERVAL_S
#                                 -- the bounded wait (30 x 5 s) for the restarted container's
#                                    healthcheck before llm_probe; cold mode only
#
# ESP32 stages (serial, motor, power, mcp_motor_smoke) and the port. SerialESP32Driver and the
# serial stage open the ESP32 UART exclusively. That lock holds between openers INSIDE one
# container; the host and other containers get their own device nodes and never see it, so what
# keeps a smoke stage off the orchestrator's port is stopping the orchestrator.
# MOUSEDROID_SMOKE_ESP32_MODE says how the stages get the port:
#   warm  run them through `docker exec` in the running container. Right while the container's
#         own esp32 is the mock (the shipped jetson_production.yaml has esp32.enabled: false),
#         because its orchestrator then holds no port. When it does own the port (esp32.enabled,
#         not mock_hardware, serial protocol) every ESP32 stage is REFUSED: recorded as FAIL with
#         "warm smoke refused" (EXPECTED-FAIL when demoted to non-blocking) and not run. So is a
#         stage when the wrapper cannot tell who owns the port.
#   cold  stop the container, run the ESP32 stages in a transient container from the same
#         compose service (`docker compose run`: same image and device passthrough, the
#         orchestrator not started), then restart the container. A dry run first proves that
#         container can be created (compose run aborts on a docker.env the calling user cannot
#         read); if it cannot, the stages are refused with compose's message and nothing is
#         stopped. The restore is `docker restart`, never `docker start` (a start while a stop
#         is still in flight does nothing), and an EXIT trap repeats it on every exit the shell
#         can catch. SIGKILL cannot be trapped: a `kill -9` of this script inside a cold window
#         leaves the container stopped (`docker start` it). If `docker stop` fails the stage is
#         not run. mcp_motor_smoke opens no window of its own while nothing owns a port: it
#         measures the mock then, and its row says so.
# An ESP32 stage whose log shows only skips (pyserial missing, bench overlay opted out) records
# SKIPPED, never PASS.

# ``-e`` (errexit) catches harness-logic bugs early (a failed cd / mkdir /
# heredoc), aligning with scripts/jetson_smoke_test.sh. STAGE failures are NOT
# fatal — they are captured explicitly via the rc/OVERALL_FAIL accumulator in
# run_stage (which brackets each stage command in `set +e ... set -e`), and the
# few blocking run_stage call sites are guarded with `|| true`/`|| break` so a
# stage failure still records + writes SUMMARY.md rather than aborting the run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_DIR}"

CONTAINER="${MOUSEDROID_SMOKE_CONTAINER:-mousedroid}"
OLED_BUS="${MOUSEDROID_SMOKE_BUS:-7}"
REPORT_ROOT="${MOUSEDROID_SMOKE_REPORT_ROOT:-${REPO_DIR}/reports/jetson_smoke}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${REPORT_ROOT}/${STAMP}"
mkdir -p "${RUN_DIR}"

SUMMARY="${RUN_DIR}/SUMMARY.md"
declare -a RESULTS=()
OVERALL_FAIL=0
STAGE_NOTE=""  # words added to the row of the stage being run (run_esp32_stage sets and clears it)

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }

# Warm or cold for the ESP32 stages (see the header). An unknown value is fatal rather than
# quietly warm: the operator asked for something this run cannot honour.
ESP32_MODE="${MOUSEDROID_SMOKE_ESP32_MODE:-warm}"
case "${ESP32_MODE}" in
    warm|cold) ;;
    *)
        log "FATAL: MOUSEDROID_SMOKE_ESP32_MODE='${ESP32_MODE}' is invalid; expected warm or cold."
        exit 2
        ;;
esac

record() {
    local name="$1" status="$2" note="${3:-}"
    RESULTS+=("${status}|${name}|${note}")
    log "${status}: ${name} ${note}"
}

resolve_blocking() {
    # Resolve effective blocking flag for a stage. Operators can promote or
    # demote any stage without code edits via:
    #   MOUSEDROID_SMOKE_BLOCKING_<STAGE_UPPER>=yes|no
    # Falls back to the default literal passed in by the caller. Values are
    # validated; unknown values fall back to the default.
    local label="$1" default_blocking="$2"
    local upper
    upper="$(printf '%s' "${label}" | tr '[:lower:]-' '[:upper:]_')"
    local override_var="MOUSEDROID_SMOKE_BLOCKING_${upper}"
    local override="${!override_var:-}"
    case "${override}" in
        yes|no) printf '%s' "${override}" ;;
        "")     printf '%s' "${default_blocking}" ;;
        *)
            log "WARN: ${override_var}='${override}' is invalid; using default '${default_blocking}'"
            printf '%s' "${default_blocking}"
            ;;
    esac
}

# The ESP32 stages (see run_esp32_stage), which run_stage must hold to a measurement.
ESP32_STAGE_LABELS=" serial motor power mcp_motor_smoke "

# The reason a stage script gave for skipping, when its log shows skips and not one pass. The stage
# script exits 0 when everything skipped (pyserial missing, bench overlay opted out), and a clean
# exit is not a measurement. Prints the reason line after the first SKIP line; returns 1 otherwise.
skipped_reason() {
    local logfile="$1" reason
    [[ -f "${logfile}" ]] || return 1
    if grep -q '^  PASS: ' "${logfile}" || ! grep -q '^  SKIP: ' "${logfile}"; then
        return 1
    fi
    reason="$(grep -A1 -m1 '^  SKIP: ' "${logfile}" | sed -n '2s/^[[:space:]]*//p' | tr -d '\r')"
    reason="${reason//|//}"
    printf '%s' "${reason:-stage skipped}"
}

# A clean exit is a PASS, unless this is an ESP32 stage that only skipped: that records SKIPPED
# with the reason, never PASS. STAGE_NOTE (set by run_esp32_stage) rides on either row.
record_clean_exit() {
    local label="$1" logfile="$2" reason
    if [[ "${ESP32_STAGE_LABELS}" == *" ${label} "* ]] && reason="$(skipped_reason "${logfile}")"; then
        record "${label}" "SKIPPED" "${reason}${STAGE_NOTE:+ -- ${STAGE_NOTE}}"
        return 0
    fi
    record "${label}" "PASS" "${STAGE_NOTE}"
}

run_stage() {
    # $1 = stage label, $2 = blocking? (yes|no), $3 = timeout seconds (0 = none),
    # $4.. = command
    local label="$1" default_blocking="$2" tmo="$3"; shift 3
    local blocking
    blocking="$(resolve_blocking "${label}" "${default_blocking}")"
    local logfile="${RUN_DIR}/${label}.log"
    log "=== STAGE ${label} (blocking=${blocking} timeout=${tmo}s) ==="
    log "    cmd: $*"
    log "    log: ${logfile}"
    set +e
    if [[ "${tmo}" != "0" ]]; then
        MOUSEDROID_SMOKE_STAGE_TIMEOUT="${tmo}" \
            timeout --signal=INT --kill-after=10 "${tmo}" "$@" >"${logfile}" 2>&1
    else
        MOUSEDROID_SMOKE_STAGE_TIMEOUT="0" "$@" >"${logfile}" 2>&1
    fi
    local rc=$?
    # Re-enable errexit (disabled above only for the stage command itself) so
    # any harness-logic bug below still fails fast. Under the old `set -uo`
    # regime this line was a redundant `set +e`; with `set -euo` it must
    # restore errexit.
    set -e
    if [[ ${rc} -eq 0 ]]; then
        record_clean_exit "${label}" "${logfile}"
        return 0
    fi
    if [[ "${blocking}" == "no" ]]; then
        local why="rc=${rc} (non-blocking)"
        [[ ${rc} -eq 124 || ${rc} -eq 137 ]] && why="rc=${rc} (timeout after ${tmo}s, non-blocking)"
        why+="${STAGE_NOTE:+ -- ${STAGE_NOTE}}"
        record "${label}" "EXPECTED-FAIL" "${why}"
        return 0
    fi
    record "${label}" "FAIL" "rc=${rc}${STAGE_NOTE:+ -- ${STAGE_NOTE}}"
    OVERALL_FAIL=1
    return 1
}

container_running() {
    docker ps --filter "name=^/${CONTAINER}$" --format '{{.Names}}' | grep -qx "${CONTAINER}"
}

# --- Container python wrapper --------------------------------------------
# jetson_smoke_test.sh resolves "${PYTHON}" from MOUSEDROID_SMOKE_PYTHON
# (must be executable). We point it at this shim that proxies into the
# running container with relevant MOUSEDROID_* env vars forwarded. `-e NAME` (no value) hands
# `docker exec` this process's NAME when NAME is set here, and REMOVES NAME from the exec'd
# environment when it is not (the container's own value included); `docker compose run -e NAME`
# passes nothing in that case, so the service's own value (docker.env) stands. Either way an unset
# variable never arrives as an empty string, which is what lets
# MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION open the Pydantic motion gate from the host.
PY_WRAPPER="${RUN_DIR}/python3-in-container"
cat > "${PY_WRAPPER}" <<EOF
#!/bin/bash
set -uo pipefail

stage_timeout="\${MOUSEDROID_SMOKE_STAGE_TIMEOUT:-0}"
inner_timeout="\${stage_timeout}"
if [[ "\${stage_timeout}" =~ ^[0-9]+$ && "\${stage_timeout}" -gt 5 ]]; then
    inner_timeout="\$((stage_timeout - 5))"
fi

docker_args=(
    docker exec
    -w ${REPO_DIR}
    -e MOUSEDROID_MOCK_HARDWARE=false
    -e MOUSEDROID_JETSON_CONFIGS
    -e MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION
    -e MOUSEDROID_FACE_DISPLAY_SMOKE
    -e MOUSEDROID_FACE_DISPLAY_BUS
    -e MOUSEDROID_LLM__N_GPU_LAYERS
    -e MOUSEDROID_LLM__MAX_TOKENS
    -e PYTHONPATH
    ${CONTAINER}
)

if [[ "\${inner_timeout}" != "0" ]]; then
    docker_args+=(timeout --signal=INT --kill-after=5 "\${inner_timeout}")
fi

docker_args+=(python3 "\$@")
exec "\${docker_args[@]}"
EOF
chmod +x "${PY_WRAPPER}"
export MOUSEDROID_SMOKE_PYTHON="${PY_WRAPPER}"

# --- ESP32 stages: warm or cold ------------------------------------------
# See the header. In cold mode the ESP32 stages use a second wrapper that runs python in a
# transient container (the compose service `mousedroid`, its entrypoint replaced so the
# orchestrator is not started) instead of `docker exec`-ing into the running one. --name
# overrides the service's container_name, so a leftover can be found and removed by name. Like
# the warm wrapper it runs python under an inner `timeout` (here as the entrypoint) sized from the
# stage budget: were only the host-side `timeout` to fire, its --kill-after would kill the docker
# CLI and leave this container running, and the next stage's --name would collide with it.
COLD_CONTAINER="${CONTAINER}-cold-smoke"
CONTAINER_STOPPED_FOR_COLD=0
CONTAINER_RESTARTED_BY_COLD=0
COLD_RESTART_FAILED=0
COLD_DRY_RUN_STATE=""
COLD_REFUSAL_REASON=""
ESP32_OWNER_STATE=""
HEALTH_RETRIES="${MOUSEDROID_SMOKE_HEALTH_RETRIES:-30}"
HEALTH_INTERVAL_S="${MOUSEDROID_SMOKE_HEALTH_INTERVAL_S:-5}"

if [[ "${ESP32_MODE}" == "cold" ]]; then
    COLD_PY_WRAPPER="${RUN_DIR}/python3-in-cold-container"
    cat > "${COLD_PY_WRAPPER}" <<EOF
#!/bin/bash
set -uo pipefail

stage_timeout="\${MOUSEDROID_SMOKE_STAGE_TIMEOUT:-0}"
inner_timeout="\${stage_timeout}"
if [[ "\${stage_timeout}" =~ ^[0-9]+$ && "\${stage_timeout}" -gt 5 ]]; then
    inner_timeout="\$((stage_timeout - 5))"
fi

docker_args=(
    docker compose --env-file /dev/null -f "${REPO_DIR}/docker-compose.jetson.yml"
    run -T --rm --no-deps --name ${COLD_CONTAINER}
    -e MOUSEDROID_MOCK_HARDWARE=false
    -e MOUSEDROID_JETSON_CONFIGS
    -e MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION
    -w "${REPO_DIR}"
)

if [[ "\${inner_timeout}" != "0" ]]; then
    docker_args+=(--entrypoint timeout mousedroid --signal=INT --kill-after=5 "\${inner_timeout}" python3)
else
    docker_args+=(--entrypoint python3 mousedroid)
fi

exec "\${docker_args[@]}" "\$@"
EOF
    chmod +x "${COLD_PY_WRAPPER}"
fi

# Does the orchestrator in ${CONTAINER} own the ESP32 port? It does when its own settings build a
# SerialESP32Driver: esp32.enabled, not mock_hardware (that builds the mock), and the serial
# protocol (a wifi board holds no port). Loaded the way the stages load them, from the container's
# own environment, which the exec inherits. Returns 0 = it does, 1 = it does not, 2 = could not
# tell (the exec failed or the settings did not load). The probe prints its sentinel only once it
# has decided, so a crash cannot read as "no".
container_esp32_enabled() {
    local out
    out="$(docker exec "${CONTAINER}" python3 -c '
try:
    from mousedroid.validation.runtime import load_runtime_settings

    cfg = load_runtime_settings()
    owns_port = cfg.esp32.enabled and not cfg.mock_hardware and cfg.esp32.protocol == "serial"
    print("esp32_owned=%d" % bool(owns_port))
except Exception:
    raise SystemExit(2)
' 2>/dev/null)" || true
    case "${out}" in
        *esp32_owned=1*) return 0 ;;
        *esp32_owned=0*) return 1 ;;
        *) return 2 ;;
    esac
}

# Probe once per run and keep the answer in ESP32_OWNER_STATE: owned | free | unknown.
probe_esp32_owner() {
    if [[ -n "${ESP32_OWNER_STATE}" ]]; then
        return 0
    fi
    local rc=0
    container_esp32_enabled || rc=$?
    case "${rc}" in
        0) ESP32_OWNER_STATE="owned" ;;
        1) ESP32_OWNER_STATE="free" ;;
        *) ESP32_OWNER_STATE="unknown" ;;
    esac
}

# Record an ESP32 stage that was not run, with the bookkeeping of a failed run_stage: a blocking
# stage is a FAIL and stops the run, a non-blocking one is EXPECTED-FAIL. The reason is also the
# stage log, so SUMMARY.md and the per-stage log say the same thing.
refuse_esp32_stage() {
    local label="$1" default_blocking="$2" why="$3"
    local blocking
    blocking="$(resolve_blocking "${label}" "${default_blocking}")"
    log "=== STAGE ${label} (blocking=${blocking}) NOT RUN ==="
    log "    ${why}"
    printf '%s\n' "${why}" > "${RUN_DIR}/${label}.log"
    if [[ "${blocking}" == "no" ]]; then
        record "${label}" "EXPECTED-FAIL" "${why} (non-blocking)"
        return 0
    fi
    record "${label}" "FAIL" "${why}"
    OVERALL_FAIL=1
    return 1
}

# Put the container back, whatever the daemon is doing with it. `docker restart`, never `docker
# start`: an interrupt can land while `docker stop` is still running (it has a grace period of
# tens of seconds), a start issued then is a no-op, and the container ends exited. A transient
# container a killed CLI left running is removed first; it would still hold the port.
restore_container() {
    docker rm -f "${COLD_CONTAINER}" >/dev/null 2>&1 || true
    docker restart "${CONTAINER}" >/dev/null 2>&1
}

# EXIT trap for cold mode: never leave the rover brain down. It runs on every exit the shell can
# catch (a failed run, a `set -e` abort, SIGINT/SIGTERM/SIGHUP); SIGKILL cannot be trapped.
restart_container_on_exit() {
    local rc=$?
    if [[ "${CONTAINER_STOPPED_FOR_COLD}" == "1" ]]; then
        log "trap: restarting ${CONTAINER} after a cold ESP32 stage (rc=${rc})"
        restore_container \
            || log "trap: docker restart ${CONTAINER} failed -- the rover brain may be down"
        CONTAINER_STOPPED_FOR_COLD=0
    fi
}

# Prove the transient container can be created BEFORE the container is stopped. `compose run`
# aborts (permission denied) on a /etc/mousedroid/docker.env the calling user cannot read, even
# with required: false, and the only safe moment to find that out is while the brain is up. Runs
# `true` in a throwaway container, once per run; the answer is kept. On a refusal COLD_REFUSAL_REASON
# carries compose's own message (one line, no `|`: it lands in a SUMMARY.md row).
cold_dry_run() {
    if [[ "${COLD_DRY_RUN_STATE}" == "ok" ]]; then
        return 0
    elif [[ "${COLD_DRY_RUN_STATE}" == "refused" ]]; then
        return 1
    fi
    local err="" rc=0
    err="$(docker compose --env-file /dev/null -f "${REPO_DIR}/docker-compose.jetson.yml" \
        run -T --rm --no-deps --name "${COLD_CONTAINER}-dry" --entrypoint true mousedroid \
        2>&1 >/dev/null)" || rc=$?
    docker rm -f "${COLD_CONTAINER}-dry" >/dev/null 2>&1 || true
    if [[ ${rc} -eq 0 ]]; then
        COLD_DRY_RUN_STATE="ok"
        return 0
    fi
    COLD_DRY_RUN_STATE="refused"
    err="${err//$'\n'/ }"
    err="${err//|//}"
    if [[ -z "${err}" ]]; then
        err="rc=${rc}"
    fi
    COLD_REFUSAL_REASON="cold smoke refused: the transient container cannot be created (${err:0:300}); ${CONTAINER} was left running"
    return 1
}

# Begin a cold window: prove the transient container can be created (cold_dry_run), then stop the
# container so the ESP32 stages can open the port. Idempotent until esp32_cold_end. The trap is
# armed BEFORE the stop, so an abort between the two cannot leave the container down. A refusal
# leaves the container running and sets COLD_REFUSAL_REASON; if the stop fails the container may
# still own the port, so the stage must not run on a guess: put it back and tell the caller.
esp32_cold_begin() {
    if [[ "${CONTAINER_STOPPED_FOR_COLD}" == "1" ]]; then
        return 0
    fi
    if ! cold_dry_run; then
        log "cold smoke: ${COLD_REFUSAL_REASON}"
        return 1
    fi
    trap restart_container_on_exit EXIT
    CONTAINER_STOPPED_FOR_COLD=1
    log "cold smoke: stopping ${CONTAINER} so the ESP32 stages can open the port"
    if docker stop "${CONTAINER}" >/dev/null 2>&1; then
        return 0
    fi
    COLD_REFUSAL_REASON="cold smoke aborted: docker stop ${CONTAINER} failed, so it may still own the ESP32 port"
    log "cold smoke: docker stop ${CONTAINER} failed"
    esp32_cold_end
    return 1
}

# End a cold window: put the container back (restore_container), disarm the trap. A no-op unless a
# cold stage stopped the container. A failed restart is a FAIL and leaves the trap armed, which
# retries it at exit.
esp32_cold_end() {
    if [[ "${CONTAINER_STOPPED_FOR_COLD}" != "1" ]]; then
        return 0
    fi
    log "cold smoke: restarting ${CONTAINER}"
    if restore_container; then
        CONTAINER_STOPPED_FOR_COLD=0
        CONTAINER_RESTARTED_BY_COLD=1
        trap - EXIT
        return 0
    fi
    log "cold smoke: docker restart ${CONTAINER} failed -- the rover brain may be down"
    if [[ "${COLD_RESTART_FAILED}" != "1" ]]; then
        COLD_RESTART_FAILED=1
        record "container_restart" "FAIL" "docker restart ${CONTAINER} failed; the EXIT trap retries it"
    fi
    OVERALL_FAIL=1
    return 0
}

# After a cold window the orchestrator is booting, and the blocking llm_probe that follows loads a
# model of its own: wait, bounded, for the container's healthcheck first. The pattern of
# jetson_full_validation.sh::wait_for_health (retries x interval, both env-overridable). It never
# fails the run: past the bound the row says the container was not healthy and the probe runs.
wait_for_restarted_container() {
    if [[ "${CONTAINER_RESTARTED_BY_COLD}" != "1" ]]; then
        return 0
    fi
    CONTAINER_RESTARTED_BY_COLD=0
    local i status=""
    for ((i = 1; i <= HEALTH_RETRIES; i++)); do
        status="$(docker inspect --format '{{.State.Health.Status}}' "${CONTAINER}" 2>/dev/null || true)"
        if [[ "${status}" == "healthy" ]]; then
            record "container_health_wait" "INFO" "healthy after ${i} poll(s) following the cold restart"
            return 0
        fi
        if ((i < HEALTH_RETRIES)); then
            sleep "${HEALTH_INTERVAL_S}"
        fi
    done
    record "container_health_wait" "INFO" "${CONTAINER} not healthy after ${HEALTH_RETRIES} polls x ${HEALTH_INTERVAL_S}s (last status: ${status:-unknown}); llm_probe ran anyway"
}

# Run an ESP32-owned stage (serial, motor, power, mcp_motor_smoke). Same arguments as run_stage:
# $1 = stage label, $2 = blocking? (yes|no), $3 = timeout seconds, $4.. = command. The command
# reaches python through the wrapper this stage runs under: the running container's, or in a cold
# window the transient container's (stage scripts get it through MOUSEDROID_SMOKE_PYTHON, and an
# argument equal to ${PY_WRAPPER} is rewritten to it). Warm mode refuses the stage while the
# orchestrator owns the port (or when that cannot be told); cold mode stops the container first
# and leaves it stopped until esp32_cold_end, so consecutive ESP32 stages share one window.
# mcp_motor_smoke is the exception to both: it stacks no bench overlay, so with nothing owning a
# port it exercises MockESP32Driver and touches no port. It then needs neither a refusal nor a
# window of its own, and its row says it was the mock.
run_esp32_stage() {
    local label="$1" default_blocking="$2" tmo="$3"
    shift 3
    local stage_mode="${ESP32_MODE}" why="" note=""
    if [[ "${label}" == "mcp_motor_smoke" ]]; then
        probe_esp32_owner
        if [[ "${ESP32_OWNER_STATE}" == "free" ]]; then
            stage_mode="warm"
            note="mock driver: ${CONTAINER} has no serial ESP32 enabled (esp32.enabled false, mock_hardware or a wifi board), so the velocity and e-stop checks ran against MockESP32Driver"
        fi
    fi
    if [[ "${stage_mode}" == "cold" ]]; then
        if ! esp32_cold_begin; then
            why="${COLD_REFUSAL_REASON}"
        fi
    else
        probe_esp32_owner
        case "${ESP32_OWNER_STATE}" in
            owned)
                why="warm smoke refused: the orchestrator in ${CONTAINER} owns the ESP32 port; re-run with MOUSEDROID_SMOKE_ESP32_MODE=cold"
                ;;
            unknown)
                why="warm smoke refused: could not tell whether the orchestrator in ${CONTAINER} owns the ESP32 port (the settings probe failed); fix the container or re-run with MOUSEDROID_SMOKE_ESP32_MODE=cold"
                ;;
        esac
    fi
    if [[ -n "${why}" ]]; then
        refuse_esp32_stage "${label}" "${default_blocking}" "${why}" || return 1
        return 0
    fi
    local stage_wrapper="${PY_WRAPPER}" args=() arg rc=0
    if [[ "${stage_mode}" == "cold" ]]; then
        stage_wrapper="${COLD_PY_WRAPPER}"
        # A transient container an earlier stage left behind would collide on --name.
        docker rm -f "${COLD_CONTAINER}" >/dev/null 2>&1 || true
    fi
    for arg in "$@"; do
        if [[ "${arg}" == "${PY_WRAPPER}" ]]; then
            args+=("${stage_wrapper}")
        else
            args+=("${arg}")
        fi
    done
    STAGE_NOTE="${note}"
    run_stage "${label}" "${default_blocking}" "${tmo}" \
        env MOUSEDROID_SMOKE_PYTHON="${stage_wrapper}" "${args[@]}" || rc=$?
    STAGE_NOTE=""
    return "${rc}"
}

if ! container_running; then
    log "FATAL: container ${CONTAINER} is not running. Start it before running smoke."
    exit 2
fi

# --- Stage 0: container_health (informational, non-blocking) -------------
{
    echo "Container: ${CONTAINER}"
    docker inspect --format 'Image: {{.Image}}' "${CONTAINER}"
    docker inspect --format 'Status: {{.State.Status}}' "${CONTAINER}"
    docker inspect --format 'Health.Status: {{.State.Health.Status}}' "${CONTAINER}" 2>/dev/null || true
    docker inspect --format 'Health.FailingStreak: {{.State.Health.FailingStreak}}' "${CONTAINER}" 2>/dev/null || true
    echo "--- last 3 healthcheck log entries ---"
    docker inspect --format '{{range .State.Health.Log}}exit={{.ExitCode}} | out={{.Output}}{{println}}{{end}}' "${CONTAINER}" 2>/dev/null | tail -3 || true
    echo "--- container python3 sanity ---"
    docker exec "${CONTAINER}" python3 -c "import sys, mousedroid, pytest; print('python', sys.version.split()[0]); print('mousedroid', mousedroid.__file__); print('pytest', pytest.__version__)"
} > "${RUN_DIR}/container_health.log" 2>&1 || true  # informational: never abort under set -e
record "container_health" "INFO" "see container_health.log"

# --- Stages 1-8: delegated to jetson_smoke_test.sh via PY_WRAPPER --------
# Bench-debug sensors (camera/audio/lidar/speaker/voice) are non-blocking until the
# physical wiring/overlays land; system/usbc/gpio/serial and the bounded motor smoke are hard gates.
# serial, motor and power own the ESP32 port, so they go through run_esp32_stage (warm or cold).
for stage in system usbc gpio; do
    run_stage "${stage}" "yes" 60 bash scripts/jetson_smoke_test.sh "${stage}" || break
done

if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_esp32_stage "serial" "yes" 60 bash scripts/jetson_smoke_test.sh serial || true
fi

if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    # Motor remains soft until M4 (rover power + encoder wiring + speaker
    # write timeout). Promote with MOUSEDROID_SMOKE_BLOCKING_MOTOR=yes once
    # encoder loopback consistently passes on the bench.
    run_esp32_stage "motor" "no" 120 \
        env \
            MOUSEDROID_SMOKE_ALLOW_MOTION=1 \
            bash scripts/jetson_smoke_test.sh motor || true
fi

if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    # Power chain (battery + zero-vel + e-stop). Blocking by default once
    # USB-C wiring is verified; operators can demote at runtime with
    # MOUSEDROID_SMOKE_BLOCKING_POWER=no.
    run_esp32_stage "power" "yes" 120 bash scripts/jetson_smoke_test.sh power || true
fi

# Cold mode keeps the container stopped across serial -> motor -> power; bring it back before the
# stages that exec into it. A no-op in warm mode, and when no ESP32 stage stopped it.
esp32_cold_end

if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    # M1: lidar is now a hard gate by default (re-blocked after LD19 hardening).
    # Operators can demote at runtime with MOUSEDROID_SMOKE_BLOCKING_LIDAR=no.
    run_stage "lidar" "yes" 60 bash scripts/jetson_smoke_test.sh lidar || true
fi

if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    for stage in camera audio speaker voice; do
        run_stage "${stage}" "no" 45 bash scripts/jetson_smoke_test.sh "${stage}"
    done
fi

# --- Stage 9: OLED (non-blocking, container stays up) ---------------------
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "oled" "no" 60 \
        env \
            MOUSEDROID_FACE_DISPLAY_SMOKE=1 \
            MOUSEDROID_FACE_DISPLAY_BUS="${OLED_BUS}" \
            "${PY_WRAPPER}" -m pytest -m hardware -v \
            tests/hardware/test_ssd1306_smoke.py
fi

# --- Stage 10: app health check -------------------------------------------
# `|| true`: run_stage already records the FAIL + sets OVERALL_FAIL; the guard
# only stops `set -e` aborting before SUMMARY.md is written for a blocking stage.
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "app_health" "yes" 60 bash scripts/jetson_smoke_test.sh app || true
fi

# --- Stage 11: hardware pytest suite (non-blocking until camera/USB speaker/HC-SR04 are fixed)
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "hardware_pytest" "no" 300 \
        bash scripts/jetson_smoke_test.sh pytest
fi

# --- Stage 12: orchestrator E2E 5s (non-blocking until camera/dev/video0 fixed)
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "e2e" "no" 60 bash scripts/jetson_smoke_test.sh e2e
fi

# --- Stage 13a: MCP + motor smoke ----------------------------------------
# Runs the rover motor smoke (velocity round-trip, e-stop latency, MCP
# resource polling under load) with the optional MCP server enabled.
#
# MOTION: the velocity round-trip commands zero velocity unless the Pydantic gate
# ESP32Config.smoke_test_allow_motion is true. That gate, not this script, is what keeps an
# unattended rover (e.g. on a table) still. The stage exports the host's
# MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION (false when the host leaves it unset) to the python
# wrapper, and the wrapper forwards the name into the container (warm: docker exec; cold: the
# transient container), so exporting it on the host reaches the gate. Set it to true ONLY when
# the rover is on rollers / tethered / under direct supervision.
#
# ESP32 stage: it goes through run_esp32_stage like serial/motor/power, so warm mode refuses it
# while the orchestrator owns the port. Unlike motor and power it does not stack the bench
# overlay, so with the shipped esp32.enabled: false its driver is the mock: run_esp32_stage then
# runs it in the running container with no refusal and no cold window (nothing touches a port),
# and its row says it measured the mock. It measures the board only when the container's own
# environment enables it, and then it gets a window of its own in cold mode.
#
# Non-blocking so an MCP/motor failure does not mask later stages but
# still surfaces in SUMMARY.md.
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_esp32_stage "mcp_motor_smoke" "no" 300 \
        env \
            MOUSEDROID_MCP__ENABLED=true \
            MOUSEDROID_MCP__BIND_TRANSPORT=false \
            MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION="${MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION:-false}" \
            "${PY_WRAPPER}" -m pytest tests/hardware/test_motor_smoke.py \
                -v -m hardware --tb=short --no-cov || true  # blocking if promoted; OVERALL_FAIL set in run_stage
fi
esp32_cold_end

# --- Stage 14: LLM live probe ---------------------------------------------
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    wait_for_restarted_container  # a no-op unless a cold window restarted the container
    LLM_PROBE='import asyncio
from mousedroid.config.loader import load_settings
from mousedroid.factory import build_llm_gateway
from mousedroid.validation.runtime import resolve_runtime_config_paths

cfg = load_settings(*resolve_runtime_config_paths())
gw = build_llm_gateway(cfg)

async def main() -> None:
    await gw.start()
    try:
        degraded = getattr(gw, "is_degraded", False)
        if degraded:
            raise RuntimeError("LLM gateway entered degraded mode during startup")
        if not gw.is_ready:
            raise RuntimeError("LLM gateway is not ready after startup")
        print("LLM_PROBE_READY")
    finally:
        await gw.stop()

asyncio.run(main())'
    run_stage "llm_probe" "yes" 120 \
        env \
            MOUSEDROID_LLM__N_GPU_LAYERS=-1 \
            "${PY_WRAPPER}" -c "${LLM_PROBE}" || true  # blocking stage; OVERALL_FAIL set in run_stage
fi

# --- Stage 15: New features probes (Phase B) ------------------------------
# Exercises operator-observable surfaces introduced by PRs #75-#82 that the
# 14-stage smoke does NOT cover (lidar raw WS, sensor liveness, port
# discovery, mDNS readiness, hello negotiation, voice event fairness,
# orchestrator FailureRecorder, ClockProtocol, dashboard data flow, logs WS).
# Blocking by default — a green Phase A with a red Phase B should NOT
# advance the deployment SHA bump.
#
# When Phase A already failed we still record an explicit SKIPPED entry
# so the SUMMARY has a Phase B row instead of looking like "the stage
# never existed" — operators triaging a partial failure need to see the
# stage is intentionally not run (Copilot review on PR #83).
mkdir -p "${RUN_DIR}/new_features"
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "new_features" "yes" 240 \
        env \
            MOUSEDROID_SMOKE_CONTAINER="${CONTAINER}" \
            MOUSEDROID_PROBE_REPORT_DIR="${RUN_DIR}/new_features" \
            bash scripts/jetson_new_features_probe.sh || true  # blocking; OVERALL_FAIL set in run_stage
else
    {
        echo "SKIPPED: Phase B was not run because Phase A failed earlier."
        echo "Re-run scripts/jetson_full_smoke_run.sh after fixing the blocking"
        echo "stage(s) above. Phase B requires a healthy app_health + llm_probe."
    } > "${RUN_DIR}/new_features.log"
    record "new_features" "SKIPPED" "Phase A failed; see preceding stages"
fi

# --- Summary -------------------------------------------------------------

# Enrich notes for the voice stage so the precise reason for any failure is
# visible directly in SUMMARY.md without having to open voice.log.
voice_log="${RUN_DIR}/voice.log"
voice_remediation=""
if [[ -f "${voice_log}" ]]; then
    for i in "${!RESULTS[@]}"; do
        IFS='|' read -r status name note <<<"${RESULTS[$i]}"
        if [[ "${name}" != "voice" || "${status}" == "PASS" ]]; then
            continue
        fi
        hints=()
        if grep -q "voice.tts_model_path is not configured" "${voice_log}"; then
            hints+=("voice.tts_model_path missing in runtime config")
        fi
        if grep -q "piper_tts_not_installed" "${voice_log}"; then
            hints+=("piper-tts python package not installed in container")
        fi
        if grep -q "piper_tts_no_model_path" "${voice_log}"; then
            hints+=("piper TTS started without a model path")
        fi
        if grep -q "piper_tts_load_failed" "${voice_log}"; then
            hints+=("piper voice model failed to load (see voice.log)")
        fi
        if grep -qE "Piper voice model failed to load from" "${voice_log}"; then
            hints+=("Piper model file unreadable at configured tts_model_path")
        fi
        if grep -q "Rocky voice TTS returned silent audio" "${voice_log}"; then
            hints+=("synthesised audio was silent (model loaded but produced no signal)")
        fi
        if grep -q "configured speaker unavailable for Rocky voice" "${voice_log}"; then
            hints+=("speaker device unavailable for Rocky voice")
        fi
        if [[ ${#hints[@]} -eq 0 ]]; then
            hints+=("see voice.log for details")
        fi
        joined=$(printf '; %s' "${hints[@]}")
        joined="${joined:2}"
        RESULTS[$i]="${status}|${name}|${note} -- ${joined}"
        voice_remediation="${joined}"
    done
fi

{
    echo "# Jetson Full Smoke Run ${STAMP}"
    echo
    echo "- Host: $(hostname)"
    echo "- Repo HEAD: $(git -C "${REPO_DIR}" rev-parse --short HEAD)"
    echo "- Branch: $(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)"
    if container_running; then
        image_id="$(docker inspect -f '{{.Image}}' "${CONTAINER}" 2>/dev/null || true)"
        echo "- Container: ${CONTAINER} image=${image_id}"
    else
        echo "- Container: ${CONTAINER} (not running at summary time)"
    fi
    echo "- Run dir: ${RUN_DIR}"
    echo "- ESP32 stages: ${ESP32_MODE}"
    echo
    echo "| Stage | Status | Note |"
    echo "|-------|--------|------|"
    for entry in "${RESULTS[@]}"; do
        IFS='|' read -r status name note <<<"${entry}"
        echo "| ${name} | ${status} | ${note} |"
    done
    if [[ -n "${voice_remediation}" ]]; then
        echo
        echo "## Rocky voice prerequisites"
        echo
        echo "Voice stage diagnostics: ${voice_remediation}"
        echo
        echo "To resolve, ensure ALL of the following are true on the Jetson:"
        echo
        echo "1. \`piper-tts\` is installed inside the \`${CONTAINER}\` container"
        echo "   (rebuild image: \`docker compose -f docker-compose.jetson.yml build --no-cache mousedroid\`)."
        echo "2. The Piper voice model exists at the path referenced by"
        echo "   \`voice.tts_model_path\` in the active runtime config"
        echo "   (default: \`/opt/voice_models/en_US-lessac-medium.onnx\` plus its \`.onnx.json\` sibling)."
        echo "3. The configured USB speaker is enumerated and not held by another process."
        echo
        echo "Full per-stage logs are in \`${RUN_DIR}\` (\`voice.log\` for this stage)."
    fi
    # Ten Pillars section — appended when validate_pillar.sh has run against
    # this same RUN_DIR (MOUSEDROID_PILLAR_REPORT_DIR="${RUN_DIR}").
    ten_pillars_log="${RUN_DIR}/ten_pillars.log"
    if [[ -f "${ten_pillars_log}" ]]; then
        echo
        echo "## Ten Pillars Validation"
        echo
        cat "${ten_pillars_log}"
    fi
} > "${SUMMARY}"

log "Summary written to ${SUMMARY}"
log "Overall: $([[ "${OVERALL_FAIL}" -eq 0 ]] && echo PASS || echo FAIL)"
exit "${OVERALL_FAIL}"
