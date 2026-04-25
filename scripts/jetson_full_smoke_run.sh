#!/bin/bash
# Full Jetson smoke run wrapper (container-first).
#
# All Python execution is delegated to `docker exec mousedroid python3` via a
# transient wrapper script exported as MOUSEDROID_SMOKE_PYTHON, so
# scripts/jetson_smoke_test.sh runs unmodified but executes inside the
# container which already has `mousedroid`, pytest, torch, tensorrt and the
# device passthroughs. The OLED stage is non-blocking; everything else is
# stop-on-first-failure.
#
# Usage (run on the Jetson host, repo at /opt/mousedroid):
#   bash scripts/jetson_full_smoke_run.sh
#
# Optional overrides:
#   MOUSEDROID_SMOKE_REPORT_ROOT  -- defaults to <repo>/reports/jetson_smoke
#   MOUSEDROID_SMOKE_BUS          -- I2C bus for OLED stage (default 7)
#   MOUSEDROID_SMOKE_CONTAINER    -- container name (default mousedroid)
#   MOUSEDROID_JETSON_CONFIGS     -- forwarded to jetson_smoke_test.sh + container

set -uo pipefail

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

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }

record() {
    local name="$1" status="$2" note="${3:-}"
    RESULTS+=("${status}|${name}|${note}")
    log "${status}: ${name} ${note}"
}

run_stage() {
    # $1 = stage label, $2 = blocking? (yes|no), $3 = timeout seconds (0 = none),
    # $4.. = command
    local label="$1" blocking="$2" tmo="$3"; shift 3
    local logfile="${RUN_DIR}/${label}.log"
    log "=== STAGE ${label} (blocking=${blocking} timeout=${tmo}s) ==="
    log "    cmd: $*"
    log "    log: ${logfile}"
    set +e
    if [[ "${tmo}" != "0" ]]; then
        timeout --signal=INT --kill-after=10 "${tmo}" "$@" >"${logfile}" 2>&1
    else
        "$@" >"${logfile}" 2>&1
    fi
    local rc=$?
    set +e
    if [[ ${rc} -eq 0 ]]; then
        record "${label}" "PASS"
        return 0
    fi
    if [[ "${blocking}" == "no" ]]; then
        local why="rc=${rc} (non-blocking)"
        [[ ${rc} -eq 124 || ${rc} -eq 137 ]] && why="rc=${rc} (timeout after ${tmo}s, non-blocking)"
        record "${label}" "EXPECTED-FAIL" "${why}"
        return 0
    fi
    record "${label}" "FAIL" "rc=${rc}"
    OVERALL_FAIL=1
    return 1
}

container_running() {
    docker ps --filter "name=^/${CONTAINER}$" --format '{{.Names}}' | grep -qx "${CONTAINER}"
}

# --- Container python wrapper --------------------------------------------
# jetson_smoke_test.sh resolves "${PYTHON}" from MOUSEDROID_SMOKE_PYTHON
# (must be executable). We point it at this shim that proxies into the
# running container with relevant MOUSEDROID_* env vars forwarded.
PY_WRAPPER="${RUN_DIR}/python3-in-container"
cat > "${PY_WRAPPER}" <<EOF
#!/bin/bash
exec docker exec \\
    -e MOUSEDROID_MOCK_HARDWARE \\
    -e MOUSEDROID_JETSON_CONFIGS \\
    -e MOUSEDROID_FACE_DISPLAY_SMOKE \\
    -e MOUSEDROID_FACE_DISPLAY_BUS \\
    -e PYTHONPATH \\
    ${CONTAINER} python3 "\$@"
EOF
chmod +x "${PY_WRAPPER}"
export MOUSEDROID_SMOKE_PYTHON="${PY_WRAPPER}"

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
} > "${RUN_DIR}/container_health.log" 2>&1
record "container_health" "INFO" "see container_health.log"

# --- Stages 1-7: delegated to jetson_smoke_test.sh via PY_WRAPPER --------
# Bench-debug sensors (camera/audio/lidar/speaker) are non-blocking until the
# physical wiring/overlays land; system/gpio/serial remain hard gates.
for stage in system gpio serial; do
    run_stage "${stage}" "yes" 60 bash scripts/jetson_smoke_test.sh "${stage}" || break
done

if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    for stage in camera audio lidar speaker; do
        run_stage "${stage}" "no" 45 bash scripts/jetson_smoke_test.sh "${stage}"
    done
fi

# --- Stage 8: OLED (non-blocking, container stays up) --------------------
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "oled" "no" 60 \
        docker exec \
            -e MOUSEDROID_FACE_DISPLAY_SMOKE=1 \
            -e MOUSEDROID_FACE_DISPLAY_BUS="${OLED_BUS}" \
            "${CONTAINER}" python3 -m pytest -m hardware -v \
            tests/hardware/test_ssd1306_smoke.py
fi

# --- Stage 9: app health check -------------------------------------------
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "app_health" "yes" 60 bash scripts/jetson_smoke_test.sh app
fi

# --- Stage 10: hardware pytest suite (non-blocking until camera/USB speaker/HC-SR04 are fixed)
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "hardware_pytest" "no" 300 \
        docker exec "${CONTAINER}" python3 -m pytest -m hardware -v tests/hardware/
fi

# --- Stage 11: orchestrator E2E 5s ---------------------------------------
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "e2e" "yes" 60 bash scripts/jetson_smoke_test.sh e2e
fi

# --- Stage 12: LLM live probe --------------------------------------------
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    LLM_PROBE='import asyncio
from mousedroid.config.loader import load_settings
from mousedroid.factory import build_llm_gateway
from mousedroid.validation.runtime import resolve_runtime_config_paths

cfg = load_settings(*resolve_runtime_config_paths())
gw = build_llm_gateway(cfg)

async def main() -> None:
    await gw.start()
    try:
        goal = await gw.translate_mission("move forward slowly")
        print(
            "LLM_PROBE_OK vx={:.3f} vy={:.3f} omega={:.3f}".format(
                goal.vx_target, goal.vy_target, goal.omega_target
            )
        )
    finally:
        await gw.stop()

asyncio.run(main())'
    run_stage "llm_probe" "yes" 120 \
        docker exec "${CONTAINER}" python3 -c "${LLM_PROBE}"
fi

# --- Summary -------------------------------------------------------------
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
    echo
    echo "| Stage | Status | Note |"
    echo "|-------|--------|------|"
    for entry in "${RESULTS[@]}"; do
        IFS='|' read -r status name note <<<"${entry}"
        echo "| ${name} | ${status} | ${note} |"
    done
} > "${SUMMARY}"

log "Summary written to ${SUMMARY}"
log "Overall: $([[ "${OVERALL_FAIL}" -eq 0 ]] && echo PASS || echo FAIL)"
exit "${OVERALL_FAIL}"
