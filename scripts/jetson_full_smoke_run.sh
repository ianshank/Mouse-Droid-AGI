#!/bin/bash
# Full Jetson smoke run wrapper.
#
# Drives every smoke stage with stop-on-first-failure semantics, except the
# OLED stage which is recorded as EXPECTED-FAIL when the panel is absent so
# the rest of the surface still validates. Per-stage logs and a SUMMARY.md
# are written under reports/jetson_smoke/<UTC>/.
#
# Usage (run on the Jetson host, repo at /opt/mousedroid):
#   bash scripts/jetson_full_smoke_run.sh
#
# Optional overrides:
#   MOUSEDROID_SMOKE_REPORT_ROOT  -- defaults to <repo>/reports/jetson_smoke
#   MOUSEDROID_SMOKE_BUS          -- I2C bus for OLED stage (default 7)
#   MOUSEDROID_SMOKE_CONTAINER    -- container name (default mousedroid)
#   MOUSEDROID_JETSON_CONFIGS     -- forwarded to jetson_smoke_test.sh

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
    # $1 = stage label (filename safe), $2 = blocking? (yes|no), $3.. command
    local label="$1" blocking="$2"; shift 2
    local logfile="${RUN_DIR}/${label}.log"
    log "=== STAGE ${label} (blocking=${blocking}) ==="
    log "    cmd: $*"
    log "    log: ${logfile}"
    if "$@" >"${logfile}" 2>&1; then
        record "${label}" "PASS"
        return 0
    fi
    local rc=$?
    if [[ "${blocking}" == "no" ]]; then
        record "${label}" "EXPECTED-FAIL" "rc=${rc} (non-blocking)"
        return 0
    fi
    record "${label}" "FAIL" "rc=${rc}"
    OVERALL_FAIL=1
    return 1
}

container_running() {
    docker ps --filter "name=^/${CONTAINER}$" --format '{{.Names}}' | grep -qx "${CONTAINER}"
}

wait_container_healthy() {
    local timeout="${1:-60}" elapsed=0
    while ((elapsed < timeout)); do
        local status
        status="$(docker inspect -f '{{.State.Health.Status}}' "${CONTAINER}" 2>/dev/null || echo missing)"
        if [[ "${status}" == "healthy" || "${status}" == "starting" ]]; then
            if [[ "${status}" == "healthy" ]]; then return 0; fi
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log "WARN: container did not reach healthy in ${timeout}s (last: ${status:-unknown})"
    return 0
}

# --- Stage 1-7: host-side script stages -----------------------------------
for stage in system gpio serial camera audio lidar speaker; do
    run_stage "${stage}" "yes" bash scripts/jetson_smoke_test.sh "${stage}" || break
done

# --- Stage 8: OLED gated, container stopped -------------------------------
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    log "Stopping container ${CONTAINER} for OLED stage..."
    docker stop "${CONTAINER}" >/dev/null 2>&1 || true
    OLED_LOG="${RUN_DIR}/oled.log"
    if MOUSEDROID_FACE_DISPLAY_SMOKE=1 \
       MOUSEDROID_FACE_DISPLAY_BUS="${OLED_BUS}" \
       /opt/mousedroid/venv/bin/pytest -m hardware -v \
           tests/hardware/test_ssd1306_smoke.py >"${OLED_LOG}" 2>&1; then
        record "oled" "PASS"
    else
        record "oled" "EXPECTED-FAIL" "panel still bench-debug; non-blocking"
    fi
    log "Restarting container ${CONTAINER}..."
    docker start "${CONTAINER}" >/dev/null 2>&1 || true
    wait_container_healthy 90
fi

# --- Stage 9-12: container/runtime stages ---------------------------------
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "app_health" "yes" \
        docker exec "${CONTAINER}" python -m mousedroid.main --health-check
fi

if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "hardware_pytest" "yes" bash scripts/jetson_smoke_test.sh pytest
fi

if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "e2e" "yes" bash scripts/jetson_smoke_test.sh e2e
fi

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
    run_stage "llm_probe" "yes" \
        docker exec "${CONTAINER}" python -c "${LLM_PROBE}"
fi

# --- Summary ---------------------------------------------------------------
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
