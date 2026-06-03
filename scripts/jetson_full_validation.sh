#!/bin/bash
# Full Jetson on-device validation: static CI -> cold hardware -> warm live.
#
# Composes the existing tooling (does NOT re-implement it) into one ordered,
# artifact-producing pass that exercises every recently-merged surface
# (#106 USB-C, #107/#111 Claude gateway, #113 CI gates, #115 /metrics) on the
# rover. Tolerates the functionally-dead ESP32 (validate-around: no motion;
# serial/motor/power are non-blocking) and follows the runbook's cold-then-warm
# discipline (stop the container for exclusive-device sensor checks, then run
# warm live checks against the running container).
#
# Usage (run on the Jetson host, repo at /opt/mousedroid):
#   bash scripts/jetson_full_validation.sh                 # all phases
#   bash scripts/jetson_full_validation.sh --phase 1       # one phase (0-3; phase 4 report always runs)
#   bash scripts/jetson_full_validation.sh --pytest-only   # hardware pytest tier only
#   bash scripts/jetson_full_validation.sh --dry-run       # print the plan, run nothing
#   bash scripts/jetson_full_validation.sh --help
#
# Env overrides (all optional; documented defaults shown):
#   MOUSEDROID_SMOKE_CONTAINER        container name              (mousedroid)
#   MOUSEDROID_VALIDATION_REPORT_ROOT report root                (<repo>/reports/jetson_full_validation)
#   MOUSEDROID_TELEMETRY_URL          live telemetry base URL    (http://127.0.0.1:8080)
#   MOUSEDROID_JETSON_CONFIG          production overlay         (config/jetson_production.yaml)
#   MOUSEDROID_LIDAR_PROBE_PORT       lidar->WS probe port       (8090)
#   MOUSEDROID_VALIDATION_MISSION     UNKNOWN NL mission         ("navigate to the cantina")
#   VENV_DIR                          host venv dir              (/opt/mousedroid/venv)
#   ANTHROPIC_API_KEY                 cloud Claude key           (unset -> local fallback; presence only checked)
#   MOUSEDROID_TELEMETRY_TOKEN        telemetry bearer token     (unset -> authed checks skip; presence only checked)
#   MOUSEDROID_VALIDATION_HEALTH_RETRIES    warm health-poll attempts  (30)
#   MOUSEDROID_VALIDATION_HEALTH_INTERVAL_S health-poll interval (s)    (1)
#   MOUSEDROID_VALIDATION_HTTP_TIMEOUT_S    curl per-request timeout(s) (5)
#   MOUSEDROID_VALIDATION_PYTEST_TIMEOUT_S  per-test timeout (s)        (120)
#   MOUSEDROID_VALIDATION_LIDAR_DURATION_S  lidar->WS listen window (s) (15)
#   MOUSEDROID_VALIDATION_LOG_TAIL          docker-logs tail lines      (2000)
#
# Secrets are NEVER echoed — only presence is checked. No motion is ever armed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_DIR}"

CONTAINER="${MOUSEDROID_SMOKE_CONTAINER:-mousedroid}"
REPORT_ROOT="${MOUSEDROID_VALIDATION_REPORT_ROOT:-${REPO_DIR}/reports/jetson_full_validation}"
TELEMETRY_URL="${MOUSEDROID_TELEMETRY_URL:-http://127.0.0.1:8080}"
TELEMETRY_URL="${TELEMETRY_URL%/}"
PROD_CONFIG="${MOUSEDROID_JETSON_CONFIG:-config/jetson_production.yaml}"
LIDAR_PROBE_PORT="${MOUSEDROID_LIDAR_PROBE_PORT:-8090}"
UNKNOWN_CMD="${MOUSEDROID_VALIDATION_MISSION:-navigate to the cantina}"
VENV_DIR="${VENV_DIR:-/opt/mousedroid/venv}"

# Tunables — all env-overridable so a flaky uplink / slow boot / long cloud
# round-trip can be accommodated without editing the script (no hardcoded
# values). Defaults suit a healthy bench rover.
HEALTH_RETRIES="${MOUSEDROID_VALIDATION_HEALTH_RETRIES:-30}"
HEALTH_INTERVAL_S="${MOUSEDROID_VALIDATION_HEALTH_INTERVAL_S:-1}"
HTTP_TIMEOUT_S="${MOUSEDROID_VALIDATION_HTTP_TIMEOUT_S:-5}"
PYTEST_TIMEOUT_S="${MOUSEDROID_VALIDATION_PYTEST_TIMEOUT_S:-120}"
LIDAR_PROBE_DURATION_S="${MOUSEDROID_VALIDATION_LIDAR_DURATION_S:-15}"
LOG_TAIL_LINES="${MOUSEDROID_VALIDATION_LOG_TAIL:-2000}"
# Metric namespace — mirrors the schema field metrics.namespace and its env
# override MOUSEDROID_METRICS__NAMESPACE (default "mousedroid"), so the /metrics
# grep tracks an operator-renamed namespace instead of a hardcoded prefix.
NAMESPACE="${MOUSEDROID_METRICS__NAMESPACE:-mousedroid}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${REPORT_ROOT}/${STAMP}"

DRY_RUN=0
PHASE_SEL="all"
PYTEST_ONLY=0

PASSES=0
WARNS=0
FAILURES=0
declare -a RESULTS=()
CONTAINER_WAS_RUNNING=0

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }

usage() {
    # Print the leading comment header (everything from line 2 up to the first
    # non-comment line), stripping the leading '# '. Robust to header growth —
    # no hardcoded line range to drift out of sync.
    awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

record() {
    # $1 = status (PASS|WARN|FAIL), $2 = name, $3 = optional note
    local status="$1" name="$2" note="${3:-}"
    case "${status}" in
        PASS) PASSES=$((PASSES + 1)) ;;
        WARN) WARNS=$((WARNS + 1)) ;;
        FAIL) FAILURES=$((FAILURES + 1)) ;;
    esac
    RESULTS+=("${status}|${name}|${note}")
    log "${status}: ${name}${note:+ — ${note}}"
}

# Resolve a host python (cold phase: container is stopped, so docker exec is
# unavailable). Mirrors jetson_smoke_test.sh resolution order.
HOST_PY=""
resolve_host_python() {
    if [[ -n "${HOST_PY}" ]]; then return 0; fi
    if [[ -x "${VENV_DIR}/bin/python" ]]; then
        HOST_PY="${VENV_DIR}/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        HOST_PY="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        HOST_PY="$(command -v python)"
    else
        return 1
    fi
    return 0
}

have_docker() { command -v docker >/dev/null 2>&1; }

container_running() {
    have_docker || return 1
    [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" == "true" ]]
}

# Run a command, tee output to a per-step log, classify the exit code with a
# blocking policy. $1=name $2=blocking(yes|no) $3=logfile $4..=command.
run_step() {
    local name="$1" blocking="$2" logfile="$3"; shift 3
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "DRY-RUN would run [${name}]: $*"
        record PASS "${name}" "dry-run"
        return 0
    fi
    log "--- ${name} (blocking=${blocking}) ---"
    local rc=0
    "$@" >"${logfile}" 2>&1 || rc=$?
    if [[ ${rc} -eq 0 ]]; then
        record PASS "${name}"
        return 0
    fi
    if [[ "${blocking}" == "no" ]]; then
        record WARN "${name}" "exit=${rc} (non-blocking; see ${logfile##*/})"
        return 0
    fi
    record FAIL "${name}" "exit=${rc} (see ${logfile##*/})"
    return "${rc}"
}

restore_container() {
    # Always bring the rover brain back up if we stopped it — never leave it down.
    if [[ "${CONTAINER_WAS_RUNNING}" == "1" ]] && have_docker; then
        if ! container_running; then
            log "restoring container ${CONTAINER} (was running before cold phase)"
            docker start "${CONTAINER}" >/dev/null 2>&1 || log "WARN: docker start ${CONTAINER} failed"
        fi
    fi
}
trap restore_container EXIT

# Run the hardware pytest tier (host venv). Shared by Phase 2 and --pytest-only
# so the invocation lives in exactly one place (validate-around the dead ESP32;
# the API key, when present, lets Test B reach the cloud primary). $1 = logfile.
run_hardware_pytest() {
    local logfile="$1"
    run_step "hardware pytest (-m hardware)" no "${logfile}" \
        env MOUSEDROID_MOCK_HARDWARE=false MOUSEDROID_ESP32__ENABLED=false \
            MOUSEDROID_JETSON_CONFIG="${PROD_CONFIG}" \
        "${HOST_PY}" -m pytest -m hardware tests/hardware/ \
        --import-mode=importlib --timeout="${PYTEST_TIMEOUT_S}" -q
}

# --------------------------------------------------------------------------- #
# Phase 0 — preconditions (blocking)
# --------------------------------------------------------------------------- #
phase0() {
    log "=== PHASE 0: preconditions ==="
    local logfile="${RUN_DIR}/phase0.log"
    : >"${logfile}"

    if [[ "$(uname -s)" == "Linux" && -e /etc/nv_tegra_release ]]; then
        record PASS "host is Jetson"
    else
        record WARN "host is NOT Jetson" "hardware phases will skip/mock"
    fi

    if [[ -f "${PROD_CONFIG}" ]]; then
        record PASS "config present" "${PROD_CONFIG}"
    else
        record FAIL "config present" "missing ${PROD_CONFIG}"
    fi

    if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        record PASS "ANTHROPIC_API_KEY present"
    else
        record WARN "ANTHROPIC_API_KEY absent" "cloud tier will fall back to local; Test B asserts served-only"
    fi

    if [[ -n "${MOUSEDROID_TELEMETRY_TOKEN:-}" ]]; then
        record PASS "MOUSEDROID_TELEMETRY_TOKEN present"
    else
        record WARN "MOUSEDROID_TELEMETRY_TOKEN absent" "authed checks (Test C) skip; /metrics is auth-exempt"
    fi
}

# --------------------------------------------------------------------------- #
# Phase 1 — static CI (mock hardware)
# --------------------------------------------------------------------------- #
phase1() {
    log "=== PHASE 1: static CI (mock hardware) ==="
    local logfile="${RUN_DIR}/phase1_ci.log"
    if container_running; then
        # Container branch: run all three checks via docker exec so coverage
        # matches the host branch (ci.sh + preflight + pillars).
        run_step "static CI (ci.sh, container)" yes "${logfile}" \
            docker exec -e MOUSEDROID_MOCK_HARDWARE=true "${CONTAINER}" \
            bash -lc "cd /opt/mousedroid && bash scripts/ci.sh"
        run_step "preflight (mock)" yes "${RUN_DIR}/phase1_preflight.log" \
            docker exec "${CONTAINER}" python3 -m mousedroid.cli.preflight --mock-hardware --json
        run_step "pillars (dry-run)" yes "${RUN_DIR}/phase1_pillars.log" \
            docker exec "${CONTAINER}" python3 -m mousedroid.cli.validate_pillars \
            --config "${PROD_CONFIG}" --dry-run --json
    else
        resolve_host_python || { record FAIL "static CI" "no python found"; return; }
        run_step "static CI (ci.sh, host)" yes "${logfile}" \
            env MOUSEDROID_MOCK_HARDWARE=true MOUSEDROID_PYTHON="${HOST_PY}" bash scripts/ci.sh
        run_step "preflight (mock)" yes "${RUN_DIR}/phase1_preflight.log" \
            "${HOST_PY}" -m mousedroid.cli.preflight --mock-hardware --json
        run_step "pillars (dry-run)" yes "${RUN_DIR}/phase1_pillars.log" \
            "${HOST_PY}" -m mousedroid.cli.validate_pillars --config "${PROD_CONFIG}" --dry-run --json
    fi
}

# --------------------------------------------------------------------------- #
# Phase 2 — cold hardware (container stopped; exclusive device access)
# --------------------------------------------------------------------------- #
phase2() {
    log "=== PHASE 2: cold hardware ==="
    if ! have_docker; then
        record WARN "cold hardware" "docker unavailable — skipping cold phase"
        return
    fi
    resolve_host_python || { record FAIL "cold hardware" "no host python"; return; }

    if container_running; then CONTAINER_WAS_RUNNING=1; fi
    if [[ "${DRY_RUN}" != "1" && "${CONTAINER_WAS_RUNNING}" == "1" ]]; then
        log "stopping container ${CONTAINER} for exclusive device access"
        docker stop "${CONTAINER}" >/dev/null 2>&1 || record WARN "docker stop" "stop failed"
    fi

    # Real preflight + per-sensor probe (host venv).
    run_step "preflight (real)" yes "${RUN_DIR}/phase2_preflight.log" \
        "${HOST_PY}" -m mousedroid.cli.preflight --config "${PROD_CONFIG}" --json
    run_step "verify_sensors (all)" no "${RUN_DIR}/phase2_sensors.log" \
        "${HOST_PY}" scripts/verify_sensors.py --sensor all --json

    # Smoke stages via jetson_smoke_test.sh single-stage interface (host venv).
    # ESP32-owned stages (serial/motor/power) are NON-blocking (validate-around
    # the dead ESP32); motion stays disabled (MOUSEDROID_SMOKE_ALLOW_MOTION unset).
    local stage
    for stage in system usbc gpio camera lidar audio speaker voice pcie_ssd hailo; do
        run_step "smoke:${stage}" yes "${RUN_DIR}/phase2_smoke_${stage}.log" \
            env MOUSEDROID_SMOKE_PYTHON="${HOST_PY}" MOUSEDROID_JETSON_CONFIGS="${PROD_CONFIG}" \
            bash scripts/jetson_smoke_test.sh "${stage}"
    done
    for stage in serial motor power; do
        run_step "smoke:${stage}" no "${RUN_DIR}/phase2_smoke_${stage}.log" \
            env MOUSEDROID_SMOKE_PYTHON="${HOST_PY}" MOUSEDROID_JETSON_CONFIGS="${PROD_CONFIG}" \
            bash scripts/jetson_smoke_test.sh "${stage}"
    done

    # Hardware pytest tier — Test B (in-process orchestrator -> real Claude ->
    # orch._metrics) populates and asserts the #115 families HERE.
    run_hardware_pytest "${RUN_DIR}/phase2_pytest.log"

    # Real pillar validation (10 pillars).
    run_step "pillars (real)" yes "${RUN_DIR}/phase2_pillars.log" \
        "${HOST_PY}" -m mousedroid.cli.validate_pillars --config "${PROD_CONFIG}" --json

    if [[ "${DRY_RUN}" != "1" && "${CONTAINER_WAS_RUNNING}" == "1" ]]; then
        log "restarting container ${CONTAINER}"
        docker start "${CONTAINER}" >/dev/null 2>&1 || record WARN "docker start" "restart failed"
    fi
}

# --------------------------------------------------------------------------- #
# Phase 3 — warm live (container running)
# --------------------------------------------------------------------------- #
wait_for_health() {
    local url="${TELEMETRY_URL}/api/v1/health" i
    for ((i = 0; i < HEALTH_RETRIES; i++)); do
        if curl -fsS --max-time "${HTTP_TIMEOUT_S}" "${url}" >/dev/null 2>&1; then return 0; fi
        sleep "${HEALTH_INTERVAL_S}"
    done
    return 1
}

phase3() {
    log "=== PHASE 3: warm live ==="
    if [[ "${DRY_RUN}" == "1" ]]; then
        record PASS "health endpoint" "dry-run"
        record PASS "translate_mission probe" "dry-run"
        record PASS "live /metrics scrape" "dry-run"
        record PASS "lidar telemetry probe" "dry-run"
        record PASS "structlog evidence captured" "dry-run"
        return
    fi
    if ! command -v curl >/dev/null 2>&1; then
        record WARN "warm live" "curl unavailable — skipping warm checks"
        return
    fi

    if wait_for_health; then
        record PASS "health endpoint" "${TELEMETRY_URL}/api/v1/health"
    else
        record FAIL "health endpoint" "no 200 from ${TELEMETRY_URL}/api/v1/health"
    fi

    # LLM gateway dry-run (own registry; bypasses the rule parser).
    if container_running; then
        run_step "translate_mission probe" no "${RUN_DIR}/phase3_translate.log" \
            docker exec "${CONTAINER}" python3 scripts/translate_mission.py --mission "${UNKNOWN_CMD}"
    fi

    # Live /metrics is auth-exempt: confirm it is a healthy Prometheus surface.
    # (Dry-run is already short-circuited at the top of phase3.)
    local metrics_log="${RUN_DIR}/phase3_metrics.log"
    if curl -fsS --max-time "${HTTP_TIMEOUT_S}" "${TELEMETRY_URL}/metrics" >"${metrics_log}" 2>&1; then
        if grep -q "^${NAMESPACE}_" "${metrics_log}"; then
            record PASS "live /metrics scrape" "${NAMESPACE}_ namespace present"
        else
            record WARN "live /metrics scrape" "200 but no ${NAMESPACE}_ samples"
        fi
        # Note which #115 families are already populated (informational — prod
        # has no HTTP ingress, so population is proven by Phase-2 Test B).
        if grep -q "${NAMESPACE}_llm_gateway_served_total{" "${metrics_log}"; then
            record PASS "#115 served counter visible on /metrics"
        else
            record WARN "#115 families not yet populated on live /metrics" \
                "expected — no HTTP mission ingress on prod (openclaw disabled); see Phase-2 Test B"
        fi
    else
        record FAIL "live /metrics scrape" "GET ${TELEMETRY_URL}/metrics failed"
    fi

    # LiDAR -> telemetry WebSocket probe (non-default port avoids collision).
    if container_running; then
        run_step "lidar telemetry probe" no "${RUN_DIR}/phase3_lidar_ws.log" \
            docker exec "${CONTAINER}" python3 tools/lidar_telemetry_probe.py \
            --port "${LIDAR_PROBE_PORT}" --duration "${LIDAR_PROBE_DURATION_S}"
    fi

    # Structured-log evidence for operator triage (best-effort).
    if container_running && [[ "${DRY_RUN}" != "1" ]]; then
        docker logs --tail "${LOG_TAIL_LINES}" "${CONTAINER}" 2>&1 \
            | grep -E 'usbc_endpoint_|esp32_serial_port_overridden|power_chain_probe_complete|esp32_raw_line|anthropic_gateway_|fallback_gateway_started' \
            >"${RUN_DIR}/phase3_structlog.log" 2>&1 || true
        record PASS "structlog evidence captured" "phase3_structlog.log"
    fi
}

# --------------------------------------------------------------------------- #
# --pytest-only — hardware tier in isolation
# --------------------------------------------------------------------------- #
pytest_only() {
    log "=== --pytest-only: hardware tier ==="
    resolve_host_python || { record FAIL "hardware pytest" "no host python"; return; }
    run_hardware_pytest "${RUN_DIR}/pytest_only.log"
}

# --------------------------------------------------------------------------- #
# Phase 4 — report + gate
# --------------------------------------------------------------------------- #
write_summary() {
    local summary="${RUN_DIR}/SUMMARY.md"
    {
        echo "# Jetson full-validation summary"
        echo ""
        echo "- UTC: ${STAMP}"
        echo "- Repo: ${REPO_DIR}"
        echo "- Config: ${PROD_CONFIG}"
        echo "- Telemetry: ${TELEMETRY_URL}"
        echo "- Totals: PASS=${PASSES} WARN=${WARNS} FAIL=${FAILURES}"
        echo ""
        echo "| Status | Check | Note |"
        echo "|--------|-------|------|"
        local row status name note
        for row in "${RESULTS[@]}"; do
            IFS='|' read -r status name note <<< "${row}"
            echo "| ${status} | ${name} | ${note} |"
        done
    } >"${summary}"
    log "summary written: ${summary}"
}

phase4() {
    log "=== PHASE 4: report + gate ==="
    write_summary
    log "TOTALS: PASS=${PASSES} WARN=${WARNS} FAIL=${FAILURES}"
    if [[ ${FAILURES} -gt 0 ]]; then
        log "RESULT: FAIL (${FAILURES} blocking failure(s))"
        return 1
    fi
    log "RESULT: PASS (WARN=${WARNS} non-blocking)"
    return 0
}

# --------------------------------------------------------------------------- #
# Arg parsing + dispatch
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --pytest-only) PYTEST_ONLY=1; shift ;;
        --phase) PHASE_SEL="${2:-all}"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

mkdir -p "${RUN_DIR}"
log "report dir: ${RUN_DIR}"

if [[ "${PYTEST_ONLY}" == "1" ]]; then
    pytest_only
    phase4; exit $?
fi

case "${PHASE_SEL}" in
    all) phase0; phase1; phase2; phase3 ;;
    0) phase0 ;;
    1) phase1 ;;
    2) phase2 ;;
    3) phase3 ;;
    *) echo "Invalid --phase '${PHASE_SEL}' (use 0-3 or all; phase 4 report+gate always runs last)" >&2; exit 2 ;;
esac

phase4
exit $?
