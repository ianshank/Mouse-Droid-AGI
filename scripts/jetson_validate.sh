#!/bin/bash
# Phase A - automated on-Jetson validation harness.
#
# SSH into the Jetson and runs the bring-up gates in order:
#   1. scripts/preflight_check.sh
#   2. scripts/verify_sensors.py for the configured sensor subset
#   3. targeted hardware pytest files (avoids known full-suite fixture gaps)
#   4. selected sections from scripts/jetson_smoke_test.sh
#   5. python -m mousedroid.main --health-check with explicit config overlays
#
# Output from each step is captured to reports/jetson_validate/<timestamp>/.
# A non-zero overall exit code is returned if ANY gate fails, and the failing
# gate is reported in the summary.
#
# Usage:
#   bash scripts/jetson_validate.sh [user@]jetson-host [--user USER]
#   bash scripts/jetson_validate.sh [user@]jetson-host [--step preflight|verify|pytest|smoke|health|all]
#
# Host resolution matches scripts/deploy_remote.sh:
#   1. first positional argument
#   2. ~/.mousedroid/jetson_host
#   3. scripts/jetson_discover.sh (mDNS/avahi)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
REMOTE_USER="${MOUSEDROID_REMOTE_USER:-jetson}"
REMOTE_SRC="${MOUSEDROID_REMOTE_SRC:-/opt/mousedroid/src}"
REMOTE_VENV="${MOUSEDROID_REMOTE_VENV:-/opt/mousedroid/venv}"
REMOTE_CONFIG_DIR="${MOUSEDROID_REMOTE_CONFIG_DIR:-/etc/mousedroid}"
REMOTE_CONFIGS_CSV="${MOUSEDROID_REMOTE_CONFIGS:-${REMOTE_CONFIG_DIR}/jetson_production.yaml,${REMOTE_CONFIG_DIR}/jetson_lidar_only.yaml}"
REMOTE_CONTAINER="${MOUSEDROID_REMOTE_CONTAINER:-mousedroid}"
REMOTE_EXECUTION_MODE="${MOUSEDROID_REMOTE_EXECUTION_MODE:-auto}"
SSH_CONNECT_TIMEOUT="${MOUSEDROID_SSH_CONNECT_TIMEOUT:-10}"
SSH_STRICT_HOST_KEY_CHECKING="${MOUSEDROID_SSH_STRICT_HOST_KEY_CHECKING:-accept-new}"
SSH_KNOWN_HOSTS_FILE="${MOUSEDROID_SSH_KNOWN_HOSTS_FILE:-}"
ESP32_DEV="${MOUSEDROID_ESP32_DEV:-/dev/ttyUSB0}"
CAMERA_DEV="${MOUSEDROID_CAMERA_DEV:-/dev/video0}"
LIDAR_DEV="${MOUSEDROID_LIDAR_DEV:-/dev/ttyUSB1}"
VERIFY_SENSORS_CSV="${MOUSEDROID_VERIFY_SENSORS:-camera,audio,lidar,speaker}"
PYTEST_TARGETS_CSV="${MOUSEDROID_HARDWARE_PYTEST_TARGETS:-tests/hardware/test_jetson_smoke.py::test_gpu_available,tests/hardware/test_jetson_smoke.py::test_tensorrt_importable,tests/hardware/test_jetson_smoke.py::test_gpio_pins_accessible,tests/hardware/test_jetson_smoke.py::test_serial_port_exists,tests/hardware/test_jetson_smoke.py::test_camera_capture,tests/hardware/test_jetson_smoke.py::test_health_monitor,tests/hardware/test_jetson_smoke.py::test_esp32_connect,tests/hardware/test_ld19_smoke.py,tests/hardware/test_mic_smoke.py,tests/hardware/test_speaker_smoke.py}"
PYTEST_EXTRA_ARGS="${MOUSEDROID_PYTEST_EXTRA_ARGS:-}"
SMOKE_STEPS_CSV="${MOUSEDROID_SMOKE_STEPS:-system,gpio,serial,camera,audio,lidar,speaker}"
STEP="all"
HOST=""

TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${PROJECT_DIR}/reports/jetson_validate/${TS}"

declare -a GATE_RESULTS=()
OVERALL_EXIT=0
REMOTE_PYTHON=""
REMOTE_RUNTIME_DESC="unresolved"
RUN_APP_IN_CONTAINER=false

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log_section() { echo ""; echo "=== $1 === [$(ts)]"; }
log_step()    { echo "--- $1 ---"; }
die()         { echo "ERROR: $1" >&2; exit 1; }

csv_to_words() {
    local csv="$1"
    csv="${csv//,/ }"
    echo "${csv}"
}

build_remote_config_args() {
    local cfg
    local args=()
    for cfg in $(csv_to_words "${REMOTE_CONFIGS_CSV}"); do
        [[ -z "${cfg}" ]] && continue
        args+=("${cfg}")
    done
    echo "${args[*]}"
}

ssh_with_options() {
    local -a opts=(
        -o "ConnectTimeout=${SSH_CONNECT_TIMEOUT}"
        -o "StrictHostKeyChecking=${SSH_STRICT_HOST_KEY_CHECKING}"
    )

    if [[ -n "${SSH_KNOWN_HOSTS_FILE}" ]]; then
        opts+=( -o "UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}" )
    fi

    ssh "${opts[@]}" "${REMOTE_USER}@${HOST}" "$@"
}

remote_cmd() {
    ssh_with_options "$@"
}

resolve_remote_source_root() {
    local candidate
    local resolved_src=""

    for candidate in "${REMOTE_SRC}" "/opt/mousedroid"; do
        if remote_cmd "test -f ${candidate}/scripts/preflight_check.sh"; then
            resolved_src="${candidate}"
            break
        fi
    done

    if [[ -z "${resolved_src}" ]]; then
        die "Remote project root not found. Checked ${REMOTE_SRC} and /opt/mousedroid"
    fi

    REMOTE_SRC="${resolved_src}"
}

resolve_remote_runtime() {
    case "${REMOTE_EXECUTION_MODE}" in
        host)
            if remote_cmd "test -x ${REMOTE_VENV}/bin/python"; then
                REMOTE_PYTHON="${REMOTE_VENV}/bin/python"
                REMOTE_RUNTIME_DESC="host:${REMOTE_PYTHON}"
                RUN_APP_IN_CONTAINER=false
                return
            fi
            die "Host execution requested but python not found at ${REMOTE_VENV}/bin/python"
            ;;
        container)
            if remote_cmd "docker ps --filter \"name=${REMOTE_CONTAINER}\" --format '{{.Names}}' | grep -qx '${REMOTE_CONTAINER}'"; then
                REMOTE_PYTHON="python3"
                REMOTE_RUNTIME_DESC="container:${REMOTE_CONTAINER}:${REMOTE_PYTHON}"
                RUN_APP_IN_CONTAINER=true
                return
            fi
            die "Container execution requested but container '${REMOTE_CONTAINER}' is not running"
            ;;
        auto)
            if remote_cmd "test -x ${REMOTE_VENV}/bin/python"; then
                REMOTE_PYTHON="${REMOTE_VENV}/bin/python"
                REMOTE_RUNTIME_DESC="host:${REMOTE_PYTHON}"
                RUN_APP_IN_CONTAINER=false
                return
            fi
            if remote_cmd "docker ps --filter \"name=${REMOTE_CONTAINER}\" --format '{{.Names}}' | grep -qx '${REMOTE_CONTAINER}'"; then
                REMOTE_PYTHON="python3"
                REMOTE_RUNTIME_DESC="container:${REMOTE_CONTAINER}:${REMOTE_PYTHON}"
                RUN_APP_IN_CONTAINER=true
                return
            fi
            die "No usable remote runtime found. Checked host python at ${REMOTE_VENV}/bin/python and container '${REMOTE_CONTAINER}'"
            ;;
        *)
            die "Invalid MOUSEDROID_REMOTE_EXECUTION_MODE='${REMOTE_EXECUTION_MODE}'. Use auto|host|container"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --step)
                STEP="${2:-all}"
                shift 2
                ;;
            --user)
                REMOTE_USER="${2:-${REMOTE_USER}}"
                shift 2
                ;;
            --help|-h)
                sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
                exit 0
                ;;
            -*)
                die "Unknown option: $1"
                ;;
            *)
                if [[ -z "${HOST}" ]]; then
                    if [[ "$1" == *@* ]]; then
                        REMOTE_USER="${1%@*}"
                        HOST="${1#*@}"
                    else
                        HOST="$1"
                    fi
                else
                    die "Unexpected argument: $1"
                fi
                shift
                ;;
        esac
    done
    case "${STEP}" in
        all|preflight|verify|pytest|smoke|health) ;;
        *) die "Invalid --step '${STEP}'. Use: all|preflight|verify|pytest|smoke|health" ;;
    esac
}

# ---------------------------------------------------------------------------
# Host resolution (mirrors deploy_remote.sh)
# ---------------------------------------------------------------------------
resolve_host() {
    if [[ -n "${HOST}" ]]; then
        log_step "Using host from argument: ${HOST}"
        return
    fi

    local host_file="${HOME}/.mousedroid/jetson_host"
    if [[ -f "${host_file}" ]]; then
        HOST="$(tr -d '[:space:]' < "${host_file}")"
        if [[ -n "${HOST}" ]]; then
            log_step "Using host from ${host_file}: ${HOST}"
            return
        fi
    fi

    if [[ -x "${SCRIPT_DIR}/jetson_discover.sh" ]]; then
        log_step "Attempting mDNS/avahi discovery..."
        HOST="$("${SCRIPT_DIR}/jetson_discover.sh" 2>/dev/null || true)"
        if [[ -n "${HOST}" ]]; then
            log_step "Discovered Jetson at: ${HOST}"
            mkdir -p "${HOME}/.mousedroid"
            echo "${HOST}" > "${host_file}"
            return
        fi
    fi

    die "Cannot determine Jetson host. Provide as arg or save to ~/.mousedroid/jetson_host"
}

check_connectivity() {
    log_step "Checking SSH connectivity to ${REMOTE_USER}@${HOST}..."
    if ! remote_cmd "echo ok" &>/dev/null; then
        die "Cannot SSH to ${REMOTE_USER}@${HOST}. Check connectivity and SSH keys."
    fi
    log_step "SSH connection verified"
}

check_remote_layout() {
    log_step "Checking remote layout on ${HOST}..."
    resolve_remote_source_root
    log_step "Resolved remote project root: ${REMOTE_SRC}"

    if [[ "${STEP}" == "preflight" ]]; then
        REMOTE_RUNTIME_DESC="host-shell-only"
        return
    fi

    if ! remote_cmd "test -f ${REMOTE_SRC}/scripts/verify_sensors.py"; then
        die "Remote source missing verify_sensors.py at ${REMOTE_SRC}. Sync code first."
    fi
    if ! remote_cmd "test -f ${REMOTE_SRC}/scripts/jetson_smoke_test.sh"; then
        die "Remote smoke test script missing at ${REMOTE_SRC}/scripts/jetson_smoke_test.sh"
    fi

    resolve_remote_runtime
    log_step "Resolved remote runtime: ${REMOTE_RUNTIME_DESC}"

    if [[ "${STEP}" == "health" || "${STEP}" == "all" ]]; then
        local cfg
        for cfg in $(csv_to_words "${REMOTE_CONFIGS_CSV}"); do
            [[ -z "${cfg}" ]] && continue
            if ! remote_cmd "test -f ${cfg}"; then
                die "Remote config missing at ${cfg}. Deploy configs first."
            fi
        done
    fi
}

# ---------------------------------------------------------------------------
# Gate runner - streams output live and tees to ${REPORT_DIR}/<gate>.log.
# ---------------------------------------------------------------------------
run_gate() {
    local name="$1"
    local cmd="$2"
    local log="${REPORT_DIR}/${name}.log"
    log_section "Gate: ${name}"
    echo "  remote: ${cmd}"
    echo "  log:    ${log}"

    local rc=0
    set +e
    ssh_with_options "${cmd}" 2>&1 | tee "${log}"
    rc="${PIPESTATUS[0]}"
    set -e

    if [[ ${rc} -eq 0 ]]; then
        GATE_RESULTS+=("PASS ${name} (log=${log})")
        log_step "Gate ${name}: PASS"
    else
        GATE_RESULTS+=("FAIL ${name} rc=${rc} (log=${log})")
        OVERALL_EXIT=1
        log_step "Gate ${name}: FAIL (rc=${rc})"
    fi
}

gate_verify() {
    local sensors
    sensors="$(csv_to_words "${VERIFY_SENSORS_CSV}")"
    if [[ "${RUN_APP_IN_CONTAINER}" == true ]]; then
        run_gate "verify_sensors" \
            "docker exec -w ${REMOTE_SRC} ${REMOTE_CONTAINER} bash -lc 'rc=0; for sensor in ${sensors}; do MOUSEDROID_JETSON_CONFIGS=${REMOTE_CONFIGS_CSV} MOUSEDROID_MOCK_HARDWARE=false ${REMOTE_PYTHON} scripts/verify_sensors.py --sensor \"\$sensor\" || rc=1; done; exit \$rc'"
    else
        run_gate "verify_sensors" \
            "cd ${REMOTE_SRC} && rc=0 && \
             for sensor in ${sensors}; do \
                 MOUSEDROID_JETSON_CONFIGS=${REMOTE_CONFIGS_CSV} MOUSEDROID_MOCK_HARDWARE=false ${REMOTE_PYTHON} scripts/verify_sensors.py --sensor \"\$sensor\" || rc=1; \
             done && exit \$rc"
    fi
}

gate_preflight() {
    run_gate "preflight_check" \
        "cd ${REMOTE_SRC} && \
         MOUSEDROID_ESP32_DEV=${ESP32_DEV} \
         MOUSEDROID_CAMERA_DEV=${CAMERA_DEV} \
         MOUSEDROID_LIDAR_DEV=${LIDAR_DEV} \
         bash scripts/preflight_check.sh"
}

gate_pytest() {
    local pytest_targets
    pytest_targets="$(csv_to_words "${PYTEST_TARGETS_CSV}")"
    if [[ "${RUN_APP_IN_CONTAINER}" == true ]]; then
        run_gate "pytest_hardware" \
            "docker exec -w ${REMOTE_SRC} ${REMOTE_CONTAINER} bash -lc 'MOUSEDROID_JETSON_CONFIGS=${REMOTE_CONFIGS_CSV} MOUSEDROID_MOCK_HARDWARE=false ${REMOTE_PYTHON} -m pytest -vv ${PYTEST_EXTRA_ARGS} ${pytest_targets}'"
    else
        run_gate "pytest_hardware" \
            "cd ${REMOTE_SRC} && \
             MOUSEDROID_JETSON_CONFIGS=${REMOTE_CONFIGS_CSV} \
             MOUSEDROID_MOCK_HARDWARE=false \
             ${REMOTE_PYTHON} -m pytest -vv ${PYTEST_EXTRA_ARGS} ${pytest_targets}"
    fi
}

gate_smoke() {
    local smoke_steps
    smoke_steps="$(csv_to_words "${SMOKE_STEPS_CSV}")"
    if [[ "${RUN_APP_IN_CONTAINER}" == true ]]; then
        run_gate "jetson_smoke_test" \
            "docker exec -w ${REMOTE_SRC} ${REMOTE_CONTAINER} bash -lc 'mkdir -p /tmp/vshim/bin && ln -sf /usr/local/bin/python3 /tmp/vshim/bin/python && rc=0; for step in ${smoke_steps}; do MOUSEDROID_JETSON_CONFIGS=${REMOTE_CONFIGS_CSV} VENV_DIR=/tmp/vshim bash scripts/jetson_smoke_test.sh \"\$step\" || rc=1; done; exit \$rc'"
    else
        run_gate "jetson_smoke_test" \
            "cd ${REMOTE_SRC} && rc=0 && \
             for step in ${smoke_steps}; do \
                 MOUSEDROID_JETSON_CONFIGS=${REMOTE_CONFIGS_CSV} VENV_DIR=${REMOTE_VENV} bash scripts/jetson_smoke_test.sh \"\${step}\" || rc=1; \
             done && exit \$rc"
    fi
}

gate_health() {
    local config_args
    config_args="$(build_remote_config_args)"
    if [[ "${RUN_APP_IN_CONTAINER}" == true ]]; then
        run_gate "mousedroid_health_check" \
            "docker exec -w ${REMOTE_SRC} ${REMOTE_CONTAINER} bash -lc 'MOUSEDROID_MOCK_HARDWARE=false ${REMOTE_PYTHON} -m mousedroid.main --config ${config_args} --health-check'"
    else
        run_gate "mousedroid_health_check" \
            "cd ${REMOTE_SRC} && \
             MOUSEDROID_MOCK_HARDWARE=false \
             ${REMOTE_PYTHON} -m mousedroid.main --config ${config_args} --health-check"
    fi
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print_summary() {
    local summary="${REPORT_DIR}/summary.txt"
    {
        echo "MouseDroid Jetson Validation - Phase A"
        echo "Host:    ${REMOTE_USER}@${HOST}"
        echo "Step:    ${STEP}"
        echo "When:    $(ts)"
        echo "Reports: ${REPORT_DIR}"
        echo "Configs: ${REMOTE_CONFIGS_CSV}"
        echo "Verify:  ${VERIFY_SENSORS_CSV}"
        echo "Pytest:  ${PYTEST_TARGETS_CSV}"
        echo "PyArgs:  ${PYTEST_EXTRA_ARGS:-<none>}"
        echo "Smoke:   ${SMOKE_STEPS_CSV}"
        echo "Runtime: ${REMOTE_RUNTIME_DESC}"
        echo "SSH:     strict=${SSH_STRICT_HOST_KEY_CHECKING} known_hosts=${SSH_KNOWN_HOSTS_FILE:-<default>} timeout=${SSH_CONNECT_TIMEOUT}"
        echo ""
        echo "Gate results:"
        for r in "${GATE_RESULTS[@]}"; do
            echo "  ${r}"
        done
        echo ""
        if [[ ${OVERALL_EXIT} -eq 0 ]]; then
            echo "OVERALL: PASS"
        else
            echo "OVERALL: FAIL"
        fi
    } | tee "${summary}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    parse_args "$@"

    log_section "MouseDroid Jetson Validation (Phase A)"
    echo "  Step:    ${STEP}"
    echo "  Reports: ${REPORT_DIR}"
    echo "  Host:    ${REMOTE_USER}@${HOST:-<auto>}"
    echo "  Configs: ${REMOTE_CONFIGS_CSV}"
    echo "  Verify:  ${VERIFY_SENSORS_CSV}"
    echo "  Pytest:  ${PYTEST_TARGETS_CSV}"
    echo "  PyArgs:  ${PYTEST_EXTRA_ARGS:-<none>}"
    echo "  Smoke:   ${SMOKE_STEPS_CSV}"
    echo "  Runtime: ${REMOTE_RUNTIME_DESC}"
    echo "  SSH:     strict=${SSH_STRICT_HOST_KEY_CHECKING} known_hosts=${SSH_KNOWN_HOSTS_FILE:-<default>} timeout=${SSH_CONNECT_TIMEOUT}"

    mkdir -p "${REPORT_DIR}"

    resolve_host
    check_connectivity
    check_remote_layout

    case "${STEP}" in
        all)
            gate_preflight
            gate_verify
            gate_pytest
            gate_smoke
            gate_health
            ;;
        preflight) gate_preflight ;;
        verify) gate_verify ;;
        pytest) gate_pytest ;;
        smoke)  gate_smoke ;;
        health) gate_health ;;
    esac

    print_summary
    exit "${OVERALL_EXIT}"
}

main "$@"
