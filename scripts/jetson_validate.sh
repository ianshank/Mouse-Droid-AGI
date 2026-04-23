#!/bin/bash
# Phase A - automated on-Jetson validation harness.
#
# SSH into the Jetson and runs the three Phase-A gates in order:
#   1. scripts/verify_sensors.py --sensor all
#   2. pytest -m hardware -vv (uses /opt/mousedroid/venv)
#   3. bash scripts/jetson_smoke_test.sh
#
# Output from each step is captured to reports/jetson_validate/<timestamp>/.
# A non-zero overall exit code is returned if ANY gate fails, and the failing
# gate is reported in the summary.
#
# Usage:
#   bash scripts/jetson_validate.sh [jetson-host] [--step verify|pytest|smoke|all]
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
STEP="all"
HOST=""

TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${PROJECT_DIR}/reports/jetson_validate/${TS}"

declare -a GATE_RESULTS=()
OVERALL_EXIT=0

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log_section() { echo ""; echo "=== $1 === [$(ts)]"; }
log_step()    { echo "--- $1 ---"; }
die()         { echo "ERROR: $1" >&2; exit 1; }

remote_cmd() {
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "${REMOTE_USER}@${HOST}" "$@"
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
            --help|-h)
                sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
                exit 0
                ;;
            -*)
                die "Unknown option: $1"
                ;;
            *)
                if [[ -z "${HOST}" ]]; then
                    HOST="$1"
                else
                    die "Unexpected argument: $1"
                fi
                shift
                ;;
        esac
    done
    case "${STEP}" in
        all|verify|pytest|smoke) ;;
        *) die "Invalid --step '${STEP}'. Use: all|verify|pytest|smoke" ;;
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
    if ! remote_cmd "test -x ${REMOTE_VENV}/bin/python"; then
        die "Remote venv missing at ${REMOTE_VENV}/bin/python. Run scripts/deploy_remote.sh --full first."
    fi
    if ! remote_cmd "test -f ${REMOTE_SRC}/scripts/verify_sensors.py"; then
        die "Remote source missing at ${REMOTE_SRC}. Run scripts/deploy_remote.sh --code-only first."
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
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "${REMOTE_USER}@${HOST}" "${cmd}" 2>&1 | tee "${log}"
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
    run_gate "verify_sensors" \
        "cd ${REMOTE_SRC} && \
         MOUSEDROID_MOCK_HARDWARE=false \
         ${REMOTE_VENV}/bin/python scripts/verify_sensors.py --sensor all"
}

gate_pytest() {
    run_gate "pytest_hardware" \
        "cd ${REMOTE_SRC} && \
         MOUSEDROID_MOCK_HARDWARE=false \
         ${REMOTE_VENV}/bin/python -m pytest -m hardware -vv --maxfail=1"
}

gate_smoke() {
    run_gate "jetson_smoke_test" \
        "cd ${REMOTE_SRC} && \
         VENV_DIR=${REMOTE_VENV} \
         bash scripts/jetson_smoke_test.sh"
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

    mkdir -p "${REPORT_DIR}"

    resolve_host
    check_connectivity
    check_remote_layout

    case "${STEP}" in
        all)
            gate_verify
            gate_pytest
            gate_smoke
            ;;
        verify) gate_verify ;;
        pytest) gate_pytest ;;
        smoke)  gate_smoke ;;
    esac

    print_summary
    exit "${OVERALL_EXIT}"
}

main "$@"
