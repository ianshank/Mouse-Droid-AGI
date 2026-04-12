#!/usr/bin/env bash
# =============================================================================
# MouseDroidAGI -- Jetson First Boot Orchestrator
# =============================================================================
# Idempotent first-boot script that calls existing setup scripts in order.
# Each step checks whether it has already been completed before running.
#
# Usage:
#   sudo bash scripts/jetson_first_boot.sh [OPTIONS]
#
# Options:
#   --help         Show usage and environment variables
#   --dry-run      Show what would be done without executing
#   --skip-model   Skip LLM model download (Step 5)
#
# Environment variables (override via /etc/mousedroid/first_boot.env):
#   MOUSEDROID_INSTALL_DIR   Install directory        (default: /opt/mousedroid)
#   MOUSEDROID_CONFIG_DIR    Config directory          (default: /etc/mousedroid)
#   MOUSEDROID_CONFIG        Production config path    (default: /etc/mousedroid/jetson_production.yaml)
#   COMPOSE_FILE             Docker Compose file       (default: ${INSTALL_DIR}/docker-compose.jetson.yml)
#   MOUSEDROID_CONTAINER     Container name            (default: mousedroid)
#   MODEL_URL                LLM model download URL    (default: HuggingFace Llama-3 Q4_K_M)
#   MODEL_PATH               LLM model file path       (default: ${INSTALL_DIR}/models/llama-3-8b-instruct.Q4_K_M.gguf)
#   MODEL_CHECKSUM           Expected SHA-256 checksum  (default: empty = skip verify)
#   MOUSEDROID_TELEMETRY_PORT Telemetry server port    (default: 8080)
#   MOUSEDROID_HOSTNAME      Target hostname           (default: mousedroid)
#   CUDA_VISIBLE_DEVICES     GPU device selection      (default: 0)
#   MOUSEDROID_MAX_GPU_MEM_MB Max GPU memory in MB     (default: 6144)
#
# Exit codes:
#   0  All steps completed successfully
#   1  A critical step failed
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Constants and Defaults
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# Load env file if present
ENV_FILE="${MOUSEDROID_ENV_FILE:-/etc/mousedroid/first_boot.env}"
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    set -a
    source "${ENV_FILE}"
    set +a
fi

# All paths from env vars with documented defaults
INSTALL_DIR="${MOUSEDROID_INSTALL_DIR:-/opt/mousedroid}"
CONFIG_DIR="${MOUSEDROID_CONFIG_DIR:-/etc/mousedroid}"
MOUSEDROID_CONFIG="${MOUSEDROID_CONFIG:-${CONFIG_DIR}/jetson_production.yaml}"
COMPOSE_FILE="${COMPOSE_FILE:-${INSTALL_DIR}/docker-compose.jetson.yml}"
CONTAINER_NAME="${MOUSEDROID_CONTAINER:-mousedroid}"
MODEL_PATH="${MODEL_PATH:-${INSTALL_DIR}/models/llama-3-8b-instruct.Q4_K_M.gguf}"
MODEL_CHECKSUM="${MODEL_CHECKSUM:-}"
TELEMETRY_PORT="${MOUSEDROID_TELEMETRY_PORT:-8080}"
SERVICE_FILE="${MOUSEDROID_SERVICE_FILE:-${SCRIPT_DIR}/mousedroid-docker.service}"
HEALTH_TIMEOUT="${MOUSEDROID_HEALTH_TIMEOUT:-30}"

# Flags
DRY_RUN=false
SKIP_MODEL=false

# Tracking
STEP_RESULTS=()
STEP_COUNT=0
STEP_PASS=0
STEP_FAIL=0
STEP_SKIP=0
START_TIME=""

# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

show_help() {
    cat <<'USAGE'
MouseDroidAGI -- Jetson First Boot Orchestrator

Usage:
  sudo bash scripts/jetson_first_boot.sh [OPTIONS]

Options:
  --help         Show this help message and exit
  --dry-run      Show what would be done without executing
  --skip-model   Skip LLM model download (Step 5)

Environment Variables:
  MOUSEDROID_INSTALL_DIR      Install directory           [/opt/mousedroid]
  MOUSEDROID_CONFIG_DIR       Config directory             [/etc/mousedroid]
  MOUSEDROID_CONFIG           Production config path       [/etc/mousedroid/jetson_production.yaml]
  COMPOSE_FILE                Docker Compose file          [${INSTALL_DIR}/docker-compose.jetson.yml]
  MOUSEDROID_CONTAINER        Container name               [mousedroid]
  MODEL_URL                   LLM model download URL       [HuggingFace Llama-3 Q4_K_M]
  MODEL_PATH                  LLM model destination path   [${INSTALL_DIR}/models/...]
  MODEL_CHECKSUM              Expected SHA-256 checksum    [empty = skip]
  MOUSEDROID_TELEMETRY_PORT   Telemetry server port        [8080]
  MOUSEDROID_HOSTNAME         Target hostname              [mousedroid]
  CUDA_VISIBLE_DEVICES        GPU device selection         [0]
  MOUSEDROID_MAX_GPU_MEM_MB   Max GPU memory in MB         [6144]
  MOUSEDROID_ENV_FILE         Path to env override file    [/etc/mousedroid/first_boot.env]
  MOUSEDROID_SERVICE_FILE     Systemd service source       [scripts/mousedroid-docker.service]
  MOUSEDROID_HEALTH_TIMEOUT   Health check timeout (sec)   [30]

Environment File:
  Copy config/jetson_first_boot.env to /etc/mousedroid/first_boot.env
  and customize values before running this script.

Steps:
  1. Bootstrap (SSH, users, groups)
  2. System setup (swap, kernel, power)
  3. Hardware setup (GPIO, UART, CSI, udev)
  4. Pull Docker image
  5. Download LLM model (skippable with --skip-model)
  6. Install systemd service
  7. Start service and verify health
  8. Run smoke test
  9. Print structured JSON summary
USAGE
}

for arg in "$@"; do
    case "${arg}" in
        --help|-h)
            show_help
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --skip-model)
            SKIP_MODEL=true
            ;;
        *)
            echo "Unknown option: ${arg}"
            echo "Run with --help for usage information."
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging Helpers
# ---------------------------------------------------------------------------

_ts() {
    date -u "+%Y-%m-%dT%H:%M:%SZ"
}

log_json() {
    local level="$1"
    shift
    local message="$*"
    local ts
    ts="$(_ts)"
    if command -v jq &>/dev/null; then
        jq -nc --arg ts "${ts}" --arg level "${level}" --arg msg "${message}" \
            '{"timestamp": $ts, "level": $level, "message": $msg}'
    else
        # Fallback: manually escape quotes for valid JSON
        local escaped_msg
        escaped_msg="${message//\\/\\\\}"
        escaped_msg="${escaped_msg//\"/\\\"}"
        echo "{\"timestamp\":\"${ts}\",\"level\":\"${level}\",\"message\":\"${escaped_msg}\"}"
    fi
}

info()  { log_json "INFO" "$*"; }
warn()  { log_json "WARN" "$*"; }
error() { log_json "ERROR" "$*" >&2; }

record_result() {
    local step_num="$1"
    local step_name="$2"
    local status="$3"
    local detail="${4:-}"
    STEP_RESULTS+=("${step_num}|${step_name}|${status}|${detail}")
    case "${status}" in
        pass) STEP_PASS=$((STEP_PASS + 1)) ;;
        fail) STEP_FAIL=$((STEP_FAIL + 1)) ;;
        skip) STEP_SKIP=$((STEP_SKIP + 1)) ;;
    esac
}

# ---------------------------------------------------------------------------
# Idempotency Markers
# ---------------------------------------------------------------------------

MARKER_DIR="${CONFIG_DIR}/.first_boot_markers"

marker_exists() {
    local name="$1"
    [[ -f "${MARKER_DIR}/${name}.done" ]]
}

set_marker() {
    local name="$1"
    mkdir -p "${MARKER_DIR}"
    _ts > "${MARKER_DIR}/${name}.done"
}

# ---------------------------------------------------------------------------
# Step Functions
# ---------------------------------------------------------------------------

step_1_bootstrap() {
    local step_name="bootstrap"
    info "Step 1: Bootstrap (SSH, users, groups)"

    if marker_exists "${step_name}"; then
        info "Step 1 already completed (marker found). Skipping."
        record_result 1 "Bootstrap" "skip" "already completed"
        return 0
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "[DRY-RUN] Would run: bash ${SCRIPT_DIR}/jetson_bootstrap.sh"
        record_result 1 "Bootstrap" "skip" "dry-run"
        return 0
    fi

    if [[ ! -x "${SCRIPT_DIR}/jetson_bootstrap.sh" ]]; then
        error "jetson_bootstrap.sh not found or not executable at ${SCRIPT_DIR}"
        record_result 1 "Bootstrap" "fail" "script not found"
        return 1
    fi

    if bash "${SCRIPT_DIR}/jetson_bootstrap.sh"; then
        set_marker "${step_name}"
        record_result 1 "Bootstrap" "pass" ""
    else
        error "Bootstrap script failed"
        record_result 1 "Bootstrap" "fail" "script returned non-zero"
        return 1
    fi
}

step_2_system_setup() {
    local step_name="system_setup"
    info "Step 2: System setup (swap, kernel, power)"

    if marker_exists "${step_name}"; then
        info "Step 2 already completed (marker found). Skipping."
        record_result 2 "System setup" "skip" "already completed"
        return 0
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "[DRY-RUN] Would run: bash ${SCRIPT_DIR}/jetson_system_setup.sh"
        record_result 2 "System setup" "skip" "dry-run"
        return 0
    fi

    if [[ ! -x "${SCRIPT_DIR}/jetson_system_setup.sh" ]]; then
        error "jetson_system_setup.sh not found or not executable at ${SCRIPT_DIR}"
        record_result 2 "System setup" "fail" "script not found"
        return 1
    fi

    if bash "${SCRIPT_DIR}/jetson_system_setup.sh"; then
        set_marker "${step_name}"
        record_result 2 "System setup" "pass" ""
    else
        error "System setup script failed"
        record_result 2 "System setup" "fail" "script returned non-zero"
        return 1
    fi
}

step_3_hardware_setup() {
    local step_name="hardware_setup"
    info "Step 3: Hardware setup (GPIO, UART, CSI, udev)"

    if marker_exists "${step_name}"; then
        info "Step 3 already completed (marker found). Skipping."
        record_result 3 "Hardware setup" "skip" "already completed"
        return 0
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "[DRY-RUN] Would run: bash ${SCRIPT_DIR}/jetson_hardware_setup.sh"
        record_result 3 "Hardware setup" "skip" "dry-run"
        return 0
    fi

    if [[ ! -x "${SCRIPT_DIR}/jetson_hardware_setup.sh" ]]; then
        error "jetson_hardware_setup.sh not found or not executable at ${SCRIPT_DIR}"
        record_result 3 "Hardware setup" "fail" "script not found"
        return 1
    fi

    if bash "${SCRIPT_DIR}/jetson_hardware_setup.sh"; then
        set_marker "${step_name}"
        record_result 3 "Hardware setup" "pass" ""
    else
        error "Hardware setup script failed"
        record_result 3 "Hardware setup" "fail" "script returned non-zero"
        return 1
    fi
}

step_4_docker_pull() {
    local step_name="docker_pull"
    info "Step 4: Pull Docker image"

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "[DRY-RUN] Would run: docker compose -f ${COMPOSE_FILE} pull"
        record_result 4 "Docker pull" "skip" "dry-run"
        return 0
    fi

    if ! command -v docker &>/dev/null; then
        error "Docker is not installed"
        record_result 4 "Docker pull" "fail" "docker not found"
        return 1
    fi

    if [[ ! -f "${COMPOSE_FILE}" ]]; then
        error "Compose file not found: ${COMPOSE_FILE}"
        record_result 4 "Docker pull" "fail" "compose file missing"
        return 1
    fi

    info "Pulling Docker image via compose..."
    if docker compose -f "${COMPOSE_FILE}" pull 2>&1; then
        record_result 4 "Docker pull" "pass" ""
    else
        error "Docker pull failed"
        record_result 4 "Docker pull" "fail" "docker compose pull returned non-zero"
        return 1
    fi
}

step_5_download_model() {
    local step_name="model_download"
    info "Step 5: Download LLM model"

    if [[ "${SKIP_MODEL}" == "true" ]]; then
        info "Step 5 skipped (--skip-model flag)."
        record_result 5 "Model download" "skip" "flag --skip-model"
        return 0
    fi

    if [[ -f "${MODEL_PATH}" ]]; then
        info "Model already exists at ${MODEL_PATH}. Skipping download."
        record_result 5 "Model download" "skip" "model file exists"
        return 0
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "[DRY-RUN] Would run: bash ${SCRIPT_DIR}/download_model.sh"
        record_result 5 "Model download" "skip" "dry-run"
        return 0
    fi

    if [[ ! -x "${SCRIPT_DIR}/download_model.sh" ]]; then
        error "download_model.sh not found or not executable at ${SCRIPT_DIR}"
        record_result 5 "Model download" "fail" "script not found"
        return 1
    fi

    info "Downloading model to ${MODEL_PATH}..."
    if MODEL_PATH="${MODEL_PATH}" MODEL_CHECKSUM="${MODEL_CHECKSUM}" \
        bash "${SCRIPT_DIR}/download_model.sh"; then
        record_result 5 "Model download" "pass" ""
    else
        error "Model download failed"
        record_result 5 "Model download" "fail" "download script returned non-zero"
        return 1
    fi
}

step_6_install_service() {
    local step_name="systemd_service"
    info "Step 6: Install systemd service"

    local target_service="/etc/systemd/system/mousedroid-docker.service"

    if [[ -f "${target_service}" ]]; then
        info "Systemd service already installed. Skipping."
        record_result 6 "Systemd service" "skip" "already installed"
        return 0
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "[DRY-RUN] Would copy ${SERVICE_FILE} to ${target_service} and enable"
        record_result 6 "Systemd service" "skip" "dry-run"
        return 0
    fi

    if [[ ! -f "${SERVICE_FILE}" ]]; then
        error "Service file not found: ${SERVICE_FILE}"
        record_result 6 "Systemd service" "fail" "service file missing"
        return 1
    fi

    cp "${SERVICE_FILE}" "${target_service}"
    systemctl daemon-reload
    systemctl enable mousedroid-docker.service
    info "Systemd service installed and enabled."
    record_result 6 "Systemd service" "pass" ""
}

step_7_start_and_verify() {
    local step_name="start_service"
    info "Step 7: Start service and verify health"

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "[DRY-RUN] Would start mousedroid-docker.service and verify health"
        record_result 7 "Start & health" "skip" "dry-run"
        return 0
    fi

    # Start via systemd if installed, otherwise docker compose
    if systemctl is-enabled mousedroid-docker.service &>/dev/null; then
        info "Starting via systemd..."
        systemctl start mousedroid-docker.service
    else
        info "Starting via docker compose..."
        if [[ ! -f "${COMPOSE_FILE}" ]]; then
            error "Compose file not found: ${COMPOSE_FILE}"
            record_result 7 "Start & health" "fail" "compose file missing"
            return 1
        fi
        docker compose -f "${COMPOSE_FILE}" up -d
    fi

    # Wait for container to become healthy
    info "Waiting up to ${HEALTH_TIMEOUT}s for container health..."
    local elapsed=0
    local interval=5
    while [[ "${elapsed}" -lt "${HEALTH_TIMEOUT}" ]]; do
        if docker ps --filter "name=${CONTAINER_NAME}" --filter "status=running" \
            --format '{{.Names}}' 2>/dev/null | grep -q "${CONTAINER_NAME}"; then
            info "Container ${CONTAINER_NAME} is running."
            record_result 7 "Start & health" "pass" "container running after ${elapsed}s"
            return 0
        fi
        sleep "${interval}"
        elapsed=$((elapsed + interval))
    done

    error "Container ${CONTAINER_NAME} not running after ${HEALTH_TIMEOUT}s"
    record_result 7 "Start & health" "fail" "timeout waiting for container"
    return 1
}

step_8_smoke_test() {
    local step_name="smoke_test"
    info "Step 8: Run smoke test"

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "[DRY-RUN] Would run: bash ${SCRIPT_DIR}/jetson_smoke_test.sh"
        record_result 8 "Smoke test" "skip" "dry-run"
        return 0
    fi

    if [[ ! -x "${SCRIPT_DIR}/jetson_smoke_test.sh" ]]; then
        error "jetson_smoke_test.sh not found or not executable at ${SCRIPT_DIR}"
        record_result 8 "Smoke test" "fail" "script not found"
        return 1
    fi

    info "Running smoke tests..."
    if bash "${SCRIPT_DIR}/jetson_smoke_test.sh"; then
        record_result 8 "Smoke test" "pass" ""
    else
        warn "Smoke test reported failures (non-fatal)"
        record_result 8 "Smoke test" "fail" "smoke test returned non-zero"
        # Non-fatal: don't return 1 here; the summary will show failures
    fi
}

step_9_summary() {
    info "Step 9: First boot summary"

    local end_time
    end_time="$(_ts)"
    local overall_status="success"
    if [[ "${STEP_FAIL}" -gt 0 ]]; then
        overall_status="failure"
    fi

    # Build JSON summary
    local steps_json="["
    local first=true
    for result in "${STEP_RESULTS[@]}"; do
        IFS='|' read -r num name status detail <<< "${result}"
        if [[ "${first}" != "true" ]]; then
            steps_json+=","
        fi
        first=false
        if command -v jq &>/dev/null; then
            steps_json+="$(jq -nc \
                --argjson num "${num}" \
                --arg name "${name}" \
                --arg status "${status}" \
                --arg detail "${detail}" \
                '{"step": $num, "name": $name, "status": $status, "detail": $detail}')"
        else
            local esc_detail="${detail//\\/\\\\}"
            esc_detail="${esc_detail//\"/\\\"}"
            local esc_name="${name//\\/\\\\}"
            esc_name="${esc_name//\"/\\\"}"
            steps_json+="{\"step\":${num},\"name\":\"${esc_name}\",\"status\":\"${status}\",\"detail\":\"${esc_detail}\"}"
        fi
    done
    steps_json+="]"

    if command -v jq &>/dev/null; then
        jq -nc \
            --arg status "${overall_status}" \
            --arg start "${START_TIME}" \
            --arg end "${end_time}" \
            --argjson passed "${STEP_PASS}" \
            --argjson failed "${STEP_FAIL}" \
            --argjson skipped "${STEP_SKIP}" \
            --argjson steps "${steps_json}" \
            --arg install_dir "${INSTALL_DIR}" \
            --arg config_dir "${CONFIG_DIR}" \
            --arg compose_file "${COMPOSE_FILE}" \
            --arg container "${CONTAINER_NAME}" \
            '{
                "event": "first_boot_complete",
                "status": $status,
                "started_at": $start,
                "finished_at": $end,
                "summary": {
                    "passed": $passed,
                    "failed": $failed,
                    "skipped": $skipped
                },
                "config": {
                    "install_dir": $install_dir,
                    "config_dir": $config_dir,
                    "compose_file": $compose_file,
                    "container": $container
                },
                "steps": $steps
            }'
    else
        # Fallback without jq
        cat <<ENDJSON
{"event":"first_boot_complete","status":"${overall_status}","started_at":"${START_TIME}","finished_at":"${end_time}","summary":{"passed":${STEP_PASS},"failed":${STEP_FAIL},"skipped":${STEP_SKIP}},"config":{"install_dir":"${INSTALL_DIR}","config_dir":"${CONFIG_DIR}","compose_file":"${COMPOSE_FILE}","container":"${CONTAINER_NAME}"},"steps":${steps_json}}
ENDJSON
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    START_TIME="$(_ts)"
    info "=== MouseDroidAGI First Boot Orchestrator ==="
    info "Install dir: ${INSTALL_DIR}"
    info "Config dir:  ${CONFIG_DIR}"
    info "Compose:     ${COMPOSE_FILE}"
    info "Container:   ${CONTAINER_NAME}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "*** DRY-RUN MODE -- no changes will be made ***"
    fi

    # Run all steps; continue on non-critical failures
    step_1_bootstrap || true
    step_2_system_setup || true
    step_3_hardware_setup || true
    step_4_docker_pull || true
    step_5_download_model || true
    step_6_install_service || true
    step_7_start_and_verify || true
    step_8_smoke_test || true
    step_9_summary

    if [[ "${STEP_FAIL}" -gt 0 ]]; then
        error "${STEP_FAIL} step(s) failed. Review output above."
        exit 1
    fi

    info "=== First boot completed successfully ==="
    exit 0
}

main "$@"
