#!/bin/bash
# Remote deployment orchestrator for MouseDroid Jetson Nano.
# Runs from the local machine (WSL2/Linux/macOS) to deploy code to the Jetson.
#
# Usage: bash scripts/deploy_remote.sh [jetson-host] [--full|--code-only|--config-only]
#
# Host resolution order:
#   1. First positional argument
#   2. ~/.mousedroid/jetson_host file
#   3. mDNS/avahi discovery via jetson_discover.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
REMOTE_USER="${MOUSEDROID_REMOTE_USER:-jetson}"
REMOTE_SRC="/opt/mousedroid/src"
REMOTE_CONFIG="/etc/mousedroid"
REMOTE_WEIGHTS="/opt/mousedroid/weights"
REMOTE_MODELS="${MOUSEDROID_LLM_MODELS_DIR:-/home/jetson/models}"
DEPLOY_MODE="code-only"
HOST=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ts() {
    date "+%Y-%m-%d %H:%M:%S"
}

log_section() {
    echo ""
    echo "=== $1 === [$(ts)]"
}

log_step() {
    echo "--- $1 ---"
}

die() {
    echo "ERROR: $1" >&2
    exit 1
}

remote_cmd() {
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "${REMOTE_USER}@${HOST}" "$@"
}

remote_sudo() {
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "${REMOTE_USER}@${HOST}" sudo -- "$@"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --full)
                DEPLOY_MODE="full"
                shift
                ;;
            --code-only)
                DEPLOY_MODE="code-only"
                shift
                ;;
            --config-only)
                DEPLOY_MODE="config-only"
                shift
                ;;
            --weights)
                DEPLOY_MODE="weights"
                shift
                ;;
            --with-llm)
                DEPLOY_MODE="full-llm"
                shift
                ;;
            --help|-h)
                echo "Usage: bash scripts/deploy_remote.sh [jetson-host] [--full|--code-only|--config-only|--weights|--with-llm]"
                echo ""
                echo "Modes:"
                echo "  --full         Full setup: system + hardware + rsync + deploy + restart"
                echo "  --code-only    (default) Rsync code + pip reinstall + restart service"
                echo "  --config-only  Update config files + restart service"
                echo "  --weights      Sync trained weights to Jetson"
                echo "  --with-llm     Full deploy + weights + LLM model provisioning"
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
}

# ---------------------------------------------------------------------------
# Host resolution
# ---------------------------------------------------------------------------

resolve_host() {
    if [[ -n "${HOST}" ]]; then
        log_step "Using host from argument: ${HOST}"
        return
    fi

    local host_file="${HOME}/.mousedroid/jetson_host"
    if [[ -f "${host_file}" ]]; then
        HOST="$(cat "${host_file}" | tr -d '[:space:]')"
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

    die "Cannot determine Jetson host. Provide as argument or save to ~/.mousedroid/jetson_host"
}

# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------

check_connectivity() {
    log_step "Checking SSH connectivity to ${REMOTE_USER}@${HOST}..."
    if ! remote_cmd "echo ok" &>/dev/null; then
        die "Cannot SSH to ${REMOTE_USER}@${HOST}. Check connectivity and SSH keys."
    fi
    log_step "SSH connection verified"
}

# ---------------------------------------------------------------------------
# Rsync project code
# ---------------------------------------------------------------------------

rsync_code() {
    log_section "Syncing project code"
    log_step "Ensuring remote directory exists"
    remote_sudo bash -c "mkdir -p ${REMOTE_SRC} && chown -R ${REMOTE_USER}:${REMOTE_USER} ${REMOTE_SRC}"

    log_step "Rsyncing ${PROJECT_DIR} -> ${REMOTE_USER}@${HOST}:${REMOTE_SRC}/"
    rsync -avz --delete \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '.venv' \
        --exclude '*.egg-info' \
        --exclude 'node_modules' \
        --exclude '.mypy_cache' \
        --exclude '.pytest_cache' \
        --exclude '.ruff_cache' \
        --exclude '.claude' \
        "${PROJECT_DIR}/" "${REMOTE_USER}@${HOST}:${REMOTE_SRC}/"
    log_step "Rsync complete"
}

# ---------------------------------------------------------------------------
# Deploy config overlays
# ---------------------------------------------------------------------------

deploy_config() {
    log_section "Deploying configuration"
    remote_sudo mkdir -p "${REMOTE_CONFIG}"

    log_step "Copying config files to ${REMOTE_CONFIG}/"
    local config_dir="${PROJECT_DIR}/config"
    if [[ -d "${config_dir}" ]]; then
        for cfg_file in "${config_dir}"/*.yaml; do
            [[ -f "${cfg_file}" ]] || continue
            local basename
            basename="$(basename "${cfg_file}")"
            log_step "  -> ${basename}"
            scp -o ConnectTimeout=10 "${cfg_file}" "${REMOTE_USER}@${HOST}:/tmp/${basename}"
            remote_sudo cp -n "/tmp/${basename}" "${REMOTE_CONFIG}/${basename}"
            remote_cmd rm -f "/tmp/${basename}"
        done
    fi
    log_step "Config deployment complete"
}

# ---------------------------------------------------------------------------
# Remote system + hardware setup (full mode only)
# ---------------------------------------------------------------------------

run_system_setup() {
    log_section "Running Jetson system setup"
    if remote_cmd "test -x ${REMOTE_SRC}/scripts/jetson_system_setup.sh"; then
        remote_sudo bash "${REMOTE_SRC}/scripts/jetson_system_setup.sh"
    else
        log_step "SKIP: jetson_system_setup.sh not found on remote"
    fi
}

run_hardware_setup() {
    log_section "Running Jetson hardware setup"
    if remote_cmd "test -x ${REMOTE_SRC}/scripts/jetson_hardware_setup.sh" 2>/dev/null || \
       remote_cmd "test -f ${REMOTE_SRC}/scripts/jetson_hardware_setup.sh" 2>/dev/null; then
        remote_sudo bash "${REMOTE_SRC}/scripts/jetson_hardware_setup.sh"
    else
        log_step "SKIP: jetson_hardware_setup.sh not found on remote"
    fi
}

# ---------------------------------------------------------------------------
# Remote deploy (pip install + systemd)
# ---------------------------------------------------------------------------

run_deploy() {
    log_section "Running deploy_jetson.sh on remote"
    remote_sudo bash "${REMOTE_SRC}/scripts/deploy_jetson.sh"
}

# ---------------------------------------------------------------------------
# Pip reinstall (code-only mode)
# ---------------------------------------------------------------------------

pip_reinstall() {
    log_section "Reinstalling mousedroid package"
    local venv="/opt/mousedroid/venv"
    if remote_cmd "test -d ${venv}"; then
        remote_sudo "${venv}/bin/pip" install --quiet -e "${REMOTE_SRC}[hardware,jetson]"
    else
        log_step "Venv not found — running full deploy_jetson.sh"
        run_deploy
    fi
}

# ---------------------------------------------------------------------------
# Service restart
# ---------------------------------------------------------------------------

restart_service() {
    log_section "Restarting mousedroid service"
    remote_sudo systemctl daemon-reload

    if remote_sudo systemctl is-enabled mousedroid &>/dev/null; then
        remote_sudo systemctl restart mousedroid
        log_step "Service restarted"
        sleep 2
        local status
        status="$(remote_sudo systemctl is-active mousedroid 2>/dev/null || true)"
        log_step "Service status: ${status}"
    else
        log_step "Service not enabled — skipping restart"
    fi
}

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

run_health_check() {
    log_section "Running remote health check"
    local venv="/opt/mousedroid/venv"
    if remote_cmd "test -d ${venv}"; then
        remote_cmd "MOUSEDROID_MOCK_HARDWARE=false ${venv}/bin/python -m mousedroid.main --health-check" || {
            echo "WARNING: Health check returned non-zero exit code"
        }
    else
        log_step "SKIP: venv not found, cannot run health check"
    fi
}

# ---------------------------------------------------------------------------
# Sync trained weights
# ---------------------------------------------------------------------------

rsync_weights() {
    log_section "Syncing trained weights"
    local local_weights="${PROJECT_DIR}/weights"
    if [[ ! -d "${local_weights}" ]]; then
        die "Local weights directory not found: ${local_weights}"
    fi
    remote_sudo bash -c "mkdir -p ${REMOTE_WEIGHTS} && chown -R ${REMOTE_USER}:${REMOTE_USER} ${REMOTE_WEIGHTS}"

    log_step "Rsyncing weights -> ${REMOTE_USER}@${HOST}:${REMOTE_WEIGHTS}/"
    rsync -avz --delete \
        --include '*.npz' \
        --include '*.pt' \
        --include '*.json' \
        --include '*/' \
        --exclude '*' \
        "${local_weights}/" "${REMOTE_USER}@${HOST}:${REMOTE_WEIGHTS}/"
    log_step "Weights sync complete"
}

# ---------------------------------------------------------------------------
# LLM model provisioning
# ---------------------------------------------------------------------------

provision_llm() {
    log_section "Provisioning LLM model on Jetson"
    remote_cmd "mkdir -p ${REMOTE_MODELS}"

    # Check if model already exists
    local model_file="${REMOTE_MODELS}/Phi-3-mini-4k-instruct-q4.gguf"
    if remote_cmd "test -f ${model_file}" 2>/dev/null; then
        log_step "LLM model already present: ${model_file}"
        return 0
    fi

    log_step "Downloading Phi-3 Mini 4K GGUF to ${REMOTE_MODELS}/"
    remote_cmd "python3 -c \"\
from huggingface_hub import hf_hub_download; \
hf_hub_download( \
    'microsoft/Phi-3-mini-4k-instruct-gguf', \
    'Phi-3-mini-4k-instruct-q4.gguf', \
    local_dir='${REMOTE_MODELS}' \
)\"" || {
        error "LLM download failed. Install huggingface-hub on Jetson: pip install huggingface-hub"
        return 1
    }
    log_step "LLM model provisioned successfully"
}

# ---------------------------------------------------------------------------
# Deployment summary
# ---------------------------------------------------------------------------

print_summary() {
    log_section "Deployment Summary"
    echo "  Host:       ${REMOTE_USER}@${HOST}"
    echo "  Mode:       ${DEPLOY_MODE}"
    echo "  Source:      ${PROJECT_DIR}"
    echo "  Remote src:  ${REMOTE_SRC}"
    echo "  Remote cfg:  ${REMOTE_CONFIG}"
    echo "  Remote wts:  ${REMOTE_WEIGHTS}"
    echo "  Remote LLM:  ${REMOTE_MODELS}"
    echo "  Started:     ${DEPLOY_START}"
    echo "  Finished:    $(ts)"
    echo ""
    echo "=== Deployment complete ==="
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    parse_args "$@"

    log_section "MouseDroid Remote Deployment"
    DEPLOY_START="$(ts)"

    resolve_host
    check_connectivity

    case "${DEPLOY_MODE}" in
        full)
            rsync_code
            run_system_setup
            run_hardware_setup
            run_deploy
            deploy_config
            restart_service
            run_health_check
            ;;
        full-llm)
            rsync_code
            run_system_setup
            run_hardware_setup
            run_deploy
            deploy_config
            rsync_weights
            provision_llm
            restart_service
            run_health_check
            ;;
        code-only)
            rsync_code
            pip_reinstall
            deploy_config
            restart_service
            run_health_check
            ;;
        weights)
            rsync_weights
            ;;
        config-only)
            deploy_config
            restart_service
            run_health_check
            ;;
        *)
            die "Unknown deploy mode: ${DEPLOY_MODE}"
            ;;
    esac

    print_summary
}

main "$@"
