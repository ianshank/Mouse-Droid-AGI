#!/usr/bin/env bash
# Jetson GPU training launcher for MouseDroid pipeline orchestrator.
#
# Wrapper script that sets up the Jetson environment for training:
#   - Validates GPU availability via nvidia-smi / tegrastats
#   - Configures CUDA_VISIBLE_DEVICES and memory limits
#   - Supports --resume, --phase, and --dry-run flags
#   - All paths configurable via environment variables
#   - Structured JSON log output
#
# Environment variables (override via export):
#   MOUSEDROID_CONFIG   — YAML config path (default: config/jetson_production.yaml)
#   MOUSEDROID_VENV     — virtualenv path (default: /opt/mousedroid/venv)
#   CUDA_VISIBLE_DEVICES — GPU device index (default: 0)
#   MOUSEDROID_LOG_DIR  — log output directory (default: /var/log/mousedroid)
#   MOUSEDROID_MAX_GPU_MEM_MB — GPU memory limit in MB (default: 6144)
#
# Usage:
#   ./scripts/jetson_train.sh
#   ./scripts/jetson_train.sh --resume
#   ./scripts/jetson_train.sh --phase rssm
#   ./scripts/jetson_train.sh --dry-run
#   MOUSEDROID_CONFIG=config/local_training.yaml ./scripts/jetson_train.sh

set -euo pipefail

# --- Configuration (override via environment) ---
MOUSEDROID_CONFIG="${MOUSEDROID_CONFIG:-config/jetson_production.yaml}"
MOUSEDROID_VENV="${MOUSEDROID_VENV:-/opt/mousedroid/venv}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MOUSEDROID_LOG_DIR="${MOUSEDROID_LOG_DIR:-/var/log/mousedroid}"
MOUSEDROID_MAX_GPU_MEM_MB="${MOUSEDROID_MAX_GPU_MEM_MB:-6144}"

export CUDA_VISIBLE_DEVICES

# --- Logging helpers (structured JSON to stderr) ---
_ts() { date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date +"%Y-%m-%dT%H:%M:%SZ"; }

log_json() {
    local level="$1"; shift
    local event="$1"; shift
    # Build JSON safely.  Use jq if available to ensure proper escaping;
    # fall back to a printf-based approach that escapes double-quotes.
    if command -v jq &>/dev/null; then
        local obj
        obj=$(jq -nc --arg ts "$(_ts)" --arg level "$level" --arg event "$event" \
            '{ts: $ts, level: $level, event: $event}')
        for kv in "$@"; do
            local key="${kv%%=*}"
            local val="${kv#*=}"
            obj=$(echo "$obj" | jq -c --arg k "$key" --arg v "$val" '. + {($k): $v}')
        done
        echo "$obj" >&2
    else
        # Fallback: escape double-quotes in values to prevent broken JSON.
        local extra=""
        for kv in "$@"; do
            local key="${kv%%=*}"
            local val="${kv#*=}"
            val="${val//\"/\\\"}"
            extra="${extra}, \"${key}\": \"${val}\""
        done
        local escaped_event="${event//\"/\\\"}"
        echo "{\"ts\": \"$(_ts)\", \"level\": \"${level}\", \"event\": \"${escaped_event}\"${extra}}" >&2
    fi
}

log_info()  { log_json "info"  "$@"; }
log_warn()  { log_json "warn"  "$@"; }
log_error() { log_json "error" "$@"; }

# --- CLI argument parsing ---
RESUME_FLAG=""
PHASE_FLAG=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        --phase)
            PHASE_FLAG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--resume] [--phase PHASE] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --resume    Resume from last completed training phase"
            echo "  --phase     Run only the specified phase (rssm|warmstart|bdi|constitutional_rl)"
            echo "  --dry-run   Validate environment without starting training"
            echo ""
            echo "Environment variables:"
            echo "  MOUSEDROID_CONFIG          Config YAML path (default: config/jetson_production.yaml)"
            echo "  MOUSEDROID_VENV            Virtualenv path (default: /opt/mousedroid/venv)"
            echo "  CUDA_VISIBLE_DEVICES       GPU index (default: 0)"
            echo "  MOUSEDROID_LOG_DIR         Log directory (default: /var/log/mousedroid)"
            echo "  MOUSEDROID_MAX_GPU_MEM_MB  GPU memory limit in MB (default: 6144)"
            exit 0
            ;;
        *)
            log_error "unknown_argument" "arg=$1"
            exit 1
            ;;
    esac
done

# --- Validate config file ---
if [[ ! -f "${MOUSEDROID_CONFIG}" ]]; then
    log_error "config_not_found" "path=${MOUSEDROID_CONFIG}"
    exit 1
fi
log_info "config_loaded" "path=${MOUSEDROID_CONFIG}"

# --- Validate GPU availability ---
validate_gpu() {
    # Try nvidia-smi first (works on desktop + Jetson with nvidia-container).
    if command -v nvidia-smi &>/dev/null; then
        if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; then
            log_info "gpu_validated" "method=nvidia-smi"
            return 0
        fi
    fi

    # Fallback: check for Jetson tegrastats / /dev/nvhost-gpu.
    if [[ -e /dev/nvhost-gpu ]] || [[ -e /dev/nvgpu/igpu0 ]]; then
        log_info "gpu_validated" "method=tegra-device-node"
        return 0
    fi

    # Fallback: check if torch.cuda reports availability.
    if command -v python3 &>/dev/null; then
        if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            log_info "gpu_validated" "method=torch-cuda"
            return 0
        fi
    fi

    log_warn "gpu_not_found" "msg=No GPU detected; training will use CPU fallback"
    return 1
}

validate_gpu || true

# --- Set GPU memory limit (PyTorch) ---
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:${MOUSEDROID_MAX_GPU_MEM_MB}"
log_info "cuda_memory_configured" "max_split_size_mb=${MOUSEDROID_MAX_GPU_MEM_MB}"

# --- Ensure log directory ---
if [[ ! -d "${MOUSEDROID_LOG_DIR}" ]]; then
    mkdir -p "${MOUSEDROID_LOG_DIR}" 2>/dev/null || true
fi

# --- Activate virtualenv if present ---
if [[ -d "${MOUSEDROID_VENV}" ]] && [[ -f "${MOUSEDROID_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${MOUSEDROID_VENV}/bin/activate"
    log_info "venv_activated" "path=${MOUSEDROID_VENV}"
else
    log_info "venv_skipped" "path=${MOUSEDROID_VENV}" "msg=not found, using system Python"
fi

# --- Build command ---
CMD=(python3 -m mousedroid.training.pipeline_orchestrator --config "${MOUSEDROID_CONFIG}")

if [[ -n "${RESUME_FLAG}" ]]; then
    CMD+=("${RESUME_FLAG}")
fi

# If --phase is set, override the config phases via env var.
if [[ -n "${PHASE_FLAG}" ]]; then
    export MOUSEDROID_TRAINING_PIPELINE__PHASES="[\"${PHASE_FLAG}\"]"
    log_info "single_phase_mode" "phase=${PHASE_FLAG}"
fi

# --- Dry run ---
if [[ "${DRY_RUN}" == "true" ]]; then
    log_info "dry_run_complete" "command=${CMD[*]}"
    echo "Dry run complete. Would execute:"
    echo "  ${CMD[*]}"
    echo ""
    echo "Environment:"
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
    echo "  MOUSEDROID_CONFIG=${MOUSEDROID_CONFIG}"
    exit 0
fi

# --- Execute ---
log_info "pipeline_starting" "command=${CMD[*]}"

LOG_FILE="${MOUSEDROID_LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

if [[ -d "${MOUSEDROID_LOG_DIR}" ]] && [[ -w "${MOUSEDROID_LOG_DIR}" ]]; then
    "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}
    log_info "pipeline_finished" "exit_code=${EXIT_CODE}" "log_file=${LOG_FILE}"
else
    "${CMD[@]}"
    EXIT_CODE=$?
    log_info "pipeline_finished" "exit_code=${EXIT_CODE}"
fi

exit "${EXIT_CODE}"
