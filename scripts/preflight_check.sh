#!/usr/bin/env bash
# =============================================================================
# MouseDroidAGI — Pre-Flight Validation
# =============================================================================
# Checks that all required hardware, configs, and models are present before
# starting the MouseDroid container or service.
#
# Exit codes:
#   0 — all checks passed
#   1 — one or more critical checks failed
#
# Usage:
#   bash scripts/preflight_check.sh
#   bash scripts/preflight_check.sh --skip-models   # skip model weight checks
#   bash scripts/preflight_check.sh --skip-devices   # skip device file checks
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (overridable via environment)
# ---------------------------------------------------------------------------
ESP32_DEV="${MOUSEDROID_ESP32_DEV:-/dev/ttyUSB0}"
CAMERA_DEV="${MOUSEDROID_CAMERA_DEV:-/dev/video0}"
LIDAR_DEV="${MOUSEDROID_LIDAR_DEV:-/dev/ttyUSB1}"
AUDIO_DEV="/dev/snd"
GPIO_CHIPS=("/dev/gpiochip0" "/dev/gpiochip1")

CONFIG_FILE="${MOUSEDROID_CONFIG:-/etc/mousedroid/jetson_production.yaml}"
WEIGHTS_DIR="${MOUSEDROID_WEIGHTS_DIR:-/opt/mousedroid/weights}"
MODEL_DIR="${MOUSEDROID_MODEL_DIR:-/opt/mousedroid/models}"
INSTALL_DIR="${MOUSEDROID_INSTALL_DIR:-/opt/mousedroid}"

MIN_DISK_GB="${MOUSEDROID_MIN_DISK_GB:-8}"

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
SKIP_MODELS=false
SKIP_DEVICES=false

for arg in "$@"; do
    case "$arg" in
        --skip-models) SKIP_MODELS=true ;;
        --skip-devices) SKIP_DEVICES=true ;;
        --help|-h)
            echo "Usage: $0 [--skip-models] [--skip-devices]"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS=0
FAIL=0
WARN=0

ok()   { echo "  ✓ $1"; ((PASS++)); }
fail() { echo "  ✗ $1" >&2; ((FAIL++)); }
warn() { echo "  ⚠ $1"; ((WARN++)); }

check_device() {
    local dev="$1"
    local label="$2"
    if [ -e "$dev" ]; then
        ok "$label ($dev)"
    else
        fail "$label missing: $dev"
    fi
}

# ---------------------------------------------------------------------------
# 1. Hardware device checks
# ---------------------------------------------------------------------------
echo "=== Hardware Devices ==="

if [ "$SKIP_DEVICES" = true ]; then
    echo "  (skipped via --skip-devices)"
else
    check_device "$ESP32_DEV" "ESP32 UART"
    check_device "$CAMERA_DEV" "Camera"
    check_device "$LIDAR_DEV" "LiDAR UART"

    if [ -d "$AUDIO_DEV" ]; then
        ok "Audio (ALSA $AUDIO_DEV)"
    else
        warn "Audio device $AUDIO_DEV not found (mic/speaker will be unavailable)"
    fi

    for chip in "${GPIO_CHIPS[@]}"; do
        check_device "$chip" "GPIO chip"
    done
fi

# ---------------------------------------------------------------------------
# 2. Configuration file
# ---------------------------------------------------------------------------
echo ""
echo "=== Configuration ==="

if [ -f "$CONFIG_FILE" ]; then
    # Prefer the venv python so PyYAML is available even if system python lacks it.
    PY="${INSTALL_DIR}/venv/bin/python3"
    command -v "$PY" >/dev/null 2>&1 || PY="python3"

    if command -v "$PY" >/dev/null 2>&1; then
        if "$PY" -c "import yaml" >/dev/null 2>&1; then
            if "$PY" -c "import yaml, sys; yaml.safe_load(open(sys.argv[1]))" -- "$CONFIG_FILE" 2>/dev/null; then
                ok "Config YAML valid ($CONFIG_FILE)"
            else
                fail "Config YAML parse error: $CONFIG_FILE"
            fi
        else
            warn "PyYAML not installed — skipping YAML syntax check for $CONFIG_FILE"
        fi
    else
        ok "Config file exists ($CONFIG_FILE) — YAML validation skipped (no python3)"
    fi
else
    fail "Config file missing: $CONFIG_FILE"
fi

# ---------------------------------------------------------------------------
# 3. Model weights
# ---------------------------------------------------------------------------
echo ""
echo "=== Model Weights ==="

if [ "$SKIP_MODELS" = true ]; then
    echo "  (skipped via --skip-models)"
else
    if [ -d "$WEIGHTS_DIR" ]; then
        WEIGHT_COUNT=$(find "$WEIGHTS_DIR" -type f 2>/dev/null | wc -l)
        if [ "$WEIGHT_COUNT" -gt 0 ]; then
            ok "BDI/RSSM weights ($WEIGHT_COUNT files in $WEIGHTS_DIR)"
        else
            warn "Weights directory empty: $WEIGHTS_DIR (run scripts/download_weights.sh)"
        fi
    else
        warn "Weights directory missing: $WEIGHTS_DIR (run scripts/download_weights.sh)"
    fi

    # LLM model (optional but large)
    LLM_PATTERN="$MODEL_DIR/*.gguf"
    # shellcheck disable=SC2086
    if compgen -G $LLM_PATTERN >/dev/null 2>&1; then
        ok "LLM model found in $MODEL_DIR"
    else
        warn "LLM model not found ($LLM_PATTERN) — run scripts/download_model.sh"
    fi
fi

# ---------------------------------------------------------------------------
# 4. Disk space (check install dir partition, not just /)
# ---------------------------------------------------------------------------
echo ""
echo "=== System Resources ==="

AVAIL_KB=$(df -P "$INSTALL_DIR" 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
AVAIL_GB=$((AVAIL_KB / 1048576))

if [ "$AVAIL_GB" -ge "$MIN_DISK_GB" ]; then
    ok "Disk space: ${AVAIL_GB}GB available (min: ${MIN_DISK_GB}GB)"
else
    fail "Insufficient disk space: ${AVAIL_GB}GB < ${MIN_DISK_GB}GB required"
fi

# Swap check
SWAP_TOTAL_KB=$(grep SwapTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "0")
SWAP_TOTAL_MB=$((SWAP_TOTAL_KB / 1024))
if [ "$SWAP_TOTAL_MB" -ge 4000 ]; then
    ok "Swap: ${SWAP_TOTAL_MB}MB"
else
    warn "Swap only ${SWAP_TOTAL_MB}MB — recommend 4GB+ for Jetson"
fi

# ---------------------------------------------------------------------------
# 5. Docker / NVIDIA runtime (if applicable)
# ---------------------------------------------------------------------------
echo ""
echo "=== Docker & GPU ==="

if command -v docker &>/dev/null; then
    ok "Docker available"
    if docker info 2>/dev/null | grep -q "nvidia"; then
        ok "NVIDIA runtime detected"
    else
        warn "NVIDIA runtime not detected in 'docker info' — GPU passthrough may fail"
    fi
else
    warn "Docker not found (not required for bare-metal install)"
fi

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    ok "GPU: $GPU_NAME"
elif [ -f /etc/nv_tegra_release ]; then
    ok "Jetson Tegra platform detected"
else
    warn "No GPU detected (nvidia-smi not found, not a Tegra device)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Pre-Flight Summary ==="
echo "  Passed: $PASS | Failed: $FAIL | Warnings: $WARN"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "❌ Pre-flight FAILED — $FAIL critical check(s) did not pass."
    echo "   Fix the issues above before starting MouseDroid."
    exit 1
else
    echo ""
    echo "✅ Pre-flight PASSED — all critical checks OK."
    if [ "$WARN" -gt 0 ]; then
        echo "   ($WARN warning(s) — review above for optional improvements)"
    fi
    exit 0
fi
