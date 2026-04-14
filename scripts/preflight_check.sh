#!/usr/bin/env bash
# =============================================================================
# MouseDroidAGI — Pre-flight Validation Script
# =============================================================================
# Checks that all required hardware devices, runtime dependencies, and
# configuration files are present before starting the MouseDroid service.
#
# Exit codes:
#   0 — all checks passed
#   1 — one or more checks failed (see stderr output)
#
# Usage:
#   scripts/preflight_check.sh                    # Use default paths
#   MOUSEDROID_ESP32_DEV=/dev/ttyACM0 scripts/preflight_check.sh
#
# Designed to be called from:
#   ExecStartPre= in mousedroid-docker.service
# =============================================================================
set -euo pipefail

# --- Configurable device paths (env vars with defaults) ---
ESP32_DEV="${MOUSEDROID_ESP32_DEV:-/dev/ttyUSB0}"
CAMERA_DEV="${MOUSEDROID_CAMERA_DEV:-/dev/video0}"
LIDAR_DEV="${MOUSEDROID_LIDAR_DEV:-/dev/ttyUSB1}"
CONFIG_PATH="${MOUSEDROID_CONFIG:-/etc/mousedroid/jetson_production.yaml}"
INSTALL_DIR="${MOUSEDROID_INSTALL_DIR:-/opt/mousedroid}"
MIN_DISK_GB="${MOUSEDROID_MIN_DISK_GB:-10}"

# Colour helpers (disabled if stdout is not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' NC=''
fi

PASS=0
FAIL=0
WARN=0

pass() { PASS=$((PASS + 1)); printf "${GREEN}[PASS]${NC} %s\n" "$1"; }
fail() { FAIL=$((FAIL + 1)); printf "${RED}[FAIL]${NC} %s\n" "$1" >&2; }
warn() { WARN=$((WARN + 1)); printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }

# ==========================================================================
# 1. Required device files
# ==========================================================================
echo "--- Hardware Devices ---"

if [ -c "$ESP32_DEV" ]; then
    pass "ESP32 serial: $ESP32_DEV"
else
    fail "ESP32 serial not found: $ESP32_DEV (required for motor control)"
fi

if [ -c "$CAMERA_DEV" ]; then
    pass "Camera: $CAMERA_DEV"
else
    fail "Camera device not found: $CAMERA_DEV (required for vision)"
fi

# GPIO — required for ultrasonic trigger/echo
for gpio in /dev/gpiochip0 /dev/gpiochip1; do
    if [ -c "$gpio" ]; then
        pass "GPIO: $gpio"
    else
        fail "GPIO not found: $gpio (required for ultrasonic sensor)"
    fi
done

# Optional devices — warn if missing
if [ -c "$LIDAR_DEV" ]; then
    pass "LiDAR serial: $LIDAR_DEV"
else
    warn "LiDAR serial not found: $LIDAR_DEV (optional — system runs without LiDAR)"
fi

if [ -d "/dev/snd" ]; then
    pass "Audio devices: /dev/snd"
else
    warn "Audio devices not found: /dev/snd (optional — Rocky voice disabled)"
fi

# ==========================================================================
# 2. Docker / NVIDIA runtime
# ==========================================================================
echo ""
echo "--- Runtime Environment ---"

if command -v docker >/dev/null 2>&1; then
    pass "Docker installed"
    if docker info 2>/dev/null | grep -q "nvidia"; then
        pass "NVIDIA container runtime available"
    else
        warn "NVIDIA container runtime not detected in 'docker info' (may be configured at compose level)"
    fi
else
    warn "Docker not installed (only required for Docker deployment mode)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    pass "nvidia-smi available"
else
    warn "nvidia-smi not found (expected on Jetson with JetPack)"
fi

# ==========================================================================
# 3. Disk space
# ==========================================================================
echo ""
echo "--- Disk Space ---"

# Check available disk space on the experience DB partition
avail_kb=$(df -P /home 2>/dev/null | awk 'NR==2 {print $4}' || echo 0)
avail_gb=$((avail_kb / 1048576))
if [ "$avail_gb" -ge "$MIN_DISK_GB" ]; then
    pass "Disk space: ${avail_gb}GB available (min ${MIN_DISK_GB}GB)"
else
    fail "Insufficient disk space: ${avail_gb}GB available (need ${MIN_DISK_GB}GB for experience DB)"
fi

# ==========================================================================
# 4. Configuration files
# ==========================================================================
echo ""
echo "--- Configuration ---"

if [ -f "$CONFIG_PATH" ]; then
    pass "Config file: $CONFIG_PATH"
    # Attempt to validate YAML syntax if python is available
    if command -v python3 >/dev/null 2>&1; then
        if python3 -c "import yaml; yaml.safe_load(open('$CONFIG_PATH'))" 2>/dev/null; then
            pass "Config YAML syntax valid"
        else
            fail "Config YAML syntax error: $CONFIG_PATH"
        fi
    fi
else
    fail "Config file not found: $CONFIG_PATH"
fi

# ==========================================================================
# 5. Model weights (optional)
# ==========================================================================
echo ""
echo "--- Model Weights ---"

MODEL_DIR="${INSTALL_DIR}/models"
if [ -d "$MODEL_DIR" ]; then
    # Check for BDI model weights
    if ls "$MODEL_DIR"/bdi*.pt "$MODEL_DIR"/bdi*.pth 2>/dev/null | head -1 >/dev/null 2>&1; then
        pass "BDI model weights found"
    else
        warn "BDI model weights not found in $MODEL_DIR (will use random init)"
    fi

    # Check for LLM model weights
    if ls "$MODEL_DIR"/llm*.gguf "$MODEL_DIR"/*.gguf 2>/dev/null | head -1 >/dev/null 2>&1; then
        pass "LLM model weights found"
    else
        warn "LLM model weights not found in $MODEL_DIR (NL commands will use rule-based parser only)"
    fi
else
    warn "Model directory not found: $MODEL_DIR"
fi

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "--- Summary ---"
printf "  Passed: %d  |  Failed: %d  |  Warnings: %d\n" "$PASS" "$FAIL" "$WARN"

if [ "$FAIL" -gt 0 ]; then
    printf "\n${RED}Pre-flight check FAILED${NC} — fix %d issue(s) before starting.\n" "$FAIL" >&2
    exit 1
fi

if [ "$WARN" -gt 0 ]; then
    printf "\n${YELLOW}Pre-flight check PASSED with warnings${NC} — some optional hardware not detected.\n"
fi

printf "\n${GREEN}Pre-flight check PASSED${NC}\n"
exit 0
