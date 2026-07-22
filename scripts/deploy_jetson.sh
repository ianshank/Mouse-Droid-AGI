#!/bin/bash
# =============================================================================
# MouseDroid — Jetson Orin Nano Deployment
# =============================================================================
# Idempotent deployment script supporting both bare-metal (venv) and
# Docker container modes.
#
# Usage:
#   sudo bash scripts/deploy_jetson.sh [OPTIONS]
#
# Options:
#   --container     Deploy using Docker container (GPU PyTorch via L4T)
#   --service       Also install and enable the systemd service
#   --help          Show this help message
#
# Modes:
#   Bare-metal (default):
#     Installs into a Python venv at /opt/mousedroid/venv.
#     Uses CPU PyTorch (Jetson CUDA wheels require manual install).
#     Service: mousedroid.service
#
#   Container (--container):
#     Deploys via Docker compose with NVIDIA runtime.
#     Full GPU PyTorch support via L4T base image.
#     Service: mousedroid-docker.service
#
# Environment variables:
#   MOUSEDROID_INSTALL_DIR   Install directory (default: /opt/mousedroid)
#   MOUSEDROID_CONFIG_DIR    Config directory (default: /etc/mousedroid)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# Configurable paths
INSTALL_DIR="${MOUSEDROID_INSTALL_DIR:-/opt/mousedroid}"
CONFIG_DIR="${MOUSEDROID_CONFIG_DIR:-/etc/mousedroid}"
VENV_DIR="${INSTALL_DIR}/venv"

# Parse arguments
USE_CONTAINER=false
INSTALL_SERVICE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)  USE_CONTAINER=true; shift ;;
        --service)    INSTALL_SERVICE=true; shift ;;
        --help|-h)
            sed -n '2,/^# ====/{ /^# ====/d; s/^# \?//p }' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run with --help for usage information." >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Container deployment path
# ---------------------------------------------------------------------------
if [ "$USE_CONTAINER" = true ]; then
    echo "=== MouseDroid Jetson Deployment (Docker Container Mode) ==="

    # Delegate to docker_deploy.sh with appropriate flags
    DEPLOY_ARGS=()
    if [ "$INSTALL_SERVICE" = true ]; then
        DEPLOY_ARGS+=("--service")
    fi

    exec bash "${SCRIPT_DIR}/docker_deploy.sh" "${DEPLOY_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Bare-metal (venv) deployment path
# ---------------------------------------------------------------------------
echo "=== MouseDroid Jetson Deployment (Bare-Metal Mode) ==="
echo "  Install dir: ${INSTALL_DIR}"
echo "  Config dir:  ${CONFIG_DIR}"
echo "  Venv dir:    ${VENV_DIR}"
echo ""

# 1. System dependencies
echo "--- Installing system dependencies ---"
apt-get update -qq
apt-get install -y -qq python3-dev python3-venv libgpiod-dev v4l-utils i2c-tools

# 2. Create venv
if [ ! -d "${VENV_DIR}" ]; then
    echo "--- Creating Python venv ---"
    mkdir -p "${INSTALL_DIR}"
    python3 -m venv --system-site-packages "${VENV_DIR}"
fi

# 3. Swap check
echo "--- Checking swap ---"
SWAP_TOTAL=$(free -m | awk '/^Swap:/ {print $2}')
if [ "${SWAP_TOTAL}" -lt 4000 ]; then
    echo "    WARNING: Swap is ${SWAP_TOTAL}MB, recommended >= 4GB"
    echo "    Run scripts/jetson_system_setup.sh to configure swap"
fi

# 4. Install Jetson PyTorch
echo "--- Installing Jetson PyTorch ---"
export CUDA_HOME=/usr/local/cuda
bash "${SCRIPT_DIR}/install_jetson_pytorch.sh" "${VENV_DIR}"

# 5. Install project
echo "--- Installing mousedroid ---"
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet -e "${PROJECT_DIR}[hardware,jetson]"

# 6. Config files
echo "--- Deploying config ---"
mkdir -p "${CONFIG_DIR}"
cp -n "${PROJECT_DIR}/config/default.yaml" "${CONFIG_DIR}/" 2>/dev/null || true
cp -n "${PROJECT_DIR}/config/jetson_production.yaml" "${CONFIG_DIR}/" 2>/dev/null || true
cp -n "${PROJECT_DIR}/config/jetson_sdcard_64gb.yaml" "${CONFIG_DIR}/" 2>/dev/null || true

# 7. Systemd service
echo "--- Installing systemd service ---"
SERVICE_FILE="/etc/systemd/system/mousedroid.service"
cp "${SCRIPT_DIR}/mousedroid.service" "${SERVICE_FILE}"
systemctl daemon-reload

if [ "$INSTALL_SERVICE" = true ]; then
    systemctl enable mousedroid
    echo "  Service enabled (will start on boot)"
fi

# 8. Health check
echo "--- Running health check ---"
MOUSEDROID_MOCK_HARDWARE=false "${VENV_DIR}/bin/python" -m mousedroid.main --health-check || true

echo ""
echo "=== Deployment complete ==="
echo "  Enable with: sudo systemctl enable mousedroid"
echo "  Start with:  sudo systemctl start mousedroid"
echo ""
echo "  For GPU PyTorch support, redeploy with: sudo bash $0 --container"
