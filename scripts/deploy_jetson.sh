#!/bin/bash
# Idempotent Jetson Orin Nano deployment script.
# Usage: sudo bash scripts/deploy_jetson.sh
set -euo pipefail

INSTALL_DIR="/opt/mousedroid"
CONFIG_DIR="/etc/mousedroid"
VENV_DIR="${INSTALL_DIR}/venv"
SERVICE_FILE="/etc/systemd/system/mousedroid.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "=== MouseDroid Jetson Deployment ==="

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
if [ "${SWAP_TOTAL}" -lt 8000 ]; then
    echo "    WARNING: Swap is ${SWAP_TOTAL}MB, recommended >= 8GB"
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

# 7. Systemd service
echo "--- Installing systemd service ---"
cp "${SCRIPT_DIR}/mousedroid.service" "${SERVICE_FILE}"
systemctl daemon-reload

# 8. Health check
echo "--- Running health check ---"
MOUSEDROID_MOCK_HARDWARE=false "${VENV_DIR}/bin/python" -m mousedroid.main --health-check || true

echo "=== Deployment complete ==="
echo "Enable with: sudo systemctl enable mousedroid"
echo "Start with: sudo systemctl start mousedroid"
