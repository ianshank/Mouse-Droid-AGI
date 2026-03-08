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
apt-get install -y -qq python3-dev python3-venv libgpiod-dev

# 2. Create venv
if [ ! -d "${VENV_DIR}" ]; then
    echo "--- Creating Python venv ---"
    mkdir -p "${INSTALL_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

# 3. Install project
echo "--- Installing mousedroid ---"
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet -e "${PROJECT_DIR}[hardware]"

# 4. Config files
echo "--- Deploying config ---"
mkdir -p "${CONFIG_DIR}"
cp -n "${PROJECT_DIR}/config/default.yaml" "${CONFIG_DIR}/" 2>/dev/null || true
cp -n "${PROJECT_DIR}/config/jetson_production.yaml" "${CONFIG_DIR}/" 2>/dev/null || true

# 5. Systemd service
echo "--- Installing systemd service ---"
cp "${SCRIPT_DIR}/mousedroid.service" "${SERVICE_FILE}"
systemctl daemon-reload

# 6. Health check
echo "--- Running health check ---"
MOUSEDROID_MOCK_HARDWARE=false "${VENV_DIR}/bin/python" -m mousedroid.main --health-check || true

echo "=== Deployment complete ==="
echo "Enable with: sudo systemctl enable mousedroid"
echo "Start with: sudo systemctl start mousedroid"
