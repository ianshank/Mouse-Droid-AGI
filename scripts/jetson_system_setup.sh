#!/bin/bash
# Idempotent Jetson system setup script (JetPack 6 / Orin Nano).
# Validates JetPack stack, configures swap/power, creates directories.
# Usage: sudo bash scripts/jetson_system_setup.sh
set -euo pipefail

INSTALL_DIR="/opt/mousedroid"
CONFIG_DIR="/etc/mousedroid"
EXPERIENCE_DIR="/home/jetson/mousedroid_experience"
SWAPFILE="/swapfile"
SWAP_SIZE="${SWAP_SIZE:-4G}"
SWAP_SIZE_NUM="${SWAP_SIZE%G}"
SWAP_SIZE_BYTES=$((SWAP_SIZE_NUM * 1024 * 1024 * 1024))
JETSON_USER="jetson"

echo "=== MouseDroid Jetson System Setup ==="

# ---- JetPack Validation ----

echo "--- Validating JetPack installation ---"
if dpkg -l nvidia-jetpack 2>/dev/null | grep -q '^ii'; then
    JETPACK_VER=$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null)
    echo "    JetPack version: ${JETPACK_VER}"
else
    echo "    JetPack not found, installing nvidia-jetpack..."
    apt-get update -qq
    apt-get install -y -qq nvidia-jetpack
    JETPACK_VER=$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null)
    echo "    JetPack version: ${JETPACK_VER}"
fi

echo "--- Validating CUDA ---"
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "    CUDA version: ${CUDA_VER}"
    if [[ ! "${CUDA_VER}" =~ ^12\. ]]; then
        echo "    WARNING: Expected CUDA 12.x, found ${CUDA_VER}"
    fi
else
    echo "    WARNING: nvcc not found. CUDA toolkit may not be installed."
    echo "    Attempting install via nvidia-jetpack..."
    apt-get update -qq
    apt-get install -y -qq nvidia-jetpack
fi

echo "--- Validating cuDNN ---"
if dpkg -l libcudnn* 2>/dev/null | grep -q '^ii'; then
    CUDNN_VER=$(dpkg-query -W -f='${Version}' libcudnn8 2>/dev/null || echo "unknown")
    echo "    cuDNN version: ${CUDNN_VER}"
else
    echo "    WARNING: cuDNN not found. Should be installed with nvidia-jetpack."
fi

echo "--- Validating TensorRT ---"
if dpkg -l tensorrt 2>/dev/null | grep -q '^ii'; then
    TRT_VER=$(dpkg-query -W -f='${Version}' tensorrt 2>/dev/null)
    echo "    TensorRT version: ${TRT_VER}"
    TRT_MAJOR=$(echo "${TRT_VER}" | cut -d'.' -f1)
    TRT_MINOR=$(echo "${TRT_VER}" | cut -d'.' -f2)
    if [ "${TRT_MAJOR}" -lt 8 ] || { [ "${TRT_MAJOR}" -eq 8 ] && [ "${TRT_MINOR}" -lt 6 ]; }; then
        echo "    WARNING: Expected TensorRT 8.6+, found ${TRT_VER}"
    fi
else
    echo "    WARNING: TensorRT not found. Should be installed with nvidia-jetpack."
fi

# ---- Swap Configuration ----

echo "--- Configuring swap ---"
CURRENT_SWAP=$(swapon --show=SIZE --bytes --noheadings 2>/dev/null | awk '{s+=$1} END {print s+0}')
if [ "${CURRENT_SWAP}" -lt "${SWAP_SIZE_BYTES}" ]; then
    if [ ! -f "${SWAPFILE}" ]; then
        echo "    Creating ${SWAP_SIZE} swap file..."
        fallocate -l "${SWAP_SIZE}" "${SWAPFILE}"
        chmod 600 "${SWAPFILE}"
        mkswap "${SWAPFILE}"
    fi
    if ! swapon --show=NAME --noheadings 2>/dev/null | grep -q "${SWAPFILE}"; then
        echo "    Enabling swap..."
        swapon "${SWAPFILE}"
    fi
    if ! grep -q "${SWAPFILE}" /etc/fstab; then
        echo "    Adding swap to fstab..."
        echo "${SWAPFILE} none swap sw 0 0" >> /etc/fstab
    fi
    echo "    Swap configured: ${SWAP_SIZE}"
else
    echo "    Swap already sufficient: $(swapon --show=SIZE --noheadings 2>/dev/null | head -1)"
fi

# ---- Kernel Parameters ----

echo "--- Setting kernel parameters ---"
if ! grep -q '^vm.swappiness=10' /etc/sysctl.conf; then
    sed -i '/^vm.swappiness=/d' /etc/sysctl.conf
    echo "vm.swappiness=10" >> /etc/sysctl.conf
    echo "    Set vm.swappiness=10"
fi
if ! grep -q '^vm.vfs_cache_pressure=50' /etc/sysctl.conf; then
    sed -i '/^vm.vfs_cache_pressure=/d' /etc/sysctl.conf
    echo "vm.vfs_cache_pressure=50" >> /etc/sysctl.conf
    echo "    Set vm.vfs_cache_pressure=50"
fi
sysctl -p &>/dev/null || true

# ---- Power Mode ----

echo "--- Setting power mode ---"
if command -v nvpmodel &>/dev/null; then
    CURRENT_MODE=$(nvpmodel -q 2>/dev/null | grep -oP 'NV Power Mode: MAXN|MODE_\d+' || echo "unknown")
    echo "    Current power mode: ${CURRENT_MODE}"
    nvpmodel -m 0
    echo "    Set power mode to 0 (15W MAXN)"
else
    echo "    WARNING: nvpmodel not found, skipping power mode configuration"
fi

echo "--- Running jetson_clocks ---"
if command -v jetson_clocks &>/dev/null; then
    jetson_clocks
    echo "    Clocks maximized"
else
    echo "    WARNING: jetson_clocks not found, skipping"
fi

# ---- Directory Structure ----

echo "--- Creating directory structure ---"
if [ ! -d "${INSTALL_DIR}" ]; then
    mkdir -p "${INSTALL_DIR}"
    echo "    Created ${INSTALL_DIR}"
fi
chown "${JETSON_USER}:${JETSON_USER}" "${INSTALL_DIR}"

if [ ! -d "${CONFIG_DIR}" ]; then
    mkdir -p "${CONFIG_DIR}"
    echo "    Created ${CONFIG_DIR}"
fi

if [ ! -d "${EXPERIENCE_DIR}" ]; then
    mkdir -p "${EXPERIENCE_DIR}"
    chown "${JETSON_USER}:${JETSON_USER}" "${EXPERIENCE_DIR}"
    echo "    Created ${EXPERIENCE_DIR}"
fi

# ---- Disk Space Check ----

echo "--- Checking available disk space ---"
AVAIL_GB=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
echo "    Available disk space: ${AVAIL_GB} GiB"
if [ "${AVAIL_GB}" -lt 10 ]; then
    echo "    WARNING: Only ${AVAIL_GB} GiB free. Consider running scripts/jetson_disk_cleanup.sh"
fi

echo "=== System setup complete ==="
