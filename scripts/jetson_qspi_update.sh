#!/bin/bash
# QSPI firmware update for Jetson Orin Nano.
# Required before JetPack 6.x SD card images will boot on factory-firmware boards.
#
# PREREQUISITE: Boot the Orin Nano from a JetPack 5.1.3 SD card first.
# Download the JetPack 5.1.3 image from:
#   https://developer.nvidia.com/embedded/downloads
# Flash it to the microSD with Balena Etcher, then boot and run this script.
#
# Usage: sudo bash scripts/jetson_qspi_update.sh
set -euo pipefail

echo "=== Jetson Orin Nano QSPI Firmware Update ==="

# ---- Check environment ----

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (sudo)."
    exit 1
fi

if [ ! -f /etc/nv_tegra_release ]; then
    echo "ERROR: Not running on a Jetson device."
    exit 1
fi

# ---- Check current firmware version ----

echo "--- Checking current firmware version ---"
CURRENT_L4T=""
if [ -f /etc/nv_tegra_release ]; then
    CURRENT_L4T=$(head -1 /etc/nv_tegra_release)
    echo "    Current L4T: ${CURRENT_L4T}"
fi

# Check if firmware is already JetPack 6 compatible (L4T 36.x)
if echo "${CURRENT_L4T}" | grep -q "R36"; then
    echo "    Firmware is already JetPack 6 compatible (L4T 36.x)."
    echo "    QSPI update is NOT needed. You can flash JetPack 6.x directly."
    exit 0
fi

echo "    Firmware is pre-JetPack 6. QSPI update required."

# ---- Update system packages ----

echo "--- Updating system packages (includes UEFI firmware) ---"
apt-get update
apt-get upgrade -y
echo "    System packages updated. Rebooting to apply UEFI update..."
echo ""
echo "    After reboot, run this script again to install the QSPI updater."
echo "    Press Ctrl+C within 10 seconds to cancel reboot."
sleep 10
reboot

# NOTE: The script exits here on first run. After reboot, run again:

# ---- Install QSPI updater ----

echo "--- Installing QSPI updater package ---"
if apt-get install -y nvidia-l4t-jetson-orin-nano-qspi-updater 2>/dev/null; then
    echo "    QSPI updater installed successfully."
    echo ""
    echo "    The device will reboot and flash QSPI firmware automatically."
    echo "    After the QSPI flash completes, the device will NOT boot from"
    echo "    the current JetPack 5.x SD card (this is expected)."
    echo ""
    echo "    NEXT STEPS:"
    echo "    1. Power off the device after QSPI update completes"
    echo "    2. Remove the JetPack 5.x SD card"
    echo "    3. Flash a JetPack 6.x image to the SD card using Balena Etcher"
    echo "    4. Insert the JetPack 6.x SD card and power on"
    echo ""
    echo "    Rebooting in 10 seconds to apply QSPI update..."
    sleep 10
    reboot
else
    echo "    WARNING: QSPI updater package not found."
    echo "    This may mean your board already has compatible firmware,"
    echo "    or the package name has changed."
    echo "    Try booting a JetPack 6.x SD card image directly."
fi
