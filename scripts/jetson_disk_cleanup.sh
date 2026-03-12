#!/bin/bash
# Disk cleanup for space-constrained Jetson deployments (64GB microSD).
# Reclaims 2-3 GiB by removing desktop packages, caches, and samples.
#
# Usage: sudo bash scripts/jetson_disk_cleanup.sh
set -euo pipefail

echo "=== Jetson Disk Cleanup ==="

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (sudo)."
    exit 1
fi

# ---- Pre-cleanup disk usage ----

echo "--- Current disk usage ---"
BEFORE_AVAIL=$(df -BM / | tail -1 | awk '{print $4}')
df -h /
echo ""

# ---- Remove desktop/GUI packages ----

echo "--- Removing desktop packages (if present) ---"
DESKTOP_PKGS="ubuntu-desktop gnome-shell gdm3 chromium-browser libreoffice* thunderbird* shotwell* rhythmbox* totem* cheese*"
for pkg in ${DESKTOP_PKGS}; do
    if dpkg -l "${pkg}" 2>/dev/null | grep -q '^ii'; then
        echo "    Removing: ${pkg}"
        apt-get purge -y "${pkg}" 2>/dev/null || true
    fi
done

# ---- Autoremove orphaned dependencies ----

echo "--- Removing orphaned dependencies ---"
apt-get autoremove -y 2>/dev/null || true

# ---- Clean apt cache ----

echo "--- Cleaning apt cache ---"
apt-get clean
rm -rf /var/cache/apt/archives/*.deb 2>/dev/null || true

# ---- Remove CUDA samples (if present) ----

echo "--- Removing CUDA samples (if present) ---"
for dir in /usr/local/cuda-*/samples; do
    if [ -d "${dir}" ]; then
        echo "    Removing: ${dir}"
        rm -rf "${dir}"
    fi
done

# ---- Remove old log files ----

echo "--- Cleaning old logs ---"
journalctl --vacuum-size=50M 2>/dev/null || true
find /var/log -name "*.gz" -delete 2>/dev/null || true
find /var/log -name "*.old" -delete 2>/dev/null || true

# ---- Post-cleanup disk usage ----

echo ""
echo "--- Disk usage after cleanup ---"
AFTER_AVAIL=$(df -BM / | tail -1 | awk '{print $4}')
df -h /

BEFORE_NUM=$(echo "${BEFORE_AVAIL}" | tr -d 'M')
AFTER_NUM=$(echo "${AFTER_AVAIL}" | tr -d 'M')
SAVED=$((AFTER_NUM - BEFORE_NUM))
echo ""
echo "    Reclaimed: ${SAVED} MiB"
echo "=== Disk cleanup complete ==="
