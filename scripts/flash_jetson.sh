#!/bin/bash
# SD card flashing guide and automation for Jetson Orin Nano.
# Designed for Windows + WSL2 users; also works on native Linux.
# Usage: bash scripts/flash_jetson.sh [--flash | --verify]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
JETPACK_VERSION="${JETPACK_VERSION:-6.1}"
L4T_RELEASE="${L4T_RELEASE:-36.4}"
FLASH_DEVICE="${FLASH_DEVICE:-}"
SSH_USER="${SSH_USER:-jetson}"
JETSON_IP="${JETSON_IP:-192.168.55.1}"
SSH_TIMEOUT=5
BOOT_WAIT_SECONDS=120
BOOT_POLL_INTERVAL=5

echo "=== MouseDroid Jetson Flash Utility ==="

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

detect_platform() {
    if grep -qi microsoft /proc/version 2>/dev/null; then
        echo "wsl2"
    elif [[ "$(uname)" == "Darwin" ]]; then
        echo "macos"
    else
        echo "linux"
    fi
}

PLATFORM="$(detect_platform)"

print_manual_steps() {
    cat <<'GUIDE'

=== MANUAL PREPARATION STEPS ===

Before running this script with --flash, complete the following:

--- 1. Enter Recovery Mode on the Jetson Orin Nano ---
    a) Power off the Jetson completely.
    b) Locate the Force Recovery (FC REC) jumper on the carrier board.
       - On the Orin Nano Developer Kit, short pins 9 and 10 on the
         button header (J14), or hold the Force Recovery button.
    c) While holding the recovery button/jumper, connect USB-C from the
       Jetson's flashing port to your host PC.
    d) Apply power (DC barrel jack or USB-C PD, depending on carrier board).
    e) Release the recovery button after 2 seconds.

--- 2. Verify Recovery Mode ---
    On Linux / WSL2:
        lsusb | grep -i nvidia
    You should see: "NVIDIA Corp. APX" (ID 0955:7523 or similar).

--- 3. WSL2-specific: USB Passthrough with usbipd-win ---
    Windows host (PowerShell as Administrator):
        winget install usbipd
        usbipd list                     # find the NVIDIA APX device bus ID
        usbipd bind --busid <BUSID>     # first-time bind (persists across reboots)
        usbipd attach --wsl --busid <BUSID>

    Inside WSL2:
        lsusb | grep -i nvidia          # confirm the device appeared

    NOTE: You must re-attach after every Jetson reboot/reconnect:
        usbipd attach --wsl --busid <BUSID>

--- 4. Install NVIDIA SDK Manager (if not already installed) ---
    Download from: https://developer.nvidia.com/sdk-manager
    Or install the CLI-only version:
        sudo apt-get install -y sdkmanager

GUIDE
}

check_sdkmanager() {
    echo "--- Checking for NVIDIA SDK Manager ---"
    if command -v sdkmanager &>/dev/null; then
        echo "    Found: $(command -v sdkmanager)"
        sdkmanager --version 2>/dev/null || true
        return 0
    fi

    # Check for GUI version in common install paths
    local gui_paths=(
        "/opt/nvidia/sdkmanager/sdkmanager"
        "$HOME/.local/share/sdkmanager/sdkmanager"
        "/usr/bin/sdkmanager"
    )
    for p in "${gui_paths[@]}"; do
        if [[ -x "$p" ]]; then
            echo "    Found SDK Manager at: $p"
            return 0
        fi
    done

    echo "    ERROR: NVIDIA SDK Manager not found."
    echo "    Download from: https://developer.nvidia.com/sdk-manager"
    echo "    Or: sudo apt-get install sdkmanager"
    return 1
}

check_recovery_mode() {
    echo "--- Checking Jetson Recovery Mode ---"
    if ! command -v lsusb &>/dev/null; then
        echo "    WARNING: lsusb not found. Install usbutils: sudo apt-get install usbutils"
        return 1
    fi

    if lsusb | grep -qi "nvidia"; then
        echo "    Jetson detected in recovery mode."
        lsusb | grep -i nvidia
        return 0
    else
        echo "    ERROR: No NVIDIA device in recovery mode detected."
        echo "    Follow the manual steps above to enter recovery mode."
        return 1
    fi
}

do_flash() {
    echo "=== Flashing JetPack ${JETPACK_VERSION} (L4T ${L4T_RELEASE}) ==="

    if [[ "${PLATFORM}" == "wsl2" ]]; then
        echo "--- WSL2 detected ---"
        echo "    Ensure usbipd has attached the NVIDIA APX device to WSL."
        echo "    Run in Windows PowerShell (Admin):"
        echo "        usbipd attach --wsl --busid <BUSID>"
        echo ""
    fi

    check_sdkmanager || exit 1
    check_recovery_mode || exit 1

    echo "--- Starting SDK Manager flash ---"
    echo "    Target: Jetson Orin Nano"
    echo "    JetPack: ${JETPACK_VERSION}"
    echo "    Components: L4T ${L4T_RELEASE}, CUDA 12.x, cuDNN, TensorRT"
    echo ""

    # sdkmanager CLI flash command.  --cli disables the GUI.
    # The user will be prompted by sdkmanager for NVIDIA login if needed.
    sdkmanager --cli install \
        --logintype devzone \
        --product Jetson \
        --target JETSON_ORIN_NANO_TARGETS \
        --version "${JETPACK_VERSION}" \
        --flash all \
        ${FLASH_DEVICE:+--select "${FLASH_DEVICE}"} \
        2>&1 | tee /tmp/mousedroid_flash.log

    echo "--- Flash command completed ---"
    echo "    Full log saved to /tmp/mousedroid_flash.log"
}

wait_for_boot() {
    local target_ip="${1:-${JETSON_IP}}"
    local elapsed=0

    echo "=== Waiting for Jetson to boot (timeout: ${BOOT_WAIT_SECONDS}s) ==="
    echo "    Target IP: ${target_ip}"

    while [[ ${elapsed} -lt ${BOOT_WAIT_SECONDS} ]]; do
        if ping -c 1 -W 2 "${target_ip}" &>/dev/null; then
            echo "    Jetson is responding to ping at ${target_ip}."
            break
        fi
        sleep "${BOOT_POLL_INTERVAL}"
        elapsed=$((elapsed + BOOT_POLL_INTERVAL))
        echo "    Waiting... (${elapsed}s / ${BOOT_WAIT_SECONDS}s)"
    done

    if [[ ${elapsed} -ge ${BOOT_WAIT_SECONDS} ]]; then
        echo "    ERROR: Jetson did not respond within ${BOOT_WAIT_SECONDS}s."
        echo "    Check network connectivity and try manual connection."
        return 1
    fi

    echo "--- Verifying SSH availability ---"
    local ssh_elapsed=0
    local ssh_timeout=60
    while [[ ${ssh_elapsed} -lt ${ssh_timeout} ]]; do
        if ssh -o StrictHostKeyChecking=accept-new \
               -o ConnectTimeout="${SSH_TIMEOUT}" \
               -o BatchMode=yes \
               "${SSH_USER}@${target_ip}" "echo ok" &>/dev/null; then
            echo "    SSH is available at ${SSH_USER}@${target_ip}."
            return 0
        fi
        sleep "${BOOT_POLL_INTERVAL}"
        ssh_elapsed=$((ssh_elapsed + BOOT_POLL_INTERVAL))
        echo "    SSH not ready yet... (${ssh_elapsed}s / ${ssh_timeout}s)"
    done

    echo "    WARNING: Ping succeeded but SSH is not available yet."
    echo "    The Jetson may still be running first-boot setup (oem-config)."
    echo "    Complete the on-screen setup, then re-run: $0 --verify"
    return 1
}

print_connection_methods() {
    cat <<'CONN'

=== POST-FLASH CONNECTION METHODS ===

--- Method 1: USB Device Mode (recommended for first setup) ---
    The Jetson exposes a virtual Ethernet adapter over USB-C.
    Default IP: 192.168.55.1
    Connect USB-C between Jetson and host, then:
        ping 192.168.55.1
        ssh jetson@192.168.55.1

    WSL2 note: You must attach the USB Ethernet device via usbipd:
        usbipd list                                 # find the RNDIS/Ethernet device
        usbipd attach --wsl --busid <BUSID>

--- Method 2: Ethernet Direct Connection ---
    Connect an Ethernet cable between the Jetson and your host/router.
    If using a direct cable (no router), configure a static IP on your host:
        sudo ip addr add 192.168.1.100/24 dev eth0  # Linux example
    The Jetson will request DHCP by default. Check your router's DHCP leases
    or use: bash scripts/jetson_discover.sh

--- Method 3: WiFi ---
    During first-boot oem-config, select your WiFi network.
    Or configure after SSH access:
        sudo nmcli device wifi connect "YOUR_SSID" password "YOUR_PASSWORD"
    Find the assigned IP:
        ip addr show wlan0 | grep inet

CONN
}

do_verify() {
    echo "=== Verifying Jetson Connection ==="

    # Try USB device mode first
    echo "--- Trying USB device mode (192.168.55.1) ---"
    if ping -c 1 -W 2 192.168.55.1 &>/dev/null; then
        JETSON_IP="192.168.55.1"
        echo "    Jetson reachable at ${JETSON_IP}"
    else
        echo "    USB device mode not available."
        # Try discover script if available
        if [[ -x "${SCRIPT_DIR}/jetson_discover.sh" ]]; then
            echo "--- Running jetson_discover.sh ---"
            JETSON_IP="$(bash "${SCRIPT_DIR}/jetson_discover.sh" 2>/dev/null || true)"
        fi
    fi

    if [[ -z "${JETSON_IP}" ]]; then
        echo "    ERROR: Could not find Jetson on the network."
        print_connection_methods
        return 1
    fi

    echo "--- Verifying SSH ---"
    if ssh -o StrictHostKeyChecking=accept-new \
           -o ConnectTimeout="${SSH_TIMEOUT}" \
           "${SSH_USER}@${JETSON_IP}" "cat /etc/nv_tegra_release" 2>/dev/null; then
        echo "    SSH verified. Jetson is ready."
    else
        echo "    WARNING: SSH connection failed. Complete first-boot setup on the Jetson."
        return 1
    fi

    echo "--- Checking JetPack components ---"
    ssh -o ConnectTimeout="${SSH_TIMEOUT}" "${SSH_USER}@${JETSON_IP}" bash -s <<'REMOTE'
        echo "  L4T version:"
        head -1 /etc/nv_tegra_release 2>/dev/null || echo "    (not found)"
        echo "  CUDA:"
        nvcc --version 2>/dev/null | grep "release" || echo "    (not found)"
        echo "  cuDNN:"
        dpkg -l 2>/dev/null | grep cudnn | head -1 || echo "    (not found)"
        echo "  TensorRT:"
        dpkg -l 2>/dev/null | grep tensorrt | head -1 || echo "    (not found)"
REMOTE

    echo "=== Verification complete ==="
}

# ---------------------------------------------------------------------------
# Main

print_sdcard_instructions() {
    cat <<'SDCARD'

=== SD CARD IMAGE METHOD (Recommended for Orin Nano Dev Kit) ===

This method flashes a pre-built JetPack image directly to a microSD card.
No recovery mode or SDK Manager required.

--- Prerequisites ---
    1. 64GB+ microSD card (A2/V30 rated recommended for performance)
    2. Balena Etcher: https://etcher.balena.io/  (Windows/macOS/Linux)
    3. QSPI firmware must be JetPack 6 compatible (L4T 36.x).
       If your board has factory firmware (L4T 35.x or older),
       run: sudo bash scripts/jetson_qspi_update.sh  FIRST.

--- Steps ---
    1. Download the JetPack SD card image from:
       https://developer.nvidia.com/embedded/jetpack
       Select: Jetson Orin Nano Developer Kit -> SD Card Image Method
       File: Jetson_Orin_Nano_Developer_Kit_SD_Card_Image.zip (~15GB)

    2. Open Balena Etcher:
       - Click "Flash from file" -> select the downloaded .zip
       - Click "Select target" -> choose the microSD card (e.g. D:)
       - Click "Flash!" and wait for completion + verification

    3. IMPORTANT: When Windows prompts to format the card, click CANCEL.
       Windows cannot read Linux ext4 partitions — this is normal.

    4. Safely eject the SD card from your computer.

    5. Insert into the Jetson Orin Nano and power on.

    6. Complete the Ubuntu first-boot wizard (user, timezone, WiFi).
       Create user: jetson  (matches all deployment scripts)

--- After First Boot ---
    Connect via USB device mode:  ssh jetson@192.168.55.1
    Or run:  bash scripts/jetson_discover.sh
    Then:    bash scripts/jetson_bootstrap.sh <IP>
    Then:    bash scripts/deploy_remote.sh <IP> --full

SDCARD
}
# ---------------------------------------------------------------------------

case "${1:-}" in
    --flash)
        print_manual_steps
        echo ""
        read -rp "Have you completed the manual steps above? [y/N] " confirm
        if [[ "${confirm}" =~ ^[Yy]$ ]]; then
            do_flash
            echo ""
            print_connection_methods
            echo ""
            wait_for_boot "${JETSON_IP}"
        else
            echo "Aborting. Complete the manual steps first."
            exit 1
        fi
        ;;
    --verify)
        do_verify
        ;;
    --sdcard)        print_sdcard_instructions        ;;
    --wait)
        wait_for_boot "${2:-${JETSON_IP}}"
        ;;
    *)
        print_manual_steps
        print_connection_methods
        echo ""
        echo "=== Usage ==="
        echo "  $0                 Show this guide"
        echo "  $0 --flash         Start the flash process"
        echo "  $0 --verify        Verify Jetson connectivity and JetPack"
        echo "  $0 --wait [IP]     Wait for Jetson to boot and verify SSH"
        echo ""
        echo "Environment variables:"
        echo "  JETPACK_VERSION    JetPack version (default: ${JETPACK_VERSION})"
        echo "  L4T_RELEASE        L4T release (default: ${L4T_RELEASE})"
        echo "  SSH_USER           SSH user (default: ${SSH_USER})"
        echo "  JETSON_IP          Jetson IP (default: ${JETSON_IP})"
        ;;
esac

