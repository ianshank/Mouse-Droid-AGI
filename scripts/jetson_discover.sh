#!/bin/bash
# Network auto-discovery for Jetson Orin Nano.
# Tries USB device mode, mDNS, and subnet scan in order.
# Cross-platform: WSL2, native Linux, macOS.
# Usage: bash scripts/jetson_discover.sh [--verbose]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${HOME}/.mousedroid"
CACHE_FILE="${CACHE_DIR}/jetson_host"
VERBOSE="${VERBOSE:-false}"
SSH_USER="${SSH_USER:-jetson}"
SSH_TIMEOUT=5

# Allow explicit override via environment variable
JETSON_HOST="${JETSON_HOST:-}"

# Common subnets to scan when other methods fail
SCAN_SUBNETS=("192.168.1.0/24" "192.168.0.0/24" "10.0.0.0/24")

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

log() {
    if [[ "${VERBOSE}" == "true" || "${1:-}" == "--force" ]]; then
        shift 2>/dev/null || true
        echo "    $*" >&2
    fi
}

log_section() {
    if [[ "${VERBOSE}" == "true" ]]; then
        echo "--- $* ---" >&2
    fi
}

# Validate that an IP/hostname is reachable via SSH.
# Returns 0 if SSH works, 1 otherwise.
validate_ssh() {
    mkdir -p "${HOME}/.mousedroid"
    local host="$1"
    ssh -o StrictHostKeyChecking=accept-new \
        -o UserKnownHostsFile="${HOME}/.mousedroid/known_hosts" \
        -o ConnectTimeout="${SSH_TIMEOUT}" \
        -o BatchMode=yes \
        -o LogLevel=ERROR \
        "${SSH_USER}@${host}" "echo ok" &>/dev/null
}

# Validate that a host responds to ping.
try_ping() {
    local host="$1"
    if [[ "${PLATFORM}" == "macos" ]]; then
        ping -c 1 -W 2 "${host}" &>/dev/null
    else
        ping -c 1 -W 2 "${host}" &>/dev/null
    fi
}

# Cache the discovered IP for future runs.
cache_host() {
    local host="$1"
    mkdir -p "${CACHE_DIR}"
    echo "${host}" > "${CACHE_FILE}"
    log "Cached Jetson host: ${host} -> ${CACHE_FILE}"
}

# Read cached host if available and still reachable.
try_cached() {
    if [[ -f "${CACHE_FILE}" ]]; then
        local cached
        cached="$(cat "${CACHE_FILE}" 2>/dev/null | tr -d '[:space:]')"
        if [[ -n "${cached}" ]]; then
            log_section "Trying cached host: ${cached}"
            if try_ping "${cached}"; then
                log "Cached host responds to ping."
                if validate_ssh "${cached}"; then
                    log "SSH verified on cached host."
                    echo "${cached}"
                    return 0
                fi
                log "Cached host ping OK but SSH failed."
            else
                log "Cached host not reachable."
            fi
        fi
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Discovery methods
# ---------------------------------------------------------------------------

# Method 1: USB device mode (192.168.55.1)
try_usb_device_mode() {
    local usb_ip="192.168.55.1"
    log_section "Trying USB device mode (${usb_ip})"

    if [[ "${PLATFORM}" == "wsl2" ]]; then
        log "WSL2 detected — USB device mode requires usbipd attach."
    fi

    if try_ping "${usb_ip}"; then
        log "USB device mode: ping OK."
        if validate_ssh "${usb_ip}"; then
            log "USB device mode: SSH OK."
            echo "${usb_ip}"
            return 0
        fi
        log "USB device mode: ping OK but SSH failed."
    else
        log "USB device mode: not reachable."
    fi
    return 1
}

# Method 2: mDNS (mousedroid.local)
try_mdns() {
    local mdns_host="mousedroid.local"
    log_section "Trying mDNS (${mdns_host})"

    # macOS has native mDNS; Linux needs avahi
    if try_ping "${mdns_host}"; then
        log "mDNS: ping OK."
        if validate_ssh "${mdns_host}"; then
            log "mDNS: SSH OK."
            echo "${mdns_host}"
            return 0
        fi
        log "mDNS: ping OK but SSH failed."
    else
        log "mDNS: not reachable."
    fi
    return 1
}

# Method 3: Subnet scan using nmap or arp-scan
try_subnet_scan() {
    log_section "Trying subnet scan"

    local scanner=""
    if command -v nmap &>/dev/null; then
        scanner="nmap"
    elif command -v arp-scan &>/dev/null; then
        scanner="arp-scan"
    else
        log "Neither nmap nor arp-scan found. Install one:"
        log "  sudo apt-get install nmap    # or"
        log "  sudo apt-get install arp-scan"
        return 1
    fi

    for subnet in "${SCAN_SUBNETS[@]}"; do
        log "Scanning ${subnet} with ${scanner}..."
        local hosts=()

        if [[ "${scanner}" == "nmap" ]]; then
            # nmap -sn: ping scan, no port scan.  Extract live hosts.
            mapfile -t hosts < <(
                nmap -sn "${subnet}" 2>/dev/null \
                | grep -oP '(?<=Nmap scan report for )\S+' \
                | grep -vE '\.1$|\.255$' \
                || true
            )
        elif [[ "${scanner}" == "arp-scan" ]]; then
            mapfile -t hosts < <(
                sudo arp-scan --localnet "${subnet}" 2>/dev/null \
                | grep -oP '^\d+\.\d+\.\d+\.\d+' \
                || true
            )
        fi

        for host in "${hosts[@]}"; do
            [[ -z "${host}" ]] && continue
            log "Trying SSH on ${host}..."
            if validate_ssh "${host}"; then
                # Verify it's actually a Jetson by checking /etc/nv_tegra_release
                if ssh -o StrictHostKeyChecking=accept-new \
                       -o UserKnownHostsFile="${HOME}/.mousedroid/known_hosts" \
                       -o ConnectTimeout="${SSH_TIMEOUT}" \
                       -o BatchMode=yes \
                       -o LogLevel=ERROR \
                       "${SSH_USER}@${host}" \
                       "test -f /etc/nv_tegra_release" &>/dev/null; then
                    log "Found Jetson at ${host}."
                    echo "${host}"
                    return 0
                fi
                log "${host} has SSH but is not a Jetson."
            fi
        done
    done

    log "Subnet scan: no Jetson found."
    return 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    local verbose_flag=false

    for arg in "$@"; do
        case "${arg}" in
            --verbose|-v)
                verbose_flag=true
                ;;
            --help|-h)
                echo "Usage: $0 [--verbose]"
                echo ""
                echo "Auto-discover the Jetson on the network and print its IP/hostname."
                echo "The result is cached in ${CACHE_FILE}."
                echo ""
                echo "Environment variables:"
                echo "  JETSON_HOST    Override auto-discovery with an explicit host"
                echo "  SSH_USER       SSH user (default: ${SSH_USER})"
                echo "  VERBOSE        Enable verbose output (default: false)"
                exit 0
                ;;
        esac
    done

    if [[ "${verbose_flag}" == "true" ]]; then
        VERBOSE="true"
    fi

    echo "=== MouseDroid Jetson Discovery ===" >&2

    # Honor explicit override
    if [[ -n "${JETSON_HOST}" ]]; then
        log_section "Using JETSON_HOST override: ${JETSON_HOST}"
        if validate_ssh "${JETSON_HOST}"; then
            cache_host "${JETSON_HOST}"
            echo "${JETSON_HOST}"
            echo "=== Discovery complete ===" >&2
            exit 0
        else
            echo "    ERROR: JETSON_HOST=${JETSON_HOST} is not reachable via SSH." >&2
            exit 1
        fi
    fi

    # Try cached host first
    local found=""
    found="$(try_cached)" && {
        echo "=== Discovery complete (cached) ===" >&2
        echo "${found}"
        exit 0
    }

    # Method 1: USB device mode
    found="$(try_usb_device_mode)" && {
        cache_host "${found}"
        echo "=== Discovery complete (USB) ===" >&2
        echo "${found}"
        exit 0
    }

    # Method 2: mDNS
    found="$(try_mdns)" && {
        cache_host "${found}"
        echo "=== Discovery complete (mDNS) ===" >&2
        echo "${found}"
        exit 0
    }

    # Method 3: Subnet scan
    found="$(try_subnet_scan)" && {
        cache_host "${found}"
        echo "=== Discovery complete (subnet scan) ===" >&2
        echo "${found}"
        exit 0
    }

    echo "    ERROR: Could not discover Jetson on the network." >&2
    echo "    Troubleshooting:" >&2
    echo "      1. Ensure the Jetson is powered on and booted." >&2
    echo "      2. Try USB device mode: connect USB-C and ping 192.168.55.1" >&2
    echo "      3. Set JETSON_HOST manually: export JETSON_HOST=<ip>" >&2
    echo "      4. Install nmap for subnet scanning: sudo apt-get install nmap" >&2
    echo "=== Discovery failed ===" >&2
    exit 1
}

main "$@"
