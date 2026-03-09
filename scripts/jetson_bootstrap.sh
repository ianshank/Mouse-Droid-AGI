#!/bin/bash
# First-time SSH + user + group setup for Jetson Orin Nano.
# Runs entirely from the local machine — all operations execute via SSH.
# Usage: bash scripts/jetson_bootstrap.sh [JETSON_HOST]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
CACHE_FILE="${HOME}/.mousedroid/jetson_host"
SSH_USER="${SSH_USER:-jetson}"
INITIAL_USER="${INITIAL_USER:-jetson}"
SSH_TIMEOUT=5
SSH_PORT="${SSH_PORT:-22}"
TARGET_HOSTNAME="mousedroid"
TARGET_USER="jetson"
TARGET_GROUPS="gpio,i2c,spi,video,dialout"

echo "=== MouseDroid Jetson Bootstrap ==="

# ---------------------------------------------------------------------------
# Resolve Jetson host
# ---------------------------------------------------------------------------

resolve_host() {
    # Priority: argument > env var > cache file > discovery script
    if [[ -n "${1:-}" ]]; then
        echo "$1"
        return 0
    fi

    if [[ -n "${JETSON_HOST:-}" ]]; then
        echo "${JETSON_HOST}"
        return 0
    fi

    if [[ -f "${CACHE_FILE}" ]]; then
        local cached
        cached="$(cat "${CACHE_FILE}" 2>/dev/null | tr -d '[:space:]')"
        if [[ -n "${cached}" ]]; then
            echo "${cached}"
            return 0
        fi
    fi

    if [[ -x "${SCRIPT_DIR}/jetson_discover.sh" ]]; then
        local discovered
        discovered="$(bash "${SCRIPT_DIR}/jetson_discover.sh" 2>/dev/null || true)"
        if [[ -n "${discovered}" ]]; then
            echo "${discovered}"
            return 0
        fi
    fi

    return 1
}

JETSON_HOST="$(resolve_host "${1:-}")" || {
    echo "    ERROR: Could not determine Jetson host."
    echo "    Usage: $0 <host>  or  export JETSON_HOST=<host>"
    echo "    Or run: bash scripts/jetson_discover.sh"
    exit 1
}

echo "    Target: ${INITIAL_USER}@${JETSON_HOST}:${SSH_PORT}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Run a command on the Jetson via SSH.
remote() {
    mkdir -p "${HOME}/.mousedroid"
    ssh -o StrictHostKeyChecking=accept-new \
        -o UserKnownHostsFile="${HOME}/.mousedroid/known_hosts" \
        -o ConnectTimeout="${SSH_TIMEOUT}" \
        -o LogLevel=ERROR \
        -p "${SSH_PORT}" \
        "${INITIAL_USER}@${JETSON_HOST}" "$@"
}

# Run a command on the Jetson as root via SSH.
remote_sudo() {
    remote sudo "$@"
}

# ---------------------------------------------------------------------------
# Step 1: Verify SSH connectivity
# ---------------------------------------------------------------------------

echo "--- Verifying SSH connectivity ---"
if ! remote "echo ok" &>/dev/null; then
    echo "    ERROR: Cannot SSH to ${INITIAL_USER}@${JETSON_HOST}:${SSH_PORT}"
    echo "    Ensure the Jetson is booted and SSH is enabled."
    echo "    If this is a fresh flash, complete the OEM first-boot setup first."
    exit 1
fi
echo "    SSH connection verified."

# ---------------------------------------------------------------------------
# Step 2: Copy SSH public key
# ---------------------------------------------------------------------------

echo "--- Setting up SSH key authentication ---"

SSH_PUBKEY="${SSH_PUBKEY:-${HOME}/.ssh/id_ed25519.pub}"
if [[ ! -f "${SSH_PUBKEY}" ]]; then
    SSH_PUBKEY="${HOME}/.ssh/id_rsa.pub"
fi

if [[ -f "${SSH_PUBKEY}" ]]; then
    # Use ssh-copy-id for idempotent key installation.
    # ssh-copy-id won't add duplicates if key is already present.
    if command -v ssh-copy-id &>/dev/null; then
        ssh-copy-id -i "${SSH_PUBKEY}" \
            -o StrictHostKeyChecking=accept-new \
            -o UserKnownHostsFile="${HOME}/.mousedroid/known_hosts" \
            -o LogLevel=ERROR \
            -p "${SSH_PORT}" \
            "${INITIAL_USER}@${JETSON_HOST}" 2>/dev/null || {
                echo "    WARNING: ssh-copy-id failed (password auth may be required)."
                echo "    You may need to run manually:"
                echo "        ssh-copy-id -i ${SSH_PUBKEY} ${INITIAL_USER}@${JETSON_HOST}"
            }
    else
        # Fallback: manually append key if not already present
        pubkey_content="$(cat "${SSH_PUBKEY}")"
        remote "mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
                grep -qF '${pubkey_content}' ~/.ssh/authorized_keys 2>/dev/null || \
                echo '${pubkey_content}' >> ~/.ssh/authorized_keys && \
                chmod 600 ~/.ssh/authorized_keys"
    fi
    echo "    SSH key installed: ${SSH_PUBKEY}"
else
    echo "    WARNING: No SSH public key found at ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub"
    echo "    Generate one with: ssh-keygen -t ed25519"
    echo "    Skipping key installation."
fi

# ---------------------------------------------------------------------------
# Step 3: Create target user
# ---------------------------------------------------------------------------

echo "--- Ensuring user '${TARGET_USER}' exists ---"

remote_sudo bash -s <<REMOTE_USER
    if id "${TARGET_USER}" &>/dev/null; then
        echo "    User '${TARGET_USER}' already exists."
    else
        echo "    Creating user '${TARGET_USER}'..."
        useradd -m -s /bin/bash "${TARGET_USER}"
        echo "    User '${TARGET_USER}' created."
    fi

    # Ensure user has sudo access (needed for deploy_jetson.sh)
    if [[ ! -f "/etc/sudoers.d/${TARGET_USER}" ]]; then
        echo "${TARGET_USER} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${TARGET_USER}"
        chmod 440 "/etc/sudoers.d/${TARGET_USER}"
        echo "    Passwordless sudo configured for '${TARGET_USER}'."
    else
        echo "    Sudoers entry already exists."
    fi
REMOTE_USER

# ---------------------------------------------------------------------------
# Step 4: Add user to hardware groups
# ---------------------------------------------------------------------------

echo "--- Adding '${TARGET_USER}' to hardware groups ---"

# Split groups and add one-by-one for idempotency (usermod -aG is already
# idempotent, but individual adds give clearer logging).
IFS=',' read -ra GROUP_LIST <<< "${TARGET_GROUPS}"
for grp in "${GROUP_LIST[@]}"; do
    remote_sudo bash -c "
        if getent group '${grp}' >/dev/null 2>&1; then
            usermod -aG '${grp}' '${TARGET_USER}' 2>/dev/null
            echo \"    ${TARGET_USER} -> ${grp} (ok)\"
        else
            echo \"    WARNING: group '${grp}' does not exist on the Jetson.\"
        fi
    "
done

# ---------------------------------------------------------------------------
# Step 5: Copy SSH key to target user (if different from initial user)
# ---------------------------------------------------------------------------

if [[ "${INITIAL_USER}" != "${TARGET_USER}" && -f "${SSH_PUBKEY}" ]]; then
    echo "--- Copying SSH key to '${TARGET_USER}' ---"
    pubkey_data="$(cat "${SSH_PUBKEY}")"
    remote_sudo bash -c "
        target_home=\"\$(eval echo ~${TARGET_USER})\"
        mkdir -p \"\${target_home}/.ssh\"
        chmod 700 \"\${target_home}/.ssh\"
        touch \"\${target_home}/.ssh/authorized_keys\"
        if ! grep -qF '${pubkey_data}' \"\${target_home}/.ssh/authorized_keys\" 2>/dev/null; then
            echo '${pubkey_data}' >> \"\${target_home}/.ssh/authorized_keys\"
            echo \"    Key installed for '${TARGET_USER}'.\"
        else
            echo \"    Key already present for '${TARGET_USER}'.\"
        fi
        chmod 600 \"\${target_home}/.ssh/authorized_keys\"
        chown -R ${TARGET_USER}:${TARGET_USER} \"\${target_home}/.ssh\"
    "
fi

# ---------------------------------------------------------------------------
# Step 6: Set hostname
# ---------------------------------------------------------------------------

echo "--- Setting hostname to '${TARGET_HOSTNAME}' ---"

remote_sudo bash -c "
    current_hostname=\"\$(hostnamectl --static 2>/dev/null || hostname)\"
    if [[ \"\${current_hostname}\" == '${TARGET_HOSTNAME}' ]]; then
        echo \"    Hostname already set to '${TARGET_HOSTNAME}'.\"
    else
        hostnamectl set-hostname '${TARGET_HOSTNAME}' 2>/dev/null || {
            echo '${TARGET_HOSTNAME}' > /etc/hostname
            hostname '${TARGET_HOSTNAME}'
        }
        # Update /etc/hosts for local resolution
        if ! grep -q '${TARGET_HOSTNAME}' /etc/hosts 2>/dev/null; then
            sed -i \"s/127\.0\.1\.1.*/127.0.1.1\t${TARGET_HOSTNAME}/\" /etc/hosts 2>/dev/null || \
            echo \"127.0.1.1\t${TARGET_HOSTNAME}\" >> /etc/hosts
        fi
        echo \"    Hostname set to '${TARGET_HOSTNAME}'.\"
    fi
"

# ---------------------------------------------------------------------------
# Step 7: Disable GUI desktop
# ---------------------------------------------------------------------------

echo "--- Disabling GUI desktop (multi-user.target) ---"

remote_sudo bash -c "
    current_target=\"\$(systemctl get-default 2>/dev/null || echo 'unknown')\"
    if [[ \"\${current_target}\" == 'multi-user.target' ]]; then
        echo \"    Default target already set to multi-user.target.\"
    else
        systemctl set-default multi-user.target
        echo \"    Default target changed from \${current_target} to multi-user.target.\"
        echo \"    GUI will be disabled on next reboot.\"
    fi
"

# ---------------------------------------------------------------------------
# Step 8: Verify setup
# ---------------------------------------------------------------------------

echo "--- Verifying bootstrap configuration ---"

remote bash -s <<'VERIFY'
    echo "    Hostname: $(hostname)"
    echo "    Default target: $(systemctl get-default 2>/dev/null || echo 'unknown')"
    echo "    L4T: $(head -1 /etc/nv_tegra_release 2>/dev/null || echo 'not found')"
    echo "    Kernel: $(uname -r)"
    echo "    Disk usage:"
    df -h / 2>/dev/null | tail -1 | awk '{print "      " $0}'
    echo "    Memory:"
    free -h 2>/dev/null | grep Mem | awk '{print "      Total: " $2 "  Available: " $7}'
VERIFY

echo "=== Bootstrap complete ==="
echo "    Jetson is ready for deployment."
echo "    Next step: bash scripts/deploy_jetson.sh"
echo "    Or remotely: ssh ${TARGET_USER}@${JETSON_HOST} 'sudo bash /opt/mousedroid/scripts/deploy_jetson.sh'"
