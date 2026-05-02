#!/usr/bin/env bash
# jetson-runner-install.sh
#
# Non-interactive installer for the GitHub Actions self-hosted runner that
# powers .github/workflows/jetson-nightly.yml. Run on a Jetson Orin Nano host
# logged in as the deploy user (default: jetson).
#
# Required env vars (real run):
#   RUNNER_TOKEN         — one-time registration token. Generate at
#                          https://github.com/${GITHUB_REPO}/settings/actions/runners/new
#                          (rotates ~1 h after issue).
#
# Optional env vars (all have sane defaults):
#   GITHUB_REPO          — slug, default "ianshank/Mouse-Droid-AGI"
#   RUNNER_VERSION       — actions/runner release, default "2.319.1"
#   RUNNER_LABELS        — comma list, default "self-hosted,jetson,linux,arm64"
#   RUNNER_INSTALL_DIR   — install path, default "/opt/actions-runner"
#   RUNNER_USER          — service user, default "jetson"
#   RUNNER_NAME          — display name, default "$(hostname)-jetson"
#
# Usage:
#   bash scripts/jetson-runner-install.sh --dry-run          # plan only, exit 0
#   RUNNER_TOKEN=AAA bash scripts/jetson-runner-install.sh   # real install
#
# Exit codes:
#   0   success
#   2   missing required env var (RUNNER_TOKEN)
#   3   download / extract failure
#   4   register / svc.sh install failure
#
# After install:
#   1. Verify systemd service is up:
#        sudo systemctl status actions.runner.<repo>.<name>
#   2. Trigger the nightly workflow manually to confirm pickup:
#        gh workflow run jetson-nightly.yml --ref main
#   3. See docs/jetson-runner-setup.md for the full operator runbook.

set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \{0,1\}//' | head -40
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

GITHUB_REPO="${GITHUB_REPO:-ianshank/Mouse-Droid-AGI}"
RUNNER_VERSION="${RUNNER_VERSION:-2.319.1}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,jetson,linux,arm64}"
RUNNER_INSTALL_DIR="${RUNNER_INSTALL_DIR:-/opt/actions-runner}"
RUNNER_USER="${RUNNER_USER:-jetson}"
RUNNER_NAME="${RUNNER_NAME:-$(hostname 2>/dev/null || echo jetson)-jetson}"
RUNNER_TARBALL="actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz"
RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_TARBALL}"
SERVICE_TEMPLATE_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_TEMPLATE="${SERVICE_TEMPLATE_DIR}/github-actions-runner.service.template"

log() { printf '[runner-install] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Dry run — no side effects, exits 0 with full plan in stdout.
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN: plan only — no files created, no commands executed."
    log "DRY-RUN: would download ${RUNNER_TARBALL} from ${RUNNER_URL}"
    log "DRY-RUN: would extract into ${RUNNER_INSTALL_DIR} as user ${RUNNER_USER}"
    log "DRY-RUN: would register against ${GITHUB_REPO} with labels ${RUNNER_LABELS}"
    log "DRY-RUN: would name the runner '${RUNNER_NAME}'"
    log "DRY-RUN: would install systemd unit from ${SERVICE_TEMPLATE}"
    log "DRY-RUN: exit 0"
    exit 0
fi

# ---------------------------------------------------------------------------
# Real-run preconditions.
# ---------------------------------------------------------------------------
if [[ -z "${RUNNER_TOKEN:-}" ]]; then
    cat >&2 <<EOF
ERROR: RUNNER_TOKEN env var required for real-run installation.

Generate a one-time token at:
  https://github.com/${GITHUB_REPO}/settings/actions/runners/new
(token rotates ~1 hour after issue — install promptly).

Usage:
  RUNNER_TOKEN=<token> bash scripts/jetson-runner-install.sh
  bash scripts/jetson-runner-install.sh --dry-run
EOF
    exit 2
fi

if [[ ! -f "${SERVICE_TEMPLATE}" ]]; then
    echo "ERROR: systemd template missing at ${SERVICE_TEMPLATE}" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# Real install path.
# ---------------------------------------------------------------------------
log "Installing GitHub Actions runner ${RUNNER_VERSION} into ${RUNNER_INSTALL_DIR}"

sudo mkdir -p "${RUNNER_INSTALL_DIR}"
sudo chown "${RUNNER_USER}:${RUNNER_USER}" "${RUNNER_INSTALL_DIR}"
cd "${RUNNER_INSTALL_DIR}"

if [[ ! -f "${RUNNER_TARBALL}" ]]; then
    log "Downloading ${RUNNER_URL}"
    sudo -u "${RUNNER_USER}" curl -fsSL "${RUNNER_URL}" -o "${RUNNER_TARBALL}" \
        || { log "ERROR: download failed"; exit 3; }
fi

log "Extracting ${RUNNER_TARBALL}"
sudo -u "${RUNNER_USER}" tar xzf "./${RUNNER_TARBALL}" \
    || { log "ERROR: extract failed"; exit 3; }

log "Registering against ${GITHUB_REPO} (labels=${RUNNER_LABELS}, name=${RUNNER_NAME})"
sudo -u "${RUNNER_USER}" ./config.sh \
    --url "https://github.com/${GITHUB_REPO}" \
    --token "${RUNNER_TOKEN}" \
    --labels "${RUNNER_LABELS}" \
    --name "${RUNNER_NAME}" \
    --unattended \
    --replace \
    || { log "ERROR: config.sh failed"; exit 4; }

log "Installing systemd service from template"
sudo install -m 0644 "${SERVICE_TEMPLATE}" /etc/systemd/system/actions-runner-mousedroid.service
sudo sed -i \
    -e "s|@RUNNER_USER@|${RUNNER_USER}|g" \
    -e "s|@RUNNER_INSTALL_DIR@|${RUNNER_INSTALL_DIR}|g" \
    /etc/systemd/system/actions-runner-mousedroid.service
sudo systemctl daemon-reload
sudo systemctl enable --now actions-runner-mousedroid.service \
    || { log "ERROR: systemctl enable/start failed"; exit 4; }

log "Runner installed and started."
log "Verify with:  sudo systemctl status actions-runner-mousedroid.service"
log "Trigger workflow: gh workflow run jetson-nightly.yml --ref main"
exit 0
