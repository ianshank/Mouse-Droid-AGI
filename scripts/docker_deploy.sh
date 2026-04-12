#!/usr/bin/env bash
# =============================================================================
# MouseDroidAGI — Docker Deploy Script for Jetson
# =============================================================================
# Idempotent deployment: pull image, build container, deploy config, start.
# Optionally installs and enables the systemd service for container management.
#
# Usage:
#   sudo bash scripts/docker_deploy.sh [OPTIONS]
#
# Options:
#   --service       Install and enable the systemd service
#   --no-build      Skip image build (pull only)
#   --health-only   Run health checks without deploying
#   --help          Show this help message
#
# Environment variables (all optional, with defaults):
#   MOUSEDROID_INSTALL_DIR   Project install dir (default: /opt/mousedroid)
#   MOUSEDROID_CONFIG_DIR    Config file dir (default: /etc/mousedroid)
#   MOUSEDROID_COMPOSE_FILE  Compose file path (default: <install_dir>/docker-compose.jetson.yml)
#   MOUSEDROID_CONTAINER     Container name (default: mousedroid)
#   MOUSEDROID_HEALTH_PORT   Telemetry health port (default: 8080)
#   MOUSEDROID_HEALTH_TIMEOUT  Health check timeout secs (default: 30)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configurable paths via environment variables
INSTALL_DIR="${MOUSEDROID_INSTALL_DIR:-/opt/mousedroid}"
CONFIG_DIR="${MOUSEDROID_CONFIG_DIR:-/etc/mousedroid}"
COMPOSE_FILE="${MOUSEDROID_COMPOSE_FILE:-${INSTALL_DIR}/docker-compose.jetson.yml}"
CONTAINER_NAME="${MOUSEDROID_CONTAINER:-mousedroid}"
HEALTH_PORT="${MOUSEDROID_HEALTH_PORT:-8080}"
HEALTH_TIMEOUT="${MOUSEDROID_HEALTH_TIMEOUT:-30}"

# Parse arguments
INSTALL_SERVICE=false
NO_BUILD=false
HEALTH_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --service)    INSTALL_SERVICE=true; shift ;;
        --no-build)   NO_BUILD=true; shift ;;
        --health-only) HEALTH_ONLY=true; shift ;;
        --help|-h)
            sed -n '2,/^# ====/{ /^# ====/d; s/^# \?//p }' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Health check function
# ---------------------------------------------------------------------------
health_check() {
    info "Running container health checks..."

    # Check container is running
    if ! docker ps --filter "name=${CONTAINER_NAME}" --filter "status=running" -q | grep -q .; then
        error "Container ${CONTAINER_NAME} is not running"
        return 1
    fi
    info "  Container status: running"

    # Check CUDA availability
    local cuda_check
    cuda_check=$(docker exec "${CONTAINER_NAME}" python3 -c \
        "import torch; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}')" 2>&1) || true
    info "  $cuda_check"

    if echo "$cuda_check" | grep -q "CUDA=True"; then
        info "  GPU acceleration: ENABLED"
    else
        warn "  GPU acceleration: DISABLED (CPU fallback active)"
    fi

    # Check mousedroid import
    local import_check
    import_check=$(docker exec "${CONTAINER_NAME}" python3 -c "import mousedroid; print('OK')" 2>&1) || true
    if [ "$import_check" = "OK" ]; then
        info "  mousedroid import: OK"
    else
        error "  mousedroid import: FAILED — $import_check"
        return 1
    fi

    # Check telemetry health endpoint (if available)
    local health_url="http://127.0.0.1:${HEALTH_PORT}/health"
    if curl -sf --max-time 5 "$health_url" >/dev/null 2>&1; then
        info "  Telemetry health endpoint: OK (${health_url})"
    else
        warn "  Telemetry health endpoint: not responding (${health_url})"
        warn "  (This is normal if telemetry is disabled or still starting)"
    fi

    # Check compose service status
    info "  Compose services:"
    docker compose -f "${COMPOSE_FILE}" ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null | \
        sed 's/^/    /' || true

    return 0
}

# ---------------------------------------------------------------------------
# Health-only mode
# ---------------------------------------------------------------------------
if [ "$HEALTH_ONLY" = true ]; then
    health_check
    exit $?
fi

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
info "=== MouseDroidAGI Docker Deployment ==="
info "  Install dir:  ${INSTALL_DIR}"
info "  Config dir:   ${CONFIG_DIR}"
info "  Compose file: ${COMPOSE_FILE}"
info "  Container:    ${CONTAINER_NAME}"
echo ""

if ! command -v docker &>/dev/null; then
    error "Docker is not installed. Install with: sudo apt-get install -y docker.io nvidia-container-toolkit"
    exit 1
fi

if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    warn "NVIDIA container runtime not detected. GPU may not work."
    warn "Install with: sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker"
fi

# ---------------------------------------------------------------------------
# Step 1: Ensure project source is deployed
# ---------------------------------------------------------------------------
info "Step 1: Checking project source at ${INSTALL_DIR}"
if [ ! -f "${INSTALL_DIR}/pyproject.toml" ]; then
    error "Project source not found at ${INSTALL_DIR}"
    error "Run the venv deployment first, or rsync the project manually."
    exit 1
fi
info "  Project source OK"

# ---------------------------------------------------------------------------
# Step 2: Deploy config files
# ---------------------------------------------------------------------------
info "Step 2: Deploying configuration files"
mkdir -p "${CONFIG_DIR}"
for cfg in "$PROJECT_DIR/config/"*.yaml; do
    [ -f "$cfg" ] || continue
    cp "$cfg" "${CONFIG_DIR}/"
    info "  -> $(basename "$cfg")"
done

# Deploy docker env template if not present
if [ ! -f "${CONFIG_DIR}/docker.env" ]; then
    if [ -f "$PROJECT_DIR/config/docker.env.example" ]; then
        cp "$PROJECT_DIR/config/docker.env.example" "${CONFIG_DIR}/docker.env"
        info "  -> docker.env (from template — edit before production use)"
    fi
fi

# ---------------------------------------------------------------------------
# Step 3: Build the container image
# ---------------------------------------------------------------------------
if [ "$NO_BUILD" = true ]; then
    info "Step 3: Pulling container image (--no-build)"
    docker compose -f "${COMPOSE_FILE}" pull 2>&1 | tail -5
else
    info "Step 3: Building mousedroid:jetson container image"
    info "  This will pull the L4T base image (~10 GB) on first run..."
    cd "${INSTALL_DIR}"
    docker compose -f "${COMPOSE_FILE}" build --no-cache 2>&1 | tail -5
fi

# ---------------------------------------------------------------------------
# Step 4: Stop existing container if running
# ---------------------------------------------------------------------------
if docker ps -q --filter "name=${CONTAINER_NAME}" | grep -q .; then
    info "Step 4: Stopping existing container"
    docker compose -f "${COMPOSE_FILE}" down --timeout 30
else
    info "Step 4: No existing container running"
fi

# ---------------------------------------------------------------------------
# Step 5: Start the container
# ---------------------------------------------------------------------------
info "Step 5: Starting mousedroid container"
docker compose -f "${COMPOSE_FILE}" up -d

# Wait for container to be healthy with timeout
info "  Waiting for container to start (timeout: ${HEALTH_TIMEOUT}s)..."
SECONDS=0
while [ $SECONDS -lt "$HEALTH_TIMEOUT" ]; do
    if docker ps --filter "name=${CONTAINER_NAME}" --filter "status=running" -q | grep -q .; then
        break
    fi
    sleep 2
done

if ! docker ps --filter "name=${CONTAINER_NAME}" --filter "status=running" -q | grep -q .; then
    error "Container failed to start within ${HEALTH_TIMEOUT}s"
    error "Logs:"
    docker compose -f "${COMPOSE_FILE}" logs --tail=20 2>&1 | sed 's/^/  /'
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 6: Health Check
# ---------------------------------------------------------------------------
info "Step 6: Running health checks"
health_check || warn "Some health checks failed (non-fatal for deployment)"

# ---------------------------------------------------------------------------
# Step 7: Install systemd service
# ---------------------------------------------------------------------------
if [ "$INSTALL_SERVICE" = true ]; then
    info "Step 7: Installing Docker systemd service"
    DOCKER_SERVICE="$SCRIPT_DIR/mousedroid-docker.service"
    if [ ! -f "$DOCKER_SERVICE" ]; then
        error "Service file not found: $DOCKER_SERVICE"
        exit 1
    fi

    cp "$DOCKER_SERVICE" /etc/systemd/system/mousedroid-docker.service
    systemctl daemon-reload
    systemctl enable mousedroid-docker
    info "  Service installed and enabled"

    # Stop the manually-started compose and let systemd manage it
    info "  Stopping manual compose (systemd will manage lifecycle)..."
    docker compose -f "${COMPOSE_FILE}" down --timeout 30
    systemctl start mousedroid-docker
    info "  Service started via systemd"

    # Verify systemd service is active
    sleep 5
    if systemctl is-active --quiet mousedroid-docker; then
        info "  systemd service: ACTIVE"
    else
        error "  systemd service: FAILED"
        error "  Check logs with: journalctl -u mousedroid-docker -n 50"
        exit 1
    fi
else
    info "Step 7: Skipping systemd service install (use --service to enable)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
info "=== Deployment Complete ==="
info "  Container:  ${CONTAINER_NAME}"
info "  Image:      mousedroid:jetson"
info "  Source:     ${INSTALL_DIR} (volume mount)"
info "  Config:     ${CONFIG_DIR}"
info "  Compose:    ${COMPOSE_FILE}"
echo ""
info "Commands:"
info "  Logs:       docker logs -f ${CONTAINER_NAME}"
info "  Shell:      docker exec -it ${CONTAINER_NAME} bash"
info "  Stop:       docker compose -f ${COMPOSE_FILE} down"
info "  Restart:    docker compose -f ${COMPOSE_FILE} restart"
info "  Health:     bash $0 --health-only"
if [ "$INSTALL_SERVICE" = true ]; then
    echo ""
    info "Systemd service:"
    info "  Status:     systemctl status mousedroid-docker"
    info "  Logs:       journalctl -u mousedroid-docker -f"
    info "  Restart:    systemctl restart mousedroid-docker"
fi
