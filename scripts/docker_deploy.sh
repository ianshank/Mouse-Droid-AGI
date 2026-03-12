#!/usr/bin/env bash
# =============================================================================
# MouseDroidAGI — Docker Deploy Script for Jetson
# =============================================================================
# Idempotent deployment: pull image, build container, deploy config, start.
#
# Usage:
#   sudo bash scripts/docker_deploy.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.jetson.yml"
CONTAINER_NAME="mousedroid"
CONFIG_DIR="/etc/mousedroid"
REMOTE_SRC="/opt/mousedroid"

# Colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
info "=== MouseDroidAGI Docker Deployment ==="

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
info "Step 1: Checking project source at $REMOTE_SRC"
if [ ! -f "$REMOTE_SRC/pyproject.toml" ]; then
    error "Project source not found at $REMOTE_SRC"
    error "Run the venv deployment first, or rsync the project manually."
    exit 1
fi
info "  Project source OK"

# ---------------------------------------------------------------------------
# Step 2: Deploy config files
# ---------------------------------------------------------------------------
info "Step 2: Deploying configuration files"
mkdir -p "$CONFIG_DIR"
for cfg in "$PROJECT_DIR/config/"*.yaml; do
    cp "$cfg" "$CONFIG_DIR/"
    info "  -> $(basename "$cfg")"
done

# ---------------------------------------------------------------------------
# Step 3: Build the container image
# ---------------------------------------------------------------------------
info "Step 3: Building mousedroid:jetson container image"
info "  This will pull the L4T base image (~10 GB) on first run..."
cd "$PROJECT_DIR"
docker compose -f "$COMPOSE_FILE" build --no-cache 2>&1 | tail -5

# ---------------------------------------------------------------------------
# Step 4: Stop existing container if running
# ---------------------------------------------------------------------------
if docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
    info "Step 4: Stopping existing container"
    docker compose -f "$COMPOSE_FILE" down
else
    info "Step 4: No existing container running"
fi

# ---------------------------------------------------------------------------
# Step 5: Start the container
# ---------------------------------------------------------------------------
info "Step 5: Starting mousedroid container"
docker compose -f "$COMPOSE_FILE" up -d

# Wait for container to be healthy
info "  Waiting for container to start..."
sleep 5

# ---------------------------------------------------------------------------
# Step 6: GPU Health Check
# ---------------------------------------------------------------------------
info "Step 6: Running GPU health check"

# Check CUDA availability
CUDA_CHECK=$(docker exec "$CONTAINER_NAME" python3 -c \
    "import torch; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}')" 2>&1)
info "  $CUDA_CHECK"

if echo "$CUDA_CHECK" | grep -q "CUDA=True"; then
    info "  GPU acceleration: ENABLED"
else
    warn "  GPU acceleration: DISABLED (CPU fallback active)"
fi

# Check mousedroid import
IMPORT_CHECK=$(docker exec "$CONTAINER_NAME" python3 -c "import mousedroid; print('OK')" 2>&1)
if [ "$IMPORT_CHECK" = "OK" ]; then
    info "  mousedroid import: OK"
else
    error "  mousedroid import: FAILED"
    error "  $IMPORT_CHECK"
fi

# ---------------------------------------------------------------------------
# Step 7: Install systemd service (optional)
# ---------------------------------------------------------------------------
DOCKER_SERVICE="$SCRIPT_DIR/mousedroid-docker.service"
if [ -f "$DOCKER_SERVICE" ]; then
    info "Step 7: Installing Docker systemd service"
    cp "$DOCKER_SERVICE" /etc/systemd/system/
    systemctl daemon-reload
    info "  Service installed. Enable with: sudo systemctl enable mousedroid-docker"
    info "  Start with: sudo systemctl start mousedroid-docker"
else
    info "Step 7: Skipping systemd service (file not found)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
info "=== Deployment Complete ==="
info "  Container:  $CONTAINER_NAME"
info "  Image:      mousedroid:jetson"
info "  Source:      $REMOTE_SRC (volume mount)"
info "  Config:     $CONFIG_DIR"
info "  Compose:    $COMPOSE_FILE"
echo ""
info "  Logs:       docker logs -f $CONTAINER_NAME"
info "  Shell:      docker exec -it $CONTAINER_NAME bash"
info "  Stop:       docker compose -f $COMPOSE_FILE down"
info "  Restart:    docker compose -f $COMPOSE_FILE restart"
