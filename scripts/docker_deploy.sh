#!/usr/bin/env bash
# =============================================================================
# MouseDroid — Docker Deploy Script for Jetson
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
#   --strict-health Promotion gate: FAIL (non-zero) on a dead telemetry
#                   endpoint, an observed ORT provider other than the
#                   configured primary, or a model-digest mismatch. Off by
#                   default — the default permissive behaviour is unchanged
#                   for bring-up flows (F-051 design D-10).
#   --help          Show this help message
#
# Environment variables (all optional, with defaults):
#   MOUSEDROID_INSTALL_DIR   Project install dir (default: /opt/mousedroid)
#   MOUSEDROID_CONFIG_DIR    Config file dir (default: /etc/mousedroid)
#   MOUSEDROID_COMPOSE_FILE  Compose file path (default: <install_dir>/docker-compose.jetson.yml)
#   MOUSEDROID_CONTAINER     Container name (default: mousedroid)
#   MOUSEDROID_HEALTH_PORT   Telemetry health port (default: 8080)
#   MOUSEDROID_HEALTH_TIMEOUT  Health check timeout secs (default: 30)
#   MOUSEDROID_STRICT_HEALTH   Set to 1/true for --strict-health without the flag
#   MOUSEDROID_DEPLOY_RECORD   Deploy record holding the expected model digest
#                              (default: <install_dir>/deployments/jetson-image.json)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configurable paths via environment variables
INSTALL_DIR="${MOUSEDROID_INSTALL_DIR:-/opt/mousedroid}"
CONFIG_DIR="${MOUSEDROID_CONFIG_DIR:-/etc/mousedroid}"
DOCKER_ENV_FILE="${MOUSEDROID_DOCKER_ENV_FILE:-${CONFIG_DIR}/docker.env}"

if [ -f "${DOCKER_ENV_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    . "${DOCKER_ENV_FILE}"
    set +a
fi

INSTALL_DIR="${MOUSEDROID_INSTALL_DIR:-/opt/mousedroid}"
CONFIG_DIR="${MOUSEDROID_CONFIG_DIR:-/etc/mousedroid}"
COMPOSE_FILE="${MOUSEDROID_COMPOSE_FILE:-${COMPOSE_FILE:-${INSTALL_DIR}/docker-compose.jetson.yml}}"
CONTAINER_NAME="${MOUSEDROID_CONTAINER:-mousedroid}"
HEALTH_PORT="${MOUSEDROID_HEALTH_PORT:-${MOUSEDROID_TELEMETRY_PORT:-8080}}"
HEALTH_PATH="${MOUSEDROID_HEALTH_PATH:-/api/v1/health}"
HEALTH_TIMEOUT="${MOUSEDROID_HEALTH_TIMEOUT:-30}"
DEPLOY_RECORD="${MOUSEDROID_DEPLOY_RECORD:-${INSTALL_DIR}/deployments/jetson-image.json}"

# Parse arguments
INSTALL_SERVICE=false
NO_BUILD=false
HEALTH_ONLY=false

# Strict health is OFF unless asked for, by flag or by env (same idiom as the
# knobs above). Any value other than 1/true/yes reads as off, so a typo fails
# open into today's behaviour rather than breaking a bring-up run.
case "${MOUSEDROID_STRICT_HEALTH:-}" in
    1|true|TRUE|yes|YES) STRICT_HEALTH=true ;;
    *)                   STRICT_HEALTH=false ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --service)    INSTALL_SERVICE=true; shift ;;
        --no-build)   NO_BUILD=true; shift ;;
        --health-only) HEALTH_ONLY=true; shift ;;
        --strict-health) STRICT_HEALTH=true; shift ;;
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
# Strict promotion probe (--strict-health only)
# ---------------------------------------------------------------------------
# Runs ONLY under --strict-health. Gating it entirely keeps the default path
# byte-identical to before this leg existed — no extra `docker exec`, no extra
# output, no new way for a bring-up run to fail (design D-10).
#
# Two questions, one exec, decided in Python where the config already lives
# rather than re-derived in shell:
#
#   provider — is the ORT execution provider ORT would actually select the
#              configured primary? The chain is DEFAULT_ORT_PROVIDERS
#              (common/onnx_session.py, TensorRT -> CUDA -> CPU) and the
#              selection is that module's own `resolve_providers`, so the probe
#              cannot drift from what the runtime does. A silent downgrade to
#              CUDA or CPU is exactly the failure this is here to catch.
#   digest   — does the ONNX artifact on the rover hash to the digest the
#              deploy record pins? Verification is
#              utils/weights_manager.verify_sha256, the same helper the
#              application uses, not a shell `sha256sum` re-implementation.
#
# Both are scoped by cfg.world_model.engine. With the default engine ("torch")
# there is no ONNX artifact in the 30 Hz path and nothing to prove, so the
# probe reports n/a and passes. Under engine="onnx_trt" it FAILS CLOSED: an
# absent expected digest, an unreadable record and a missing artifact are all
# failures, because "cannot tell" is not "fine" for a promotion gate.
#
# The record path is passed through as-is: /opt/mousedroid is bind-mounted at
# the same path inside the container, so a record under the install dir
# resolves identically. A record pointed outside that mount is unreadable to
# the probe and strict-health fails — deliberately, rather than skipping.
#
# The probe source is held in a variable rather than heredoc'd straight into the
# `docker exec`: a heredoc cannot be fed to a command inside `$(...)` command
# substitution (bash reports "unterminated here-document" and reads the body as
# script), and the exit status has to be captured, so the body goes in first and
# is piped in as a here-string.
read -r -d '' STRICT_PROBE_PY <<'PY' || true
import json
import os
import sys
from pathlib import Path

from mousedroid.common.onnx_session import DEFAULT_ORT_PROVIDERS, resolve_providers
from mousedroid.config.loader import load_settings
from mousedroid.utils.weights_manager import verify_sha256

record_path = Path(sys.argv[1])
overlay = os.environ.get("MOUSEDROID_CONFIG", "").strip()
cfg = load_settings(*([Path(overlay)] if overlay else []))
engine = cfg.world_model.engine
print(f"engine={engine}")

if engine != "onnx_trt":
    print("provider=n/a digest=n/a (engine is not onnx_trt: nothing to prove)")
    sys.exit(0)

problems: list[str] = []

expected_provider = DEFAULT_ORT_PROVIDERS[0]
try:
    import onnxruntime as ort
except ImportError:
    problems.append("onnxruntime is not importable but engine=onnx_trt")
else:
    active = resolve_providers(DEFAULT_ORT_PROVIDERS, tuple(ort.get_available_providers()))
    observed_provider = active[0] if active else ""
    print(f"provider expected={expected_provider} observed={observed_provider or 'none'}")
    if observed_provider != expected_provider:
        problems.append(
            f"ORT provider downgrade: expected {expected_provider}, "
            f"ORT would select {observed_provider or 'none'}"
        )

try:
    record = json.loads(record_path.read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    problems.append(f"deploy record unreadable ({type(exc).__name__}): {record_path}")
    record = None

if record is not None:
    expected_digest = record.get("model_sha256")
    if not expected_digest:
        problems.append(
            f"deploy record pins no model_sha256 ({record_path}) — "
            "cannot prove the artifact under engine=onnx_trt"
        )
    else:
        wm = cfg.world_model
        artifact = (
            Path(wm.onnx_path) if wm.onnx_path else Path(wm.onnx_cache_dir) / wm.onnx_filename
        )
        print(f"digest artifact={artifact}")
        if not verify_sha256(artifact, str(expected_digest), log_event_prefix="promotion_model"):
            problems.append(f"model digest mismatch or artifact missing: {artifact}")

for problem in problems:
    print(f"FAIL: {problem}", file=sys.stderr)
sys.exit(1 if problems else 0)
PY

strict_promotion_probe() {
    local probe_out
    if probe_out="$(docker exec -i "${CONTAINER_NAME}" python3 - "${DEPLOY_RECORD}" \
            <<<"${STRICT_PROBE_PY}" 2>&1)"; then
        echo "$probe_out" | sed 's/^/    /'
        info "  Strict promotion probe: OK"
        return 0
    fi
    error "  Strict promotion probe: FAILED"
    echo "$probe_out" | sed 's/^/    /' >&2
    return 1
}

# ---------------------------------------------------------------------------
# Health check function
# ---------------------------------------------------------------------------
health_check() {
    info "Running container health checks..."
    # Failure accumulator: the strict legs report EVERY problem before
    # returning, so one promotion run tells the operator the whole story
    # instead of one symptom per re-run.
    local failures=0

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
    #
    # The one leg whose severity depends on the mode. A dead endpoint during
    # bring-up is genuinely normal (telemetry disabled, or still starting), so
    # the default stays a warn, word for word. For a promotion it is a
    # release-blocking failure: an unreachable health endpoint means no
    # observability on the thing that just replaced a working rover.
    local health_url="http://127.0.0.1:${HEALTH_PORT}${HEALTH_PATH}"
    if curl -sf --max-time 5 "$health_url" >/dev/null 2>&1; then
        info "  Telemetry health endpoint: OK (${health_url})"
    elif [ "$STRICT_HEALTH" = true ]; then
        error "  Telemetry health endpoint: not responding (${health_url})"
        error "  --strict-health: a promotion requires a live telemetry endpoint"
        failures=$((failures + 1))
    else
        warn "  Telemetry health endpoint: not responding (${health_url})"
        warn "  (This is normal if telemetry is disabled or still starting)"
    fi

    # Provider + digest legs — strict mode only (see strict_promotion_probe).
    if [ "$STRICT_HEALTH" = true ]; then
        strict_promotion_probe || failures=$((failures + 1))
    fi

    # Check compose service status
    info "  Compose services:"
    docker compose -f "${COMPOSE_FILE}" ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null | \
        sed 's/^/    /' || true

    if [ "$failures" -gt 0 ]; then
        error "  ${failures} strict health check(s) failed"
        return 1
    fi
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
info "=== MouseDroid Docker Deployment ==="
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
# The softening is what --strict-health removes. Without the flag the call is
# unchanged: health_check's own `return 1` legs (container not running,
# mousedroid import broken) stay downgraded to a warn, exactly as before, so
# every existing bring-up flow behaves identically. With the flag a failed
# check aborts the run and leaves the prior release in place.
if [ "$STRICT_HEALTH" = true ]; then
    if ! health_check; then
        error "Health checks failed under --strict-health — promotion aborted"
        error "The previous release is still the running code; nothing was rolled forward."
        exit 1
    fi
else
    health_check || warn "Some health checks failed (non-fatal for deployment)"
fi

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
    if command -v systemd-analyze >/dev/null 2>&1; then
        info "  Verifying systemd unit"
        systemd-analyze verify /etc/systemd/system/mousedroid-docker.service
    fi
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
info "  Promote:    bash $0 --health-only --strict-health   # fails on provider/digest/telemetry"
if [ "$INSTALL_SERVICE" = true ]; then
    echo ""
    info "Systemd service:"
    info "  Status:     systemctl status mousedroid-docker"
    info "  Logs:       journalctl -u mousedroid-docker -f"
    info "  Restart:    systemctl restart mousedroid-docker"
fi
