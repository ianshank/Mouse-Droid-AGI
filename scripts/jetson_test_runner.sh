#!/usr/bin/env bash
# =============================================================================
# MouseDroid — Jetson Container Test Runner
# =============================================================================
# Runs the test suite inside the Docker container with GPU access.
#
# Usage:
#   bash scripts/jetson_test_runner.sh              # Full suite
#   bash scripts/jetson_test_runner.sh unit          # Unit tests only
#   bash scripts/jetson_test_runner.sh integration   # Integration only
#   bash scripts/jetson_test_runner.sh gpu           # GPU tests only
# =============================================================================
set -euo pipefail

CONTAINER_NAME="${MOUSEDROID_CONTAINER:-mousedroid}"
SRC_DIR="/opt/mousedroid"
CATEGORY="${1:-all}"

# Colours
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if ! docker ps --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
    error "Container '${CONTAINER_NAME}' is not running."
    error "Start it with: docker compose -f docker-compose.jetson.yml up -d"
    exit 1
fi

# ---------------------------------------------------------------------------
# GPU smoke test
# ---------------------------------------------------------------------------
info "=== GPU Smoke Test ==="
docker exec "${CONTAINER_NAME}" python3 -c "
import torch
print(f'  torch={torch.__version__}')
print(f'  CUDA={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU={torch.cuda.get_device_name(0)}')
    print(f'  CUDA version={torch.version.cuda}')
"

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
case "${CATEGORY}" in
    unit)
        info "=== Running Unit Tests ==="
        docker exec -w "${SRC_DIR}" "${CONTAINER_NAME}" \
            python3 -m pytest tests/unit -v --tb=short \
                --cov=src/mousedroid --cov-report=term-missing
        ;;
    integration)
        info "=== Running Integration Tests ==="
        docker exec -w "${SRC_DIR}" -e MOUSEDROID_MOCK_HARDWARE=true "${CONTAINER_NAME}" \
            python3 -m pytest tests/integration -v --tb=short
        ;;
    gpu)
        info "=== Running GPU Tests ==="
        docker exec -w "${SRC_DIR}" "${CONTAINER_NAME}" \
            python3 -m pytest tests/integration/test_docker_gpu.py -v --tb=short
        ;;
    lint)
        info "=== Running Lint ==="
        docker exec -w "${SRC_DIR}" "${CONTAINER_NAME}" ruff check src/ tests/
        docker exec -w "${SRC_DIR}" "${CONTAINER_NAME}" ruff format --check src/ tests/
        ;;
    typecheck)
        info "=== Running Type Check ==="
        docker exec -w "${SRC_DIR}" "${CONTAINER_NAME}" \
            mypy src/ --strict --ignore-missing-imports
        ;;
    all)
        info "=== Running Full Suite ==="
        docker exec -w "${SRC_DIR}" -e MOUSEDROID_MOCK_HARDWARE=true "${CONTAINER_NAME}" \
            python3 -m pytest tests/unit tests/property tests/integration \
                -v --tb=short \
                --cov=src/mousedroid \
                --cov-report=term-missing \
                -x
        ;;
    *)
        error "Unknown category: ${CATEGORY}"
        echo "Usage: $0 {unit|integration|gpu|lint|typecheck|all}"
        exit 1
        ;;
esac

info "=== Tests Complete ==="
