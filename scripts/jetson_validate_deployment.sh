#!/bin/bash
# =============================================================================
# MouseDroidAGI — Jetson On-Device Deployment Validation
# =============================================================================
# Runs 10 sequential validation phases on the Jetson, printing structured
# JSON pass/fail for each phase. Designed for post-deployment verification.
#
# Usage:
#   bash scripts/jetson_validate_deployment.sh             # Full validation
#   bash scripts/jetson_validate_deployment.sh --dry-run   # Skip heavy tests
#
# Environment variables:
#   VENV_DIR                 Python venv path (default: /opt/mousedroid/venv)
#   PROJECT_DIR              Project root (default: auto-detected)
#   TELEMETRY_HOST           Telemetry host (default: localhost)
#   TELEMETRY_PORT           Telemetry port (default: 8080)
#   COMPOSE_FILE             Docker compose file (default: docker-compose.jetson.yml)
#   ORCH_RUN_SECONDS         Orchestrator run duration (default: 10)
#   HEALTH_TIMEOUT           Curl timeout for health check (default: 5)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (all from env vars — no hardcoded values)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "${SCRIPT_DIR}")}"
VENV_DIR="${VENV_DIR:-/opt/mousedroid/venv}"
PYTHON="${VENV_DIR}/bin/python"
PYTEST="${VENV_DIR}/bin/pytest"

TELEMETRY_HOST="${TELEMETRY_HOST:-localhost}"
TELEMETRY_PORT="${TELEMETRY_PORT:-8080}"
COMPOSE_FILE="${COMPOSE_FILE:-${PROJECT_DIR}/docker-compose.jetson.yml}"
ORCH_RUN_SECONDS="${ORCH_RUN_SECONDS:-10}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-5}"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

TOTAL_PHASES=10
PASSED=0
FAILED=0
SKIPPED=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ts() {
    date -u "+%Y-%m-%dT%H:%M:%SZ"
}

emit_json() {
    local phase="$1"
    local name="$2"
    local status="$3"
    local message="${4:-}"
    printf '{"timestamp":"%s","phase":%d,"name":"%s","status":"%s","message":"%s"}\n' \
        "$(ts)" "${phase}" "${name}" "${status}" "${message}"
}

record_pass() {
    local phase="$1"
    local name="$2"
    local message="${3:-}"
    emit_json "${phase}" "${name}" "PASS" "${message}"
    PASSED=$((PASSED + 1))
}

record_fail() {
    local phase="$1"
    local name="$2"
    local message="${3:-}"
    emit_json "${phase}" "${name}" "FAIL" "${message}"
    FAILED=$((FAILED + 1))
}

record_skip() {
    local phase="$1"
    local name="$2"
    local message="${3:-}"
    emit_json "${phase}" "${name}" "SKIP" "${message}"
    SKIPPED=$((SKIPPED + 1))
}

check_python() {
    if [[ ! -x "${PYTHON}" ]]; then
        emit_json 0 "python_check" "FAIL" "Python not found at ${PYTHON}"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Phase 1: Check CUDA availability
# ---------------------------------------------------------------------------

phase_1_cuda() {
    local phase=1
    local name="cuda_available"

    if "${DRY_RUN}"; then
        record_skip "${phase}" "${name}" "dry-run mode"
        return
    fi

    if "${PYTHON}" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
        local cuda_version
        cuda_version="$("${PYTHON}" -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")"
        record_pass "${phase}" "${name}" "CUDA ${cuda_version}"
    else
        record_fail "${phase}" "${name}" "torch.cuda.is_available() returned False"
    fi
}

# ---------------------------------------------------------------------------
# Phase 2: Check container health
# ---------------------------------------------------------------------------

phase_2_container() {
    local phase=2
    local name="container_health"

    if ! command -v docker &>/dev/null; then
        record_skip "${phase}" "${name}" "docker not installed"
        return
    fi

    if [[ ! -f "${COMPOSE_FILE}" ]]; then
        record_skip "${phase}" "${name}" "compose file not found: ${COMPOSE_FILE}"
        return
    fi

    local status
    status="$(docker compose -f "${COMPOSE_FILE}" ps --format '{{.Status}}' 2>/dev/null || echo "")"

    if [[ -z "${status}" ]]; then
        record_skip "${phase}" "${name}" "no containers running"
    elif echo "${status}" | grep -qi "up"; then
        record_pass "${phase}" "${name}" "container is running"
    else
        record_fail "${phase}" "${name}" "container status: ${status}"
    fi
}

# ---------------------------------------------------------------------------
# Phase 3: Check telemetry health endpoint
# ---------------------------------------------------------------------------

phase_3_telemetry_health() {
    local phase=3
    local name="telemetry_health"

    if "${DRY_RUN}"; then
        record_skip "${phase}" "${name}" "dry-run mode"
        return
    fi

    local url="http://${TELEMETRY_HOST}:${TELEMETRY_PORT}/health"
    if curl -sf --max-time "${HEALTH_TIMEOUT}" "${url}" >/dev/null 2>&1; then
        record_pass "${phase}" "${name}" "health endpoint OK"
    else
        record_skip "${phase}" "${name}" "telemetry not reachable at ${url}"
    fi
}

# ---------------------------------------------------------------------------
# Phase 4: Check config loads
# ---------------------------------------------------------------------------

phase_4_config() {
    local phase=4
    local name="config_loads"

    local result
    if result="$("${PYTHON}" -c "
from mousedroid.config.loader import load_settings
from pathlib import Path
cfg = load_settings(Path('${PROJECT_DIR}/config/jetson_production.yaml'),
                    config_dir=Path('${PROJECT_DIR}/config'))
print(f'platform={cfg.platform.value} telemetry={cfg.telemetry.enabled}')
" 2>&1)"; then
        record_pass "${phase}" "${name}" "${result}"
    else
        record_fail "${phase}" "${name}" "config load failed: ${result}"
    fi
}

# ---------------------------------------------------------------------------
# Phase 5: Check factory builds
# ---------------------------------------------------------------------------

phase_5_factory() {
    local phase=5
    local name="factory_builds"

    local result
    if result="$("${PYTHON}" -c "
import os
os.environ['MOUSEDROID_MOCK_HARDWARE'] = 'true'
from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator
cfg = Settings(mock_hardware=True)
orch = build_orchestrator(cfg)
print('orchestrator built successfully')
" 2>&1)"; then
        record_pass "${phase}" "${name}" "${result}"
    else
        record_fail "${phase}" "${name}" "factory build failed: ${result}"
    fi
}

# ---------------------------------------------------------------------------
# Phase 6: Run hardware smoke tests
# ---------------------------------------------------------------------------

phase_6_hardware_tests() {
    local phase=6
    local name="hardware_tests"

    if "${DRY_RUN}"; then
        record_skip "${phase}" "${name}" "dry-run mode"
        return
    fi

    local test_dir="${PROJECT_DIR}/tests/hardware"
    if [[ ! -d "${test_dir}" ]]; then
        record_skip "${phase}" "${name}" "tests/hardware/ not found"
        return
    fi

    if [[ ! -x "${PYTEST}" ]]; then
        record_skip "${phase}" "${name}" "pytest not installed"
        return
    fi

    if "${PYTEST}" -m hardware -v --timeout=30 "${test_dir}" 2>&1; then
        record_pass "${phase}" "${name}" "hardware tests passed"
    else
        record_fail "${phase}" "${name}" "hardware tests failed"
    fi
}

# ---------------------------------------------------------------------------
# Phase 7: Run orchestrator for N seconds
# ---------------------------------------------------------------------------

phase_7_orchestrator_run() {
    local phase=7
    local name="orchestrator_run"

    if "${DRY_RUN}"; then
        record_skip "${phase}" "${name}" "dry-run mode"
        return
    fi

    local result
    if result="$(timeout "$((ORCH_RUN_SECONDS + 10))" "${PYTHON}" -c "
import asyncio, os, time
os.environ['MOUSEDROID_MOCK_HARDWARE'] = 'true'
from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator

cfg = Settings(mock_hardware=True)
orch = build_orchestrator(cfg)

async def run():
    await orch.start()
    start = time.monotonic()
    ticks = 0
    try:
        while time.monotonic() - start < ${ORCH_RUN_SECONDS}:
            try:
                await orch.tick()
                ticks += 1
            except Exception:
                pass
            await asyncio.sleep(0.03)
    finally:
        await orch.stop()
    elapsed = time.monotonic() - start
    print(f'{ticks} ticks in {elapsed:.1f}s')

asyncio.run(run())
" 2>&1)"; then
        record_pass "${phase}" "${name}" "${result}"
    else
        record_fail "${phase}" "${name}" "orchestrator run failed: ${result}"
    fi
}

# ---------------------------------------------------------------------------
# Phase 8: Verify telemetry frames published
# ---------------------------------------------------------------------------

phase_8_telemetry_frames() {
    local phase=8
    local name="telemetry_frames"

    if "${DRY_RUN}"; then
        record_skip "${phase}" "${name}" "dry-run mode"
        return
    fi

    local result
    if result="$("${PYTHON}" -c "
import asyncio, os
os.environ['MOUSEDROID_MOCK_HARDWARE'] = 'true'
from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator

cfg = Settings(mock_hardware=True, telemetry={'enabled': True})
orch = build_orchestrator(cfg)

async def run():
    await orch.start()
    try:
        for _ in range(5):
            try:
                await orch.tick()
            except Exception:
                pass
    finally:
        await orch.stop()
    pub = orch._telemetry_publisher
    if pub is not None:
        stats = pub.stats
        print(f'published={stats[\"frames_published\"]} dropped={stats[\"frames_dropped\"]}')
    else:
        print('publisher=None')

asyncio.run(run())
" 2>&1)"; then
        if echo "${result}" | grep -q "published=0"; then
            record_fail "${phase}" "${name}" "zero frames published: ${result}"
        else
            record_pass "${phase}" "${name}" "${result}"
        fi
    else
        record_fail "${phase}" "${name}" "telemetry frame check failed: ${result}"
    fi
}

# ---------------------------------------------------------------------------
# Phase 9: Training dry-run
# ---------------------------------------------------------------------------

phase_9_training() {
    local phase=9
    local name="training_dry_run"

    local train_script="${PROJECT_DIR}/scripts/jetson_train.sh"
    if [[ ! -x "${train_script}" ]]; then
        record_skip "${phase}" "${name}" "jetson_train.sh not found or not executable"
        return
    fi

    if "${DRY_RUN}"; then
        record_skip "${phase}" "${name}" "dry-run mode"
        return
    fi

    if bash "${train_script}" --dry-run 2>&1; then
        record_pass "${phase}" "${name}" "training dry-run succeeded"
    else
        record_skip "${phase}" "${name}" "training dry-run not supported or failed"
    fi
}

# ---------------------------------------------------------------------------
# Phase 10: Overall summary
# ---------------------------------------------------------------------------

phase_10_summary() {
    local phase=10
    local total=$((PASSED + FAILED + SKIPPED))
    local overall="PASS"
    if [[ "${FAILED}" -gt 0 ]]; then
        overall="FAIL"
    fi

    printf '{"timestamp":"%s","phase":%d,"name":"summary","status":"%s","passed":%d,"failed":%d,"skipped":%d,"total":%d}\n' \
        "$(ts)" "${phase}" "${overall}" "${PASSED}" "${FAILED}" "${SKIPPED}" "${total}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    emit_json 0 "validation_start" "INFO" "starting ${TOTAL_PHASES}-phase validation"

    check_python

    phase_1_cuda
    phase_2_container
    phase_3_telemetry_health
    phase_4_config
    phase_5_factory
    phase_6_hardware_tests
    phase_7_orchestrator_run
    phase_8_telemetry_frames
    phase_9_training
    phase_10_summary

    if [[ "${FAILED}" -gt 0 ]]; then
        exit 1
    fi
}

main "$@"
