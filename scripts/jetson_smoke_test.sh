#!/bin/bash
# Comprehensive smoke test suite for MouseDroid on Jetson Nano.
# Runs on the Jetson itself to validate system, hardware, and application health.
#
# Usage:
#   bash scripts/jetson_smoke_test.sh              # Run all tests
#   bash scripts/jetson_smoke_test.sh system        # Run only system health tests
#   bash scripts/jetson_smoke_test.sh gpio          # Run only GPIO tests
#   bash scripts/jetson_smoke_test.sh serial        # Run only serial tests
#   bash scripts/jetson_smoke_test.sh motor         # Run only motor loopback smoke
#   bash scripts/jetson_smoke_test.sh camera        # Run only camera tests
#   bash scripts/jetson_smoke_test.sh lidar         # Run only LiDAR tests
#   bash scripts/jetson_smoke_test.sh speaker       # Run only speaker tests
#   bash scripts/jetson_smoke_test.sh pcie_ssd      # Run only NVMe SSD on PCIe smoke
#   bash scripts/jetson_smoke_test.sh hailo         # Run only Hailo-8 accelerator smoke
#   bash scripts/jetson_smoke_test.sh app           # Run only application health check
#   bash scripts/jetson_smoke_test.sh pytest        # Run only hardware pytest suite
#   bash scripts/jetson_smoke_test.sh e2e           # Run only E2E 5-second run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
VENV_DIR="${VENV_DIR:-/opt/mousedroid/venv}"
PYTHON=""
FAILURES=0
PASSES=0
SKIPS=0
RESULTS=()
ALLOW_MOTION="${MOUSEDROID_SMOKE_ALLOW_MOTION:-0}"
CONFIGS_CSV="${MOUSEDROID_JETSON_CONFIGS:-}"
declare -a CONFIG_ARGS=()

if [[ -n "${CONFIGS_CSV}" ]]; then
    IFS=',' read -r -a RAW_CONFIG_PATHS <<< "${CONFIGS_CSV}"
    for cfg in "${RAW_CONFIG_PATHS[@]}"; do
        cfg="${cfg#"${cfg%%[![:space:]]*}"}"
        cfg="${cfg%"${cfg##*[![:space:]]}"}"
        if [[ -n "${cfg}" ]]; then
            if [[ ${#CONFIG_ARGS[@]} -eq 0 ]]; then
                CONFIG_ARGS=(--config)
            fi
            CONFIG_ARGS+=("${cfg}")
        fi
    done
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ts() {
    date "+%Y-%m-%d %H:%M:%S"
}

log_section() {
    echo ""
    echo "=== $1 === [$(ts)]"
}

log_step() {
    echo "--- $1 ---"
}

record_pass() {
    local name="$1"
    echo "  PASS: ${name}"
    PASSES=$((PASSES + 1))
    RESULTS+=("PASS: ${name}")
}

record_fail() {
    local name="$1"
    local reason="${2:-}"
    echo "  FAIL: ${name}"
    if [[ -n "${reason}" ]]; then
        echo "        ${reason}"
    fi
    FAILURES=$((FAILURES + 1))
    RESULTS+=("FAIL: ${name} ${reason}")
}

record_skip() {
    local name="$1"
    local reason="${2:-}"
    echo "  SKIP: ${name}"
    if [[ -n "${reason}" ]]; then
        echo "        ${reason}"
    fi
    SKIPS=$((SKIPS + 1))
    RESULTS+=("SKIP: ${name} ${reason}")
}

# ---------------------------------------------------------------------------
# Host detection + tunable paths
# ---------------------------------------------------------------------------
#
# Mirrors `tests/_jetson_hardware.py::is_jetson_host` so bash + python agree:
# a Jetson host is Linux with /etc/nv_tegra_release present (existence — NOT
# readability — to match Python's `Path(...).exists()`; the script never reads
# the contents of that file, only uses presence as a marker).
#
# Override for tests / dev-box debugging:
#   MOUSEDROID_SMOKE_FORCE_PLATFORM=jetson      -> always treat as Jetson
#   MOUSEDROID_SMOKE_FORCE_PLATFORM=non-jetson  -> always treat as non-Jetson
#   MOUSEDROID_SMOKE_FORCE_PLATFORM=auto        -> use real detection (default)
#   (unset and empty-string both map to "auto"; unknown values warn + fall through)
#
# Optional path overrides for tests / vendor builds where Jetson paths differ:
#   MOUSEDROID_SMOKE_TEGRA_RELEASE_PATH=/path/to/marker  (default: /etc/nv_tegra_release)
#   MOUSEDROID_SMOKE_MEMINFO_PATH=/path/to/meminfo       (default: /proc/meminfo)
#   MOUSEDROID_SMOKE_THERMAL_PATH=/path/to/temp          (default: /sys/devices/virtual/thermal/thermal_zone0/temp)

# Resolve the configured tegra-release marker once, exposed so test_system can
# log the actually-checked path (rather than hardcoding /etc/nv_tegra_release
# in the human-readable echo, which would mis-report when an override is set).
_jetson_tegra_release_path() {
    printf '%s' "${MOUSEDROID_SMOKE_TEGRA_RELEASE_PATH:-/etc/nv_tegra_release}"
}

is_jetson_host() {
    local override="${MOUSEDROID_SMOKE_FORCE_PLATFORM:-auto}"
    # Treat empty-string the same as unset to avoid surprising users who
    # `export MOUSEDROID_SMOKE_FORCE_PLATFORM=` to "clear" it.
    if [[ -z "${override}" ]]; then
        override="auto"
    fi
    case "${override}" in
        jetson)     return 0 ;;
        non-jetson) return 1 ;;
        auto)       ;;
        *)
            echo "WARN: ignoring unknown MOUSEDROID_SMOKE_FORCE_PLATFORM='${override}' (use jetson|non-jetson|auto)" >&2
            ;;
    esac

    local tegra_release_path
    tegra_release_path="$(_jetson_tegra_release_path)"
    [[ "$(uname -s)" == "Linux" && -e "${tegra_release_path}" ]]
}

resolve_python() {
    local candidate=""

    if [[ -n "${MOUSEDROID_SMOKE_PYTHON:-}" ]]; then
        if [[ -x "${MOUSEDROID_SMOKE_PYTHON}" ]]; then
            PYTHON="${MOUSEDROID_SMOKE_PYTHON}"
            log_step "Using Python runtime from MOUSEDROID_SMOKE_PYTHON: ${PYTHON}"
            return
        fi

        candidate="$(command -v "${MOUSEDROID_SMOKE_PYTHON}" 2>/dev/null || true)"
        if [[ -n "${candidate}" ]]; then
            PYTHON="${candidate}"
            log_step "Using Python runtime from MOUSEDROID_SMOKE_PYTHON: ${PYTHON}"
            return
        fi

        echo "WARN: MOUSEDROID_SMOKE_PYTHON='${MOUSEDROID_SMOKE_PYTHON}' is not executable; falling back"
    fi

    if [[ -x "${VENV_DIR}/bin/python" ]]; then
        PYTHON="${VENV_DIR}/bin/python"
        log_step "Using Python runtime from VENV_DIR: ${PYTHON}"
        return
    fi

    candidate="$(command -v python3 2>/dev/null || true)"
    if [[ -n "${candidate}" ]]; then
        PYTHON="${candidate}"
        log_step "Using fallback Python runtime from PATH: ${PYTHON}"
        return
    fi

    echo "ERROR: No Python runtime found for jetson_smoke_test.sh"
    echo "Checked MOUSEDROID_SMOKE_PYTHON, ${VENV_DIR}/bin/python, and python3 on PATH."
    exit 1
}

check_python() {
    resolve_python
}

# ---------------------------------------------------------------------------
# 1. System health
# ---------------------------------------------------------------------------

test_system() {
    log_section "System Health"

    local tegra_path
    tegra_path="$(_jetson_tegra_release_path)"
    local force_platform="${MOUSEDROID_SMOKE_FORCE_PLATFORM:-}"
    [[ -z "${force_platform}" ]] && force_platform="auto"
    local on_jetson=0
    if is_jetson_host; then
        on_jetson=1
        case "${force_platform}" in
            jetson)
                echo "  Host treated as Jetson (forced via MOUSEDROID_SMOKE_FORCE_PLATFORM=jetson; uname / ${tegra_path} were not checked)"
                ;;
            *)
                echo "  Host detected as Jetson (Linux + ${tegra_path} present)"
                ;;
        esac
    else
        case "${force_platform}" in
            non-jetson)
                echo "  Host treated as NOT a Jetson (forced via MOUSEDROID_SMOKE_FORCE_PLATFORM=non-jetson); Jetson-only checks will be SKIPped"
                ;;
            *)
                echo "  Host is NOT a Jetson (${tegra_path} absent or non-Linux); Jetson-only checks will be SKIPped"
                ;;
        esac
    fi

    # CUDA / torch  --  Jetson-only (CI/dev-box may legitimately lack CUDA wheels)
    log_step "Checking torch.cuda.is_available()"
    if [[ "${on_jetson}" -eq 0 ]]; then
        record_skip "torch.cuda.is_available" "not running on a Jetson host"
    elif "${PYTHON}" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
        record_pass "torch.cuda.is_available"
    else
        record_fail "torch.cuda.is_available" "CUDA GPU not detected by PyTorch"
    fi

    # TensorRT  --  Jetson-only
    log_step "Checking TensorRT import"
    if [[ "${on_jetson}" -eq 0 ]]; then
        record_skip "import tensorrt" "not running on a Jetson host"
    elif "${PYTHON}" -c "import tensorrt; print(f'TensorRT {tensorrt.__version__}')" 2>/dev/null; then
        record_pass "import tensorrt"
    else
        record_fail "import tensorrt" "TensorRT not importable"
    fi

    # Thermal sensor  --  Jetson-only (path is tegra-specific)
    log_step "Reading thermal sensor"
    local thermal_path="${MOUSEDROID_SMOKE_THERMAL_PATH:-/sys/devices/virtual/thermal/thermal_zone0/temp}"
    if [[ "${on_jetson}" -eq 0 ]]; then
        record_skip "thermal sensor read" "not running on a Jetson host"
    elif [[ -r "${thermal_path}" ]]; then
        local temp_raw
        temp_raw="$(cat "${thermal_path}")"
        local temp_c
        temp_c="$(echo "scale=1; ${temp_raw} / 1000" | bc 2>/dev/null || echo "unknown")"
        echo "  GPU temperature: ${temp_c} C"
        record_pass "thermal sensor read (${temp_c} C)"
    else
        record_fail "thermal sensor read" "${thermal_path} not readable"
    fi

    # Memory check  --  needs Linux /proc; SKIP cleanly on non-Linux dev boxes
    log_step "Checking available memory"
    local meminfo_path="${MOUSEDROID_SMOKE_MEMINFO_PATH:-/proc/meminfo}"
    if [[ ! -r "${meminfo_path}" ]]; then
        record_skip "memory check" "${meminfo_path} not readable (non-Linux host?)"
    else
        local mem_total_kb mem_avail_kb mem_used_pct mem_source="MemAvailable"
        mem_total_kb="$(awk '/^MemTotal:/ {print $2; exit}' "${meminfo_path}" 2>/dev/null || true)"
        mem_avail_kb="$(awk '/^MemAvailable:/ {print $2; exit}' "${meminfo_path}" 2>/dev/null || true)"
        # Fallback for kernels < 3.14 or non-Linux emulation layers that lack MemAvailable.
        if [[ -z "${mem_avail_kb}" ]]; then
            mem_avail_kb="$(awk '/^MemFree:/ {print $2; exit}' "${meminfo_path}" 2>/dev/null || true)"
            mem_source="MemFree"
            echo "WARN: MemAvailable not present in ${meminfo_path}; falling back to MemFree (kernel < 3.14 or emulated /proc?)" >&2
        fi
        if [[ -n "${mem_total_kb}" && -n "${mem_avail_kb}" \
                && "${mem_total_kb}" =~ ^[0-9]+$ && "${mem_avail_kb}" =~ ^[0-9]+$ \
                && "${mem_total_kb}" -gt 0 ]]; then
            mem_used_pct="$(( (mem_total_kb - mem_avail_kb) * 100 / mem_total_kb ))"
            echo "  Memory: ${mem_used_pct}% used (${mem_avail_kb}kB ${mem_source} of ${mem_total_kb}kB)"
            if [[ "${mem_used_pct}" -lt 95 ]]; then
                record_pass "memory check (${mem_used_pct}% used)"
            else
                record_fail "memory check" "Memory usage critical: ${mem_used_pct}%"
            fi
        else
            record_fail "memory check" "Could not parse ${meminfo_path}"
        fi
    fi
}

# ---------------------------------------------------------------------------
# 1b. USB-C enumeration (config-driven, gated on usbc_discovery.enabled)
# ---------------------------------------------------------------------------

test_usbc() {
    log_section "USB-C Enumeration"
    log_step "Running scripts/check_usbc_devices.py"

    # check_usbc_devices.py auto-resolves overlays via resolve_runtime_config_paths()
    # when --config is omitted, so we no longer need a skip-if-empty guard
    # here (which previously caused the blocking-stage silent-bypass that
    # CodeRabbit flagged: record_skip → return 0 → wrapper sees PASS even
    # when nothing was actually checked).
    local output rc
    set +e
    if [[ ${#CONFIG_ARGS[@]} -gt 0 ]]; then
        output="$("${PYTHON}" "${PROJECT_DIR}/scripts/check_usbc_devices.py" "${CONFIG_ARGS[@]}" 2>&1)"
    else
        output="$("${PYTHON}" "${PROJECT_DIR}/scripts/check_usbc_devices.py" 2>&1)"
    fi
    rc=$?
    set -e

    echo "${output}"
    if [[ ${rc} -eq 0 ]]; then
        record_pass "usbc enumeration"
    else
        record_fail "usbc enumeration" "missing required endpoint(s) (see above)"
    fi
}

# ---------------------------------------------------------------------------
# 2. GPIO
# ---------------------------------------------------------------------------

test_gpio() {
    log_section "GPIO Test"
    log_step "Testing configured ultrasonic GPIO pins"

    local gpio_script
    gpio_script=$(cat <<'PYEOF'
import sys

from mousedroid.validation.runtime import load_runtime_settings

try:
    import Jetson.GPIO as GPIO
except ImportError:
    print("SKIP:Jetson.GPIO not available")
    sys.exit(0)

cfg = load_runtime_settings()
if cfg.ultrasonic is None:
    print("SKIP:Ultrasonic disabled in config")
    sys.exit(0)

trigger_pin = cfg.ultrasonic.trigger_pin
echo_pin = cfg.ultrasonic.echo_pin

try:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(trigger_pin, GPIO.OUT)
    GPIO.setup(echo_pin, GPIO.IN)
    print(f"PASS:GPIO pins {trigger_pin}/{echo_pin} setup and teardown OK")
finally:
    GPIO.cleanup()
PYEOF
    )

    local result
    result="$("${PYTHON}" -c "${gpio_script}" 2>&1)" || true

    if echo "${result}" | grep -q "^PASS:"; then
        record_pass "GPIO pin setup/teardown"
    elif echo "${result}" | grep -q "^SKIP:"; then
        record_skip "GPIO pin setup/teardown" "Jetson.GPIO not installed"
    else
        record_fail "GPIO pin setup/teardown" "${result}"
    fi
}

# ---------------------------------------------------------------------------
# 3. Serial
# ---------------------------------------------------------------------------

test_serial() {
    log_section "Serial Port Test"
    log_step "Testing configured ESP32 serial port"

    local serial_script
    serial_script=$(cat <<'PYEOF'
import json
import sys
import time
from pathlib import Path

from mousedroid.validation.runtime import load_runtime_settings

try:
    import serial
except ImportError:
    print("SKIP:pyserial not installed")
    sys.exit(0)

cfg = load_runtime_settings()
serial_port = Path(cfg.esp32.serial_port)
baud_rate = cfg.esp32.serial_baud

if not serial_port.exists():
    print(f"FAIL:Serial port {serial_port} not found")
    sys.exit(0)

try:
    ser = serial.Serial(str(serial_port), baud_rate, timeout=2)
    time.sleep(0.1)  # Allow port to stabilise
    cmd = json.dumps({"T": 0}) + "\n"
    ser.write(cmd.encode())
    ser.flush()
    response = ser.readline().decode().strip()
    ser.close()
    if response:
        print(f"PASS:Got response: {response}")
    else:
        print("WARN:No response from ESP32 (timeout) -- device may not be running")
except serial.SerialException as exc:
    print(f"FAIL:Serial error: {exc}")
except Exception as exc:
    print(f"FAIL:Unexpected error: {exc}")
PYEOF
    )

    local result
    result="$("${PYTHON}" -c "${serial_script}" 2>&1)" || true

    if echo "${result}" | grep -q "^PASS:"; then
        record_pass "serial send/receive"
    elif echo "${result}" | grep -q "^SKIP:"; then
        record_skip "serial send/receive" "pyserial not installed"
    elif echo "${result}" | grep -q "^WARN:"; then
        local msg
        msg="$(echo "${result}" | grep "^WARN:" | sed 's/^WARN://')"
        record_pass "serial open/write (no response: ${msg})"
    else
        local msg
        msg="$(echo "${result}" | grep "^FAIL:" | sed 's/^FAIL://')"
        record_fail "serial send/receive" "${msg}"
    fi
}

# ---------------------------------------------------------------------------
# 3b. Motor loopback smoke
# ---------------------------------------------------------------------------

test_motor() {
    log_section "Motor Loopback Smoke"
    log_step "Running ESP32 loopback motor smoke test"
    local pytest_hint="rebuild the Jetson container with docker compose -f docker-compose.jetson.yml build mousedroid"

    if [[ "${ALLOW_MOTION}" != "1" ]]; then
        record_skip "motor loopback smoke" "motion disabled; set MOUSEDROID_SMOKE_ALLOW_MOTION=1"
        return
    fi

    if ! "${PYTHON}" -m pytest --version >/dev/null 2>&1; then
        if [[ "${ALLOW_MOTION}" == "1" ]]; then
            record_fail "motor loopback smoke" "pytest not available in selected Python runtime; ${pytest_hint}"
        else
            record_skip "motor loopback smoke" "pytest not available in selected Python runtime"
        fi
        return
    fi

    local test_file="${PROJECT_DIR}/tests/hardware/test_esp32_loopback.py"
    if [[ ! -f "${test_file}" ]]; then
        if [[ "${ALLOW_MOTION}" == "1" ]]; then
            record_fail "motor loopback smoke" "tests/hardware/test_esp32_loopback.py not found"
        else
            record_skip "motor loopback smoke" "tests/hardware/test_esp32_loopback.py not found"
        fi
        return
    fi

    local pytest_output
    if pytest_output="$(MOUSEDROID_JETSON_CONFIGS="${CONFIGS_CSV}" MOUSEDROID_MOCK_HARDWARE=false "${PYTHON}" -m pytest -m hardware -ra -v "${test_file}" 2>&1)"; then
        echo "${pytest_output}"
        if ! echo "${pytest_output}" | grep -q "test_send_velocity_moves_encoders PASSED"; then
            local motor_reason
            motor_reason="$(echo "${pytest_output}" | grep -m1 "encoder loopback inactive" | sed 's/^.*encoder loopback inactive/encoder loopback inactive/' || true)"
            if [[ -n "${motor_reason}" ]]; then
                record_fail "motor loopback smoke" "${motor_reason}"
            else
                record_fail "motor loopback smoke" "encoder loopback test did not pass"
            fi
            return
        fi
        if ! echo "${pytest_output}" | grep -q "test_emergency_stop_latency PASSED"; then
            record_fail "motor loopback smoke" "emergency stop latency test did not pass"
            return
        fi
        record_pass "motor loopback smoke"
    else
        echo "${pytest_output}"
        local failed_count
        failed_count="$(echo "${pytest_output}" | grep -oP '\d+ failed' | grep -oP '\d+' || echo "?")"
        record_fail "motor loopback smoke" "${failed_count} test(s) failed"
    fi
}

# ---------------------------------------------------------------------------
# 3c. Power chain smoke (battery + zero-vel + e-stop within budget)
# ---------------------------------------------------------------------------

test_power() {
    log_section "Power Chain Smoke"
    log_step "Running power-chain hardware test"
    local test_file="${PROJECT_DIR}/tests/hardware/test_power_chain_smoke.py"
    if [[ ! -f "${test_file}" ]]; then
        record_skip "power chain smoke" "test_power_chain_smoke.py not found"
        return
    fi

    if ! "${PYTHON}" -m pytest --version >/dev/null 2>&1; then
        record_skip "power chain smoke" "pytest not available in selected Python runtime"
        return
    fi

    local pytest_output
    if pytest_output="$(MOUSEDROID_JETSON_CONFIGS="${CONFIGS_CSV}" MOUSEDROID_MOCK_HARDWARE=false \
            "${PYTHON}" -m pytest -m hardware -ra -v "${test_file}" 2>&1)"; then
        echo "${pytest_output}"
        record_pass "power chain smoke"
    else
        echo "${pytest_output}"
        record_fail "power chain smoke" "see output for battery/e-stop violation"
    fi
}

# ---------------------------------------------------------------------------
# 4. Camera / 5. Audio -- delegate to scripts/verify_sensors.py
# ---------------------------------------------------------------------------

test_camera() {
    _run_verify_sensor "camera" "Camera"
}

test_audio() {
    _run_verify_sensor "audio" "Audio"
}

# ---------------------------------------------------------------------------
# 5a. LiDAR / 5b. Speaker / 5c. Rocky voice -- delegate to verify_sensors.py
# ---------------------------------------------------------------------------

_run_verify_sensor() {
    # $1 = sensor name (lidar|speaker|voice), $2 = record label
    local sensor="$1"
    local label="$2"
    log_section "${label} Test"
    log_step "Running verify_sensors.py --sensor ${sensor}"

    local output rc
    set +e
    output="$(MOUSEDROID_MOCK_HARDWARE=false MOUSEDROID_JETSON_CONFIGS="${CONFIGS_CSV}" "${PYTHON}" "${PROJECT_DIR}/scripts/verify_sensors.py" "${CONFIG_ARGS[@]}" --sensor "${sensor}" 2>&1)"
    rc=$?
    set -e

    echo "${output}"

    if [[ ${rc} -eq 0 ]]; then
        if echo "${output}" | grep -q "\[SKIP\]"; then
            record_skip "${label}" "device not detected (see verify_sensors output)"
        else
            record_pass "${label}"
        fi
    else
        local first_fail
        first_fail="$(echo "${output}" | grep -m1 "\[FAIL\]" || true)"
        record_fail "${label}" "${first_fail:-non-zero exit (${rc})}"
    fi
}

test_lidar() {
    _run_verify_sensor "lidar" "LiDAR"
}

test_speaker() {
    _run_verify_sensor "speaker" "Speaker"
}

test_voice() {
    _run_verify_sensor "voice" "Rocky Voice"
}

# ---------------------------------------------------------------------------
# 5d. PCIe NVMe SSD / 5e. Hailo-8 accelerator -- delegate to verify_sensors.py
# ---------------------------------------------------------------------------

test_pcie_ssd() {
    _run_verify_sensor "pcie_ssd" "PCIe NVMe SSD"
}

test_hailo() {
    _run_verify_sensor "hailo" "Hailo-8"
}

# ---------------------------------------------------------------------------
# 6. Application health check
# ---------------------------------------------------------------------------

test_app() {
    log_section "Application Health Check"
    log_step "Running mousedroid.main --health-check"

    if "${PYTHON}" -m mousedroid.main "${CONFIG_ARGS[@]}" --health-check 2>&1; then
        record_pass "mousedroid --health-check"
    else
        record_fail "mousedroid --health-check" "Non-zero exit code"
    fi
}

# ---------------------------------------------------------------------------
# 7. Hardware pytest
# ---------------------------------------------------------------------------

test_pytest() {
    log_section "Hardware Pytest Suite"
    log_step "Running pytest -m hardware"
    local pytest_hint="rebuild the Jetson container with docker compose -f docker-compose.jetson.yml build mousedroid"

    if ! "${PYTHON}" -m pytest --version >/dev/null 2>&1; then
        if [[ -n "${MOUSEDROID_SMOKE_PYTHON:-}" ]]; then
            record_fail "hardware pytest suite" "pytest not available in selected Python runtime; ${pytest_hint}"
        else
            record_skip "hardware pytest" "pytest not available in selected Python runtime"
        fi
        return
    fi

    local test_dir="${PROJECT_DIR}/tests/hardware"
    if [[ ! -d "${test_dir}" ]]; then
        record_skip "hardware pytest" "tests/hardware/ directory not found"
        return
    fi

    local pytest_output
    if pytest_output="$(MOUSEDROID_JETSON_CONFIGS="${CONFIGS_CSV}" MOUSEDROID_MOCK_HARDWARE=false "${PYTHON}" -m pytest -m hardware -v "${test_dir}" 2>&1)"; then
        echo "${pytest_output}"
        record_pass "hardware pytest suite"
    else
        echo "${pytest_output}"
        local failed_count
        failed_count="$(echo "${pytest_output}" | grep -oP '\d+ failed' | grep -oP '\d+' || echo "?")"
        record_fail "hardware pytest suite" "${failed_count} test(s) failed"
    fi
}

# ---------------------------------------------------------------------------
# 8. E2E 5-second run
# ---------------------------------------------------------------------------

test_e2e() {
    log_section "E2E 5-Second Run"
    log_step "Starting orchestrator for 5 seconds, then SIGINT shutdown"

    local e2e_script
    e2e_script=$(cat <<'PYEOF'
import asyncio
import signal
import sys
import time

# Load config
from mousedroid.config.loader import load_settings
from mousedroid.factory import build_orchestrator
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.validation.runtime import camera_unavailable_reason, resolve_runtime_config_paths

cfg = load_settings(*resolve_runtime_config_paths())

orch = build_orchestrator(cfg)
if not isinstance(orch, MouseDroidOrchestrator):
    # Explicit raise (not assert) — PYTHONOPTIMIZE=1 on the Jetson Docker
    # entrypoint strips asserts, which would silently turn a wrong-type
    # return into a spurious E2E PASS. Per CLAUDE.md validation contract.
    raise RuntimeError(
        f"build_orchestrator returned {type(orch).__name__}, expected MouseDroidOrchestrator"
    )

async def run_e2e():
    await orch.start()
    start = time.monotonic()
    try:
        # Run tick loop for ~5 seconds
        while time.monotonic() - start < 5.0:
            await orch.tick()
            await asyncio.sleep(0.1)
    finally:
        await orch.stop()

    elapsed = time.monotonic() - start
    print(f"PASS:E2E ran for {elapsed:.1f}s, clean shutdown")

try:
    asyncio.run(run_e2e())
except KeyboardInterrupt:
    print("PASS:E2E interrupted cleanly via SIGINT")
except Exception as exc:
    reason = camera_unavailable_reason(cfg, exc)
    if reason is not None:
        print(f"SKIP:E2E blocked by unavailable camera: {reason}")
        sys.exit(0)
    print(f"FAIL:E2E error: {exc}")
    sys.exit(1)
PYEOF
    )

    local result
    # Run with a 15s timeout and send SIGINT after 5s if still running
    result="$(timeout --signal=INT 15 "${PYTHON}" -c "${e2e_script}" 2>&1)" || true

    if echo "${result}" | grep -q "^PASS:"; then
        local msg
        msg="$(echo "${result}" | grep "^PASS:" | tail -1 | sed 's/^PASS://')"
        record_pass "E2E 5-second run: ${msg}"
    elif echo "${result}" | grep -q "^SKIP:"; then
        local msg
        msg="$(echo "${result}" | grep "^SKIP:" | tail -1 | sed 's/^SKIP://')"
        record_skip "E2E 5-second run" "${msg}"
    else
        local msg
        msg="$(echo "${result}" | grep "^FAIL:" | sed 's/^FAIL://')"
        record_fail "E2E 5-second run" "${msg:-unknown error}"
    fi
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print_summary() {
    log_section "Smoke Test Summary"
    echo ""
    for r in "${RESULTS[@]}"; do
        echo "  ${r}"
    done
    echo ""
    echo "  Passed:  ${PASSES}"
    echo "  Failed:  ${FAILURES}"
    echo "  Skipped: ${SKIPS}"
    echo ""
    if [[ "${FAILURES}" -eq 0 ]]; then
        echo "=== All smoke tests passed ==="
    else
        echo "=== ${FAILURES} smoke test(s) FAILED ==="
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    local step="${1:-all}"

    log_section "MouseDroid Jetson Smoke Tests"
    check_python

    case "${step}" in
        all)
            test_system
            test_usbc
            test_gpio
            test_serial
            test_motor
            test_power
            test_camera
            test_audio
            test_lidar
            test_speaker
            test_voice
            test_pcie_ssd
            test_hailo
            test_app
            test_pytest
            test_e2e
            ;;
        system)   test_system ;;
        usbc)     test_usbc ;;
        gpio)     test_gpio ;;
        serial)   test_serial ;;
        motor)    test_motor ;;
        power)    test_power ;;
        camera)   test_camera ;;
        audio)    test_audio ;;
        lidar)    test_lidar ;;
        speaker)  test_speaker ;;
        voice)    test_voice ;;
        pcie_ssd|ssd) test_pcie_ssd ;;
        hailo)    test_hailo ;;
        app)      test_app ;;
        pytest)   test_pytest ;;
        e2e)      test_e2e ;;
        *)
            echo "Unknown step: ${step}"
            echo "Valid steps: all, system, usbc, gpio, serial, motor, power, camera, audio, lidar, speaker, voice, pcie_ssd, hailo, app, pytest, e2e"
            exit 1
            ;;
    esac

    print_summary
    exit "${FAILURES}"
}

main "$@"
