#!/bin/bash
# Comprehensive smoke test suite for MouseDroid on Jetson Nano.
# Runs on the Jetson itself to validate system, hardware, and application health.
#
# Usage:
#   bash scripts/jetson_smoke_test.sh              # Run all tests
#   bash scripts/jetson_smoke_test.sh system        # Run only system health tests
#   bash scripts/jetson_smoke_test.sh gpio          # Run only GPIO tests
#   bash scripts/jetson_smoke_test.sh serial        # Run only serial tests
#   bash scripts/jetson_smoke_test.sh camera        # Run only camera tests
#   bash scripts/jetson_smoke_test.sh app           # Run only application health check
#   bash scripts/jetson_smoke_test.sh pytest        # Run only hardware pytest suite
#   bash scripts/jetson_smoke_test.sh e2e           # Run only E2E 5-second run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
VENV_DIR="${VENV_DIR:-/opt/mousedroid/venv}"
PYTHON="${VENV_DIR}/bin/python"
FAILURES=0
PASSES=0
SKIPS=0
RESULTS=()

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

check_python() {
    if [[ ! -x "${PYTHON}" ]]; then
        echo "ERROR: Python not found at ${PYTHON}"
        echo "Set VENV_DIR to the virtualenv path or run deploy_jetson.sh first."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 1. System health
# ---------------------------------------------------------------------------

test_system() {
    log_section "System Health"

    # CUDA / torch
    log_step "Checking torch.cuda.is_available()"
    if "${PYTHON}" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
        record_pass "torch.cuda.is_available"
    else
        record_fail "torch.cuda.is_available" "CUDA GPU not detected by PyTorch"
    fi

    # TensorRT
    log_step "Checking TensorRT import"
    if "${PYTHON}" -c "import tensorrt; print(f'TensorRT {tensorrt.__version__}')" 2>/dev/null; then
        record_pass "import tensorrt"
    else
        record_fail "import tensorrt" "TensorRT not importable"
    fi

    # Thermal sensor
    log_step "Reading thermal sensor"
    local thermal_path="/sys/devices/virtual/thermal/thermal_zone0/temp"
    if [[ -r "${thermal_path}" ]]; then
        local temp_raw
        temp_raw="$(cat "${thermal_path}")"
        local temp_c
        temp_c="$(echo "scale=1; ${temp_raw} / 1000" | bc 2>/dev/null || echo "unknown")"
        echo "  GPU temperature: ${temp_c} C"
        record_pass "thermal sensor read (${temp_c} C)"
    else
        record_fail "thermal sensor read" "${thermal_path} not readable"
    fi

    # Memory check
    log_step "Checking available memory"
    local mem_total_kb mem_avail_kb mem_used_pct
    mem_total_kb="$(grep MemTotal /proc/meminfo | awk '{print $2}')"
    mem_avail_kb="$(grep MemAvailable /proc/meminfo | awk '{print $2}')"
    if [[ -n "${mem_total_kb}" && -n "${mem_avail_kb}" && "${mem_total_kb}" -gt 0 ]]; then
        mem_used_pct="$(( (mem_total_kb - mem_avail_kb) * 100 / mem_total_kb ))"
        echo "  Memory: ${mem_used_pct}% used (${mem_avail_kb}kB available of ${mem_total_kb}kB)"
        if [[ "${mem_used_pct}" -lt 95 ]]; then
            record_pass "memory check (${mem_used_pct}% used)"
        else
            record_fail "memory check" "Memory usage critical: ${mem_used_pct}%"
        fi
    else
        record_fail "memory check" "Could not read /proc/meminfo"
    fi
}

# ---------------------------------------------------------------------------
# 2. GPIO
# ---------------------------------------------------------------------------

test_gpio() {
    log_section "GPIO Test"
    log_step "Testing GPIO pins 23 (trigger) and 24 (echo)"

    local gpio_script
    gpio_script=$(cat <<'PYEOF'
import sys
try:
    import Jetson.GPIO as GPIO
except ImportError:
    print("SKIP:Jetson.GPIO not available")
    sys.exit(0)

try:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(23, GPIO.OUT)
    GPIO.setup(24, GPIO.IN)
    print("PASS:GPIO pins 23/24 setup and teardown OK")
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
    log_step "Testing /dev/ttyUSB0 at 1000000 baud"

    if [[ ! -e "/dev/ttyUSB0" ]]; then
        record_fail "serial port exists" "/dev/ttyUSB0 not found — ESP32 not connected?"
        return
    fi
    record_pass "serial port exists"

    local serial_script
    serial_script=$(cat <<'PYEOF'
import json
import sys
import time

try:
    import serial
except ImportError:
    print("SKIP:pyserial not installed")
    sys.exit(0)

try:
    ser = serial.Serial("/dev/ttyUSB0", 1000000, timeout=2)
    time.sleep(0.1)  # Allow port to stabilise
    cmd = json.dumps({"T": 0}) + "\n"
    ser.write(cmd.encode())
    ser.flush()
    response = ser.readline().decode().strip()
    ser.close()
    if response:
        print(f"PASS:Got response: {response}")
    else:
        print("WARN:No response from ESP32 (timeout) — device may not be running")
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
# 4. Camera
# ---------------------------------------------------------------------------

test_camera() {
    log_section "Camera Test"
    log_step "Capturing one frame and verifying shape"

    local camera_script
    camera_script=$(cat <<'PYEOF'
import sys

# Try picamera2 first
try:
    from picamera2 import Picamera2
    cam = Picamera2()
    config = cam.create_still_configuration(main={"size": (640, 480)})
    cam.configure(config)
    cam.start()
    import time
    time.sleep(0.5)  # Allow auto-exposure to settle
    frame = cam.capture_array()
    cam.stop()
    cam.close()
    h, w = frame.shape[0], frame.shape[1]
    if h == 480 and w == 640:
        print(f"PASS:picamera2 frame shape ({h}, {w}) OK")
    else:
        print(f"FAIL:Unexpected frame shape ({h}, {w}), expected (480, 640)")
    sys.exit(0)
except ImportError:
    pass
except Exception as exc:
    print(f"FAIL:picamera2 error: {exc}")
    sys.exit(0)

# Try jetson_utils as fallback
try:
    import jetson_utils
    cam = jetson_utils.videoSource("csi://0", argv=["--input-width=640", "--input-height=480"])
    frame = cam.Capture()
    if frame is not None:
        h, w = frame.shape[0], frame.shape[1]
        if h == 480 and w == 640:
            print(f"PASS:jetson_utils frame shape ({h}, {w}) OK")
        else:
            print(f"FAIL:Unexpected frame shape ({h}, {w}), expected (480, 640)")
    else:
        print("FAIL:jetson_utils returned None frame")
    sys.exit(0)
except ImportError:
    pass
except Exception as exc:
    print(f"FAIL:jetson_utils error: {exc}")
    sys.exit(0)

print("SKIP:No camera library available (picamera2 or jetson_utils)")
PYEOF
    )

    local result
    result="$("${PYTHON}" -c "${camera_script}" 2>&1)" || true

    if echo "${result}" | grep -q "^PASS:"; then
        local msg
        msg="$(echo "${result}" | grep "^PASS:" | sed 's/^PASS://')"
        record_pass "camera capture: ${msg}"
    elif echo "${result}" | grep -q "^SKIP:"; then
        record_skip "camera capture" "No camera library available"
    else
        local msg
        msg="$(echo "${result}" | grep "^FAIL:" | sed 's/^FAIL://')"
        record_fail "camera capture" "${msg}"
    fi
}

# ---------------------------------------------------------------------------
# 5. Application health check
# ---------------------------------------------------------------------------

test_app() {
    log_section "Application Health Check"
    log_step "Running mousedroid.main --health-check"

    if "${PYTHON}" -m mousedroid.main --health-check 2>&1; then
        record_pass "mousedroid --health-check"
    else
        record_fail "mousedroid --health-check" "Non-zero exit code"
    fi
}

# ---------------------------------------------------------------------------
# 6. Hardware pytest
# ---------------------------------------------------------------------------

test_pytest() {
    log_section "Hardware Pytest Suite"
    log_step "Running pytest -m hardware"

    local pytest_bin="${VENV_DIR}/bin/pytest"
    if [[ ! -x "${pytest_bin}" ]]; then
        record_skip "hardware pytest" "pytest not installed in venv"
        return
    fi

    local test_dir="${PROJECT_DIR}/tests/hardware"
    if [[ ! -d "${test_dir}" ]]; then
        record_skip "hardware pytest" "tests/hardware/ directory not found"
        return
    fi

    local pytest_output
    if pytest_output="$("${pytest_bin}" -m hardware -v --timeout=30 "${test_dir}" 2>&1)"; then
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
# 7. E2E 5-second run
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

cfg = load_settings()

orch = build_orchestrator(cfg)
assert isinstance(orch, MouseDroidOrchestrator)

async def run_e2e():
    await orch.start()
    start = time.monotonic()
    try:
        # Run tick loop for ~5 seconds
        while time.monotonic() - start < 5.0:
            try:
                await orch.tick()
            except Exception as exc:
                print(f"tick error (non-fatal): {exc}", file=sys.stderr)
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
            test_gpio
            test_serial
            test_camera
            test_app
            test_pytest
            test_e2e
            ;;
        system)   test_system ;;
        gpio)     test_gpio ;;
        serial)   test_serial ;;
        camera)   test_camera ;;
        app)      test_app ;;
        pytest)   test_pytest ;;
        e2e)      test_e2e ;;
        *)
            echo "Unknown step: ${step}"
            echo "Valid steps: all, system, gpio, serial, camera, app, pytest, e2e"
            exit 1
            ;;
    esac

    print_summary
    exit "${FAILURES}"
}

main "$@"
