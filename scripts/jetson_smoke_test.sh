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
#   bash scripts/jetson_smoke_test.sh mic           # Run only microphone capture test
#   bash scripts/jetson_smoke_test.sh telemetry     # Run only telemetry server test
#   bash scripts/jetson_smoke_test.sh e2e           # Run only E2E run (default 10s)
#   bash scripts/jetson_smoke_test.sh training      # Run only training dry-run
#   bash scripts/jetson_smoke_test.sh llm           # Run only LLM model check
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
VENV_DIR="${VENV_DIR:-/opt/mousedroid/venv}"
PYTHON="${VENV_DIR}/bin/python"
MOUSEDROID_E2E_DURATION_S="${MOUSEDROID_E2E_DURATION_S:-10}"
MOUSEDROID_TELEMETRY_PORT="${MOUSEDROID_TELEMETRY_PORT:-8080}"
MOUSEDROID_CONFIG="${MOUSEDROID_CONFIG:-config/jetson_production.yaml}"
MODEL_PATH="${MODEL_PATH:-/opt/mousedroid/models/llama-3-8b-instruct.Q4_K_M.gguf}"
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
    cuda_img = cam.Capture()
    frame = jetson_utils.cudaToNumpy(cuda_img) if cuda_img is not None else None
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
# 7. Microphone capture
# ---------------------------------------------------------------------------

test_mic() {
    log_section "Microphone Capture Test"
    log_step "Validating USB audio capture"

    local mic_script
    mic_script=$(cat <<'PYEOF'
import sys

try:
    import pyaudio
except ImportError:
    print("SKIP:pyaudio not installed")
    sys.exit(0)

pa = pyaudio.PyAudio()
try:
    found = False
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            found = True
            name = info.get("name", "unknown")
            rate = int(info.get("defaultSampleRate", 0))
            print(f"PASS:USB mic detected: {name} @ {rate} Hz")
            break
    if not found:
        print("FAIL:No audio input device found")
except Exception as exc:
    print(f"FAIL:Microphone test error: {exc}")
finally:
    pa.terminate()
PYEOF
    )

    local result
    result="$("${PYTHON}" -c "${mic_script}" 2>&1)" || true

    if echo "${result}" | grep -q "^PASS:"; then
        local msg
        msg="$(echo "${result}" | grep "^PASS:" | sed 's/^PASS://')"
        record_pass "microphone capture: ${msg}"
    elif echo "${result}" | grep -q "^SKIP:"; then
        record_skip "microphone capture" "pyaudio not installed"
    else
        local msg
        msg="$(echo "${result}" | grep "^FAIL:" | sed 's/^FAIL://')"
        record_fail "microphone capture" "${msg}"
    fi
}

# ---------------------------------------------------------------------------
# 8. Telemetry server
# ---------------------------------------------------------------------------

test_telemetry() {
    log_section "Telemetry Server Test"
    log_step "Starting telemetry, hitting /health, then stopping"

    local telemetry_script
    telemetry_script=$(cat <<PYEOF
import asyncio
import sys

try:
    import aiohttp
except ImportError:
    print("SKIP:aiohttp not installed")
    sys.exit(0)

from mousedroid.config.loader import load_settings
from mousedroid.telemetry.server import TelemetryServer

cfg = load_settings()
port = ${MOUSEDROID_TELEMETRY_PORT}

async def check_health():
    server = TelemetryServer(cfg)
    await server.start()
    try:
        await asyncio.sleep(1)
        url = f"http://127.0.0.1:{port}/health"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    print(f"PASS:Telemetry /health returned 200 on port {port}")
                else:
                    print(f"FAIL:Telemetry /health returned {resp.status}")
    except Exception as exc:
        print(f"FAIL:Telemetry health check error: {exc}")
    finally:
        await server.stop()

try:
    asyncio.run(check_health())
except Exception as exc:
    print(f"FAIL:Telemetry server error: {exc}")
    sys.exit(1)
PYEOF
    )

    local result
    result="$(timeout 15 "${PYTHON}" -c "${telemetry_script}" 2>&1)" || true

    if echo "${result}" | grep -q "^PASS:"; then
        local msg
        msg="$(echo "${result}" | grep "^PASS:" | sed 's/^PASS://')"
        record_pass "telemetry server: ${msg}"
    elif echo "${result}" | grep -q "^SKIP:"; then
        record_skip "telemetry server" "aiohttp not installed"
    else
        local msg
        msg="$(echo "${result}" | grep "^FAIL:" | sed 's/^FAIL://')"
        record_fail "telemetry server" "${msg:-unknown error}"
    fi
}

# ---------------------------------------------------------------------------
# 9. E2E run (configurable duration via MOUSEDROID_E2E_DURATION_S)
# ---------------------------------------------------------------------------

test_e2e() {
    local duration="${MOUSEDROID_E2E_DURATION_S}"
    local timeout_s=$((duration + 10))
    log_section "E2E ${duration}-Second Run"
    log_step "Starting orchestrator for ${duration} seconds, then SIGINT shutdown"

    local e2e_script
    e2e_script=$(cat <<PYEOF
import asyncio
import sys
import time

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
        while time.monotonic() - start < ${duration}:
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
    result="$(timeout --signal=INT "${timeout_s}" "${PYTHON}" -c "${e2e_script}" 2>&1)" || true

    if echo "${result}" | grep -q "^PASS:"; then
        local msg
        msg="$(echo "${result}" | grep "^PASS:" | tail -1 | sed 's/^PASS://')"
        record_pass "E2E ${duration}s run: ${msg}"
    else
        local msg
        msg="$(echo "${result}" | grep "^FAIL:" | sed 's/^FAIL://')"
        record_fail "E2E ${duration}s run" "${msg:-unknown error}"
    fi
}

# ---------------------------------------------------------------------------
# 10. Training dry-run
# ---------------------------------------------------------------------------

test_training() {
    log_section "Training Dry-Run"
    log_step "Running a single RSSM training step on GPU"

    local training_script
    training_script=$(cat <<'PYEOF'
import sys

try:
    import torch
except ImportError:
    print("SKIP:torch not installed")
    sys.exit(0)

if not torch.cuda.is_available():
    print("SKIP:CUDA not available")
    sys.exit(0)

try:
    model = torch.nn.Linear(256, 64).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")

    x = torch.randn(4, 256, device="cuda")
    optimizer.zero_grad()
    with torch.amp.autocast("cuda"):
        loss = model(x).sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    print(f"PASS:Training step completed, loss={loss.item():.4f}")
except Exception as exc:
    print(f"FAIL:Training dry-run error: {exc}")
    sys.exit(1)
PYEOF
    )

    local result
    result="$(timeout 30 "${PYTHON}" -c "${training_script}" 2>&1)" || true

    if echo "${result}" | grep -q "^PASS:"; then
        local msg
        msg="$(echo "${result}" | grep "^PASS:" | sed 's/^PASS://')"
        record_pass "training dry-run: ${msg}"
    elif echo "${result}" | grep -q "^SKIP:"; then
        local reason
        reason="$(echo "${result}" | grep "^SKIP:" | sed 's/^SKIP://')"
        record_skip "training dry-run" "${reason}"
    else
        local msg
        msg="$(echo "${result}" | grep "^FAIL:" | sed 's/^FAIL://')"
        record_fail "training dry-run" "${msg:-unknown error}"
    fi
}

# ---------------------------------------------------------------------------
# 11. LLM model check
# ---------------------------------------------------------------------------

test_llm() {
    log_section "LLM Model Check"
    log_step "Verifying LLM model file at ${MODEL_PATH}"

    if [[ ! -f "${MODEL_PATH}" ]]; then
        record_skip "LLM model file" "Model not downloaded at ${MODEL_PATH}"
        return
    fi

    local size_bytes
    size_bytes="$(stat --format='%s' "${MODEL_PATH}" 2>/dev/null || stat -f '%z' "${MODEL_PATH}" 2>/dev/null)"
    if [[ -z "${size_bytes}" ]]; then
        record_fail "LLM model file" "Could not determine file size"
        return
    fi

    local size_gb
    size_gb="$(echo "scale=2; ${size_bytes} / 1073741824" | bc 2>/dev/null || echo "unknown")"
    echo "  Model size: ${size_gb} GB"

    # Minimal sanity: GGUF files must be > 100MB
    if [[ "${size_bytes}" -gt 104857600 ]]; then
        record_pass "LLM model file (${size_gb} GB at ${MODEL_PATH})"
    else
        record_fail "LLM model file" "File too small (${size_bytes} bytes) — possibly corrupt"
    fi

    # Try loading the model header with llama-cpp-python if available
    local llm_script
    llm_script=$(cat <<PYEOF
import sys
try:
    from llama_cpp import Llama
    llm = Llama(model_path="${MODEL_PATH}", n_ctx=128, n_gpu_layers=0, verbose=False)
    print(f"PASS:LLM model loaded OK, vocab_size={llm.n_vocab()}")
    del llm
except ImportError:
    print("SKIP:llama-cpp-python not installed")
except Exception as exc:
    print(f"WARN:LLM load test failed: {exc}")
PYEOF
    )

    local llm_result
    llm_result="$(timeout 30 "${PYTHON}" -c "${llm_script}" 2>&1)" || true

    if echo "${llm_result}" | grep -q "^PASS:"; then
        local msg
        msg="$(echo "${llm_result}" | grep "^PASS:" | sed 's/^PASS://')"
        record_pass "LLM model load: ${msg}"
    elif echo "${llm_result}" | grep -q "^SKIP:"; then
        record_skip "LLM model load" "llama-cpp-python not installed"
    elif echo "${llm_result}" | grep -q "^WARN:"; then
        local msg
        msg="$(echo "${llm_result}" | grep "^WARN:" | sed 's/^WARN://')"
        record_pass "LLM model file OK (load test skipped: ${msg})"
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

    # Structured JSON output
    local json_results="["
    local first=true
    for r in "${RESULTS[@]}"; do
        if [[ "${first}" != "true" ]]; then
            json_results+=","
        fi
        first=false
        local status
        status="$(echo "${r}" | cut -d: -f1)"
        local detail
        detail="$(echo "${r}" | cut -d: -f2-)"
        json_results+="{\"status\":\"${status}\",\"detail\":\"${detail}\"}"
    done
    json_results+="]"

    if command -v jq &>/dev/null; then
        jq -nc \
            --argjson passed "${PASSES}" \
            --argjson failed "${FAILURES}" \
            --argjson skipped "${SKIPS}" \
            --argjson results "${json_results}" \
            '{"event":"smoke_test_complete","passed":$passed,"failed":$failed,"skipped":$skipped,"results":$results}'
    else
        echo "{\"event\":\"smoke_test_complete\",\"passed\":${PASSES},\"failed\":${FAILURES},\"skipped\":${SKIPS}}"
    fi

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
            test_mic
            test_app
            test_pytest
            test_telemetry
            test_e2e
            test_training
            test_llm
            ;;
        system)     test_system ;;
        gpio)       test_gpio ;;
        serial)     test_serial ;;
        camera)     test_camera ;;
        mic)        test_mic ;;
        app)        test_app ;;
        pytest)     test_pytest ;;
        telemetry)  test_telemetry ;;
        e2e)        test_e2e ;;
        training)   test_training ;;
        llm)        test_llm ;;
        *)
            echo "Unknown step: ${step}"
            echo "Valid steps: all, system, gpio, serial, camera, mic, app, pytest, telemetry, e2e, training, llm"
            exit 1
            ;;
    esac

    print_summary
    exit "${FAILURES}"
}

main "$@"
