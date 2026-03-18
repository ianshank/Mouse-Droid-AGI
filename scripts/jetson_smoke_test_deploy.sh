#!/bin/bash
set -euo pipefail

CONTAINER_NAME="${MOUSEDROID_CONTAINER:-mousedroid}"
CHECK_MIC="${MOUSEDROID_CHECK_MIC:-true}"
CHECK_ULTRASONIC="${MOUSEDROID_CHECK_ULTRASONIC:-true}"
LOG_LINES="${MOUSEDROID_LOG_LINES:-200}"

run_in_container() {
    sudo docker exec "${CONTAINER_NAME}" "$@"
}

echo '=== CONTAINER STATUS ==='
sudo docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'

if ! sudo docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "ERROR: container '${CONTAINER_NAME}' is not running"
    exit 1
fi

echo '=== IMPORT TEST ==='
run_in_container python3 -c 'import mousedroid; print("mousedroid OK")'

echo '=== TORCH/CUDA TEST ==='
run_in_container python3 -c '
import torch
v = torch.__version__
cuda = torch.cuda.is_available()
gpu = torch.cuda.get_device_name(0) if cuda else None
print(f"torch={v}, cuda={cuda}, gpu={gpu}")
'

echo '=== WEIGHTS + LLM TEST ==='
run_in_container python3 /opt/mousedroid/scripts/_test_weights_llm.py

if [[ "${CHECK_ULTRASONIC}" == "true" ]]; then
    echo '=== ULTRASONIC TEST ==='
    sudo docker exec -i "${CONTAINER_NAME}" python3 - <<'PY'
import asyncio
from pathlib import Path

from mousedroid.config.loader import load_settings
from mousedroid.hardware.sensors.ultrasonic import HcSr04

cfg = load_settings(Path("/etc/mousedroid/jetson_production.yaml"))
if cfg.ultrasonic is None:
    raise SystemExit("ultrasonic config missing")

sensor = HcSr04(cfg.ultrasonic)
distance = asyncio.run(sensor.read_distance_m())
print(f"ULTRASONIC_TEST_PASSED distance_m={distance:.3f}")
PY
fi

if [[ "${CHECK_MIC}" == "true" ]]; then
    echo '=== MICROPHONE TEST ==='
    run_in_container python3 /opt/mousedroid/scripts/_test_mic.py
fi

echo '=== LOG HEALTH ==='
LOG_OUTPUT="$(sudo docker logs --tail "${LOG_LINES}" "${CONTAINER_NAME}" 2>&1 || true)"
printf '%s
' "${LOG_OUTPUT}" | tail -20

if printf '%s
' "${LOG_OUTPUT}" | grep -Eq 'audio_capture_failed|microphone_start_failed|Traceback'; then
    echo 'ERROR: unhealthy log patterns detected in recent container logs'
    exit 1
fi

if [[ "${CHECK_MIC}" == "true" ]] && ! printf '%s
' "${LOG_OUTPUT}" | grep -q 'usb_microphone_started'; then
    echo 'ERROR: microphone test passed but usb_microphone_started was not observed in recent logs'
    exit 1
fi

echo '=== ALL SMOKE TESTS PASSED ==='
