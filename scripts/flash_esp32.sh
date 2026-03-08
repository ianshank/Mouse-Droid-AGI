#!/bin/bash
# Flash ESP32 firmware for Wave Rover motor controller.
# Usage: sudo bash scripts/flash_esp32.sh /dev/ttyUSB0 firmware/waverover_mousedroid.bin
set -euo pipefail

PORT="${1:?Usage: flash_esp32.sh <port> <firmware.bin>}"
FIRMWARE="${2:?Usage: flash_esp32.sh <port> <firmware.bin>}"

if [ ! -f "${FIRMWARE}" ]; then
    echo "ERROR: Firmware file not found: ${FIRMWARE}"
    exit 1
fi

if [ ! -c "${PORT}" ]; then
    echo "ERROR: Serial port not found: ${PORT}"
    exit 1
fi

echo "=== Flashing ESP32 ==="
echo "Port: ${PORT}"
echo "Firmware: ${FIRMWARE}"

if ! command -v esptool.py &>/dev/null; then
    echo "Installing esptool..."
    pip install esptool
fi

esptool.py --port "${PORT}" --baud 460800 write_flash 0x0 "${FIRMWARE}"

echo "=== Flash complete ==="
