#!/bin/bash
# Re-flash the WAVE ROVER ESP32 with the STOCK Waveshare General_Driver image set.
# No firmware is committed (*.bin is gitignored). Stage the four parts (default
# ./firmware/); every part is sha256-verified against the tracked record BEFORE
# esptool.py runs. DRY RUN is the default: verify + print the argv. Nothing is
# written without --write. Afterwards send {"T":900,"main":2,"module":2} once.
#
# Usage: bash scripts/flash_esp32.sh --port /dev/ttyUSB0 [--bin-dir firmware]
#        [--record deployments/esp32-firmware.json] [--baud 115200] [--write | --dry-run]
#        --dry-run (the default) always wins over --write, in either order.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
PORT=""; BIN_DIR="${PROJECT_DIR}/firmware"
RECORD="${PROJECT_DIR}/deployments/esp32-firmware.json"; BAUD=""; WRITE=0; DRY_RUN_SEEN=0
die() { echo "ERROR: $*" >&2; exit 1; }
need_value() { [[ $# -ge 2 ]] || die "$1 needs a value"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)    need_value "$@"; PORT="$2"; shift 2 ;;
        --bin-dir) need_value "$@"; BIN_DIR="$2"; shift 2 ;;
        --record)  need_value "$@"; RECORD="$2"; shift 2 ;;
        --baud)    need_value "$@"; BAUD="$2"; shift 2 ;;
        --write)   WRITE=1; shift ;;
        --dry-run) DRY_RUN_SEEN=1; shift ;;
        --help|-h) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done
if [[ "${DRY_RUN_SEEN}" -eq 1 ]]; then WRITE=0; fi
[[ -n "${PORT}" ]] || die "--port is required (e.g. /dev/ttyUSB0)"
[[ -f "${RECORD}" ]] || die "firmware record not found: ${RECORD}"
[[ -d "${BIN_DIR}" ]] || die "firmware directory not found: ${BIN_DIR}"
PY_BIN="${MOUSEDROID_PYTHON:-}"
[[ -n "${PY_BIN}" ]] || PY_BIN="$(command -v python3 || command -v python || true)"
[[ -n "${PY_BIN}" ]] || die "python3 is required to read ${RECORD}"
sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
RECORD_BAUD="$("${PY_BIN}" -c 'import json,sys;print(int(json.load(open(sys.argv[1],encoding="utf-8"))["connect_baud"]))' "${RECORD}")" || die "could not read connect_baud"
BAUD="${BAUD:-${RECORD_BAUD}}"
PARTS_TXT="$("${PY_BIN}" - "${RECORD}" <<'PY'
import json, re, sys
parts = json.load(open(sys.argv[1], encoding="utf-8"))["parts"]
if len(parts) != 4:
    sys.exit(f"record must list exactly 4 parts, got {len(parts)}")
for p in parts:
    if not re.fullmatch(r"0x[0-9A-Fa-f]+", p["offset"]): sys.exit(f"bad offset {p['offset']!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", p["sha256"]): sys.exit(f"bad sha256 for {p['name']}")
    if "/" in p["file"] or p["file"] in ("", ".", ".."): sys.exit(f"bad file name {p['file']!r}")
    print(p["offset"], p["file"], p["sha256"], int(p["size"]))
PY
)" || die "could not read parts from ${RECORD}"
PARTS=()
while IFS= read -r line; do PARTS+=("${line}"); done <<< "${PARTS_TXT}"
[[ ${#PARTS[@]} -eq 4 ]] || die "expected 4 parts from ${RECORD}, got ${#PARTS[@]}"
ARGS=()
for line in "${PARTS[@]}"; do
    read -r offset file expected size <<< "${line}"
    path="${BIN_DIR}/${file}"
    if [ ! -f "${path}" ]; then die "part missing: ${path}"; fi
    actual_size="$(wc -c < "${path}" | tr -d '[:space:]')"
    [[ "${actual_size}" == "${size}" ]] || die "size mismatch for ${file}: ${actual_size} != ${size}"
    actual="$(sha256_of "${path}")"
    [[ "${actual}" == "${expected}" ]] || die "sha256 mismatch for ${file}: ${actual} != ${expected} — refusing to flash"
    echo "verified ${file} @ ${offset} sha256=${expected}"
    ARGS+=("${offset}" "${path}")
done
# Factory flags: the esptool block of the record documents them and a unit test pins the two.
COMMON=(esptool.py --chip esp32 --port "${PORT}" --baud "${BAUD}" --before default_reset --after no_reset)
WRITE_CMD=("${COMMON[@]}" write_flash --flash_mode keep --flash_freq keep --flash_size keep --compress "${ARGS[@]}")
VERIFY_CMD=("${COMMON[@]}" verify_flash "${ARGS[@]}")
echo "esptool argv (write):  ${WRITE_CMD[*]}"
echo "esptool argv (verify): ${VERIFY_CMD[*]}"
if [[ "${WRITE}" -ne 1 ]]; then echo "DRY RUN — 4/4 parts verified; nothing flashed. Re-run with --write."; exit 0; fi
if [ ! -c "${PORT}" ]; then die "serial port not found: ${PORT}"; fi
command -v esptool.py >/dev/null 2>&1 || die "esptool.py not on PATH — run: pip install esptool (never done automatically)"
"${WRITE_CMD[@]}"
"${VERIFY_CMD[@]}"
echo "=== Flash complete. Power-cycle the chassis, then send the post_flash_config frame from deployments/esp32-firmware.json once ==="
