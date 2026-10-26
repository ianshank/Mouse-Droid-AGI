# Deployment Guide

## Path

1. **Preflight** — validate hardware before starting: `bash scripts/preflight_check.sh`
   (used as the systemd `ExecStartPre`), or `python -m mousedroid.cli.preflight --json` for a machine-parseable
   report.
2. **Flash ESP32** — Flashing is not required: the board runs stock firmware. Repair-only: stage the
   four factory parts under `firmware/` (gitignored) with exactly the file names in
   `deployments/esp32-firmware.json` (`General_Driver.ino.bootloader.bin`,
   `General_Driver.ino.partitions.bin`, `boot_app0.bin`, `General_Driver.ino.bin`); the record pins
   each part by size and sha256 and is authoritative. The parts come from the Waveshare WAVE ROVER
   factory package (`flash_download_tool_3.9.5/bin/`). Run
   `bash scripts/flash_esp32.sh --port /dev/ttyUSB0` (a dry run: it verifies all four parts and writes
   nothing), then add `--write`. Writing needs access to the serial device: use `sudo` or an account in
   the `dialout` group.
3. **Deploy** — native `sudo bash scripts/deploy_jetson.sh`, or Docker `sudo bash scripts/docker_deploy.sh`.
4. **Service** — install `scripts/mousedroid.service` (native, `Type=notify`, `WatchdogSec=30`) or
   `scripts/mousedroid-docker.service`, then `systemctl enable --now mousedroid`.

## NVMe SSD (recommended)

The Orin Nano has 8 GB of shared RAM; put Docker data and a 16 GB swapfile on the 500 GB NVMe so
memory-intensive builds (e.g. `llama-cpp-python` CUDA compilation) don't thrash. See
[architecture/ADR-l4t-container.md](architecture/ADR-l4t-container.md) for the container architecture.

## Probe-first bring-up

A dead ESP32 with `esp32.enabled=true` makes `orchestrator.start()` retry-then-raise, which crash-loops the
container. **Probe the ESP32 first**; if it does not respond, set `MOUSEDROID_ESP32__ENABLED=false` to fall
back to the mock driver (the resilience wrapper stays in place). Full operator runbooks:

- [runbooks/jetson-full-bringup.md](runbooks/jetson-full-bringup.md) — full rover bring-up.
- [runbooks/jetson-rover-smoke.md](runbooks/jetson-rover-smoke.md) — USB-C rover smoke test.
- [runbooks/jetson-claude-pilot-deploy.md](runbooks/jetson-claude-pilot-deploy.md) — cloud/local LLM deploy.
