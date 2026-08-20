---
name: jetson-hardware-debug
description: "Debug and verify MouseDroid Jetson Orin Nano hardware deployment. Use for: sensor verification (camera, GPIO, LiDAR, speaker, microphone), Jetson SSH connection, config sync and Docker deployment, smoke testing, and troubleshooting hardware failures. WHEN: setup Jetson, deploy to Jetson, verify sensors, debug hardware on Jetson, Jetson sensor failing, GPIO not working, camera not initializing, LiDAR not detected, speaker not playing, fix Jetson hardware issue, Jetson pinmux, Jetson config overlay, run smoke test."
argument-hint: "Specify task: connect, deploy, verify-sensors, smoke-test, or debug-<sensor>"
---

# Jetson Hardware Debug & Deployment

Debug and verify MouseDroid hardware on NVIDIA Jetson Orin Nano. Covers SSH connection, config overlay sync, Docker deployment, sensor verification, smoke testing, and common failure recovery.

> **Environment variables:** Commands below use `${JETSON_USB_IP}` (USB-C)
> and `${JETSON_ETH_IP}` (Ethernet). Set these from your network, or see
> `docs/runbooks/claude-code-on-jetson.md` for the real values.

## Connection & Access

### SSH to Jetson

**Over USB-C (fastest, <1ms latency):**
```bash
ssh -i ~/.ssh/id_ed25519 ian@
```

**Over Ethernet:**
```bash
ssh -i ~/.ssh/id_ed25519 ian@
ssh -i ~/.ssh/id_ed25519 ian@mousedroid.local
```

**Hostname:** `mousedroid` (Orin Nano Super, p3767-0005-super)

### Verify SSH Access & Basic System Health

```bash
ssh ian@ "uname -a && nvidia-smi && docker ps"
```

Expected: Jetson Linux kernel, NVIDIA GPU visible, `mousedroid` container running.

---

## Pre-Deployment Checklist

Before deploying code or config changes to Jetson:

1. **Config is synced in repo** — all YAML overlays committed
2. **No CRLF line endings** — after any file transfer from Windows, run:
   ```bash
   ssh ian@ "find /opt/mousedroid -name '*.py' -o -name '*.yaml' | xargs sed -i 's/\r$//'"
   ```
3. **Local changes on Jetson are committed or backed up** — repo on Jetson may have uncommitted edits
4. **Docker image is built** — `docker-compose build` on Jetson or rebuild locally and push

---

## Config Overlay Sync & Docker Deployment

### Scenario: Deploy new config to running Jetson

**On your local machine:**

1. Commit config changes to repo:
   ```bash
   git add config/jetson_production.yaml
   git commit -m "Update Jetson production config"
   git push
   ```

2. SSH to Jetson and pull latest:
   ```bash
   ssh ian@ "cd /opt/mousedroid && git pull"
   ```

3. Sync config overlay to `/etc/mousedroid/`:
   ```bash
   ssh ian@ "sudo cp /opt/mousedroid/config/jetson_production.yaml /etc/mousedroid/jetson_production.yaml"
   ```

4. Restart the `mousedroid` container:
   ```bash
   ssh ian@ "cd /opt/mousedroid && docker compose -f docker-compose.jetson.yml restart mousedroid"
   ```

5. Verify container is healthy:
   ```bash
   ssh ian@ "docker logs mousedroid | tail -20"
   ```

### Scenario: Full Jetson Docker Build & Deploy

If code changes require a Docker rebuild:

1. On Jetson, rebuild the image:
   ```bash
   ssh ian@ "cd /opt/mousedroid && docker compose -f docker-compose.jetson.yml build --no-cache mousedroid"
   ```

2. Restart:
   ```bash
   ssh ian@ "docker compose -f docker-compose.jetson.yml restart mousedroid"
   ```

3. Check logs for build errors:
   ```bash
   ssh ian@ "docker logs mousedroid | head -50"
   ```

---

## Sensor Verification

### Verify All Sensors at Once

**On Jetson** (inside or outside container):

```bash
# Inside Docker container:
docker exec -w /opt/mousedroid mousedroid python3 scripts/verify_sensors.py --sensor all

# Or outside Docker (if venv is available):
python3 scripts/verify_sensors.py --sensor all --config config/jetson_production.yaml
```

Exit code is non-zero if any sensor fails.

### Verify Specific Sensor

**Camera:**
```bash
docker exec -w /opt/mousedroid mousedroid python3 scripts/verify_sensors.py --sensor camera
```

**GPIO / Ultrasonic Distance:**
```bash
docker exec -w /opt/mousedroid mousedroid python3 scripts/verify_sensors.py --sensor gpio
```

**LiDAR (LD19):**
```bash
docker exec -w /opt/mousedroid mousedroid python3 scripts/verify_sensors.py --sensor lidar
```

**Microphone:**
```bash
docker exec -w /opt/mousedroid mousedroid python3 scripts/verify_sensors.py --sensor microphone
```

**Speaker:**
```bash
docker exec -w /opt/mousedroid mousedroid python3 scripts/verify_sensors.py --sensor speaker
```

**JSON output** (for parsing):
```bash
docker exec -w /opt/mousedroid mousedroid python3 scripts/verify_sensors.py --json
```

---

## Smoke Testing

Smoke tests validate system, hardware, and application health in stages.

### Run Full Smoke Suite

**On Jetson:**
```bash
bash scripts/jetson_smoke_test.sh
```

This runs all stages: system, GPIO, serial, camera, LiDAR, speaker, app health, pytest, E2E.

### Run Specific Smoke Stage

```bash
bash scripts/jetson_smoke_test.sh system        # System health (CPU, disk, memory)
bash scripts/jetson_smoke_test.sh gpio          # GPIO / ultrasonic
bash scripts/jetson_smoke_test.sh serial        # ESP32 serial communication
bash scripts/jetson_smoke_test.sh camera        # Camera initialization
bash scripts/jetson_smoke_test.sh lidar         # LiDAR detection
bash scripts/jetson_smoke_test.sh speaker       # Speaker playback
bash scripts/jetson_smoke_test.sh app           # Application health endpoint
bash scripts/jetson_smoke_test.sh pytest        # Hardware pytest suite
bash scripts/jetson_smoke_test.sh e2e           # 5-second E2E runtime
```

**Smoke test summary:** Outputs `SUMMARY.md` in the current directory with results.

---

## Common Hardware Failures & Recovery

### GPIO / Ultrasonic Not Working

**Symptom:** GPIO test fails, ultrasonic returns 999.0m (degraded).

**Cause:** Pinmux not set after reboot. HC-SR04 trigger pin (BCM 23) defaults to input.

**Fix:** Run pinmux correction on Jetson:
```bash
ssh ian@ "sudo busybox devmem 0x243D020 w 0x5"
```

**Verify:**
```bash
docker exec -w /opt/mousedroid mousedroid python3 scripts/verify_sensors.py --sensor gpio
```

### Camera Not Initializing

**Symptom:** Camera test fails, verify output shows `FAIL`.

**Cause:** Ribbon camera not seated, libargus socket missing, or CSI/I2C permission denied.

**Fixes:**

1. Check libargus is running:
   ```bash
   ssh ian@ "ps aux | grep argus"
   ```

2. Verify camera is visible to Jetson:
   ```bash
   ssh ian@ "ls -la /dev/video*"
   ```

3. Check container has privileged access in `docker-compose.jetson.yml`:
   ```yaml
   privileged: true
   ```

4. Reseat the ribbon cable and restart the container:
   ```bash
   ssh ian@ "docker restart mousedroid"
   ```

### LiDAR (LD19) Not Detected

**Symptom:** LiDAR test fails, serial port not found.

**Cause:** USB-to-serial adapter not detected, wrong baud rate, or missing udev permissions.

**Fixes:**

1. Verify USB device is present:
   ```bash
   ssh ian@ "lsusb | grep -i 'qinheng\|ch340'"
   ```

2. Check device node exists:
   ```bash
   ssh ian@ "ls -la /dev/ttyUSB*"
   ```

3. Verify container has `/dev/ttyUSB0` mounted in `docker-compose.jetson.yml`:
   ```yaml
   devices:
     - /dev/ttyUSB0:/dev/ttyUSB0
   ```

4. Check baud rate in `jetson_production.yaml` matches LiDAR setting (typically 230400).

5. Restart container:
   ```bash
   ssh ian@ "docker restart mousedroid"
   ```

### Speaker Not Playing Audio

**Symptom:** Speaker test fails, no sound output.

**Cause:** PyAudio import failure, USB speaker not detected, or volume muted.

**Fixes:**

1. Check USB audio device is detected:
   ```bash
   ssh ian@ "arecord -l && aplay -l"
   ```

2. Verify PyAudio is installed in container:
   ```bash
   docker exec mousedroid python3 -c "import pyaudio; print(pyaudio.__version__)"
   ```

3. Check speaker is not muted:
   ```bash
   ssh ian@ "amixer sset Master unmute && amixer sset Master 80%"
   ```

4. Verify container has audio device mounted in `docker-compose.jetson.yml`:
   ```yaml
   devices:
     - /dev/snd:/dev/snd
   ```

### ESP32 Connection Failure

**Symptom:** Serial test fails, ESP32 not responding. System continues in degraded mode.

**Cause:** USB cable loose, wrong COM port, or baud rate mismatch.

**Fixes:**

1. Check USB-to-serial device:
   ```bash
   ssh ian@ "lsusb | grep CH340"
   ```

2. Verify correct `/dev/ttyUSB*` in config:
   ```yaml
   esp32:
     serial_port: /dev/ttyUSB0
     baud_rate: 115200
   ```

3. Reset ESP32:
   ```bash
   ssh ian@ "docker exec mousedroid python3 -c 'import serial; s=serial.Serial(\"/dev/ttyUSB0\", 115200); s.reset_input_buffer()'"
   ```

---

## Hardware Configuration Reference

### Pin Mappings

| Component | Pin Type | Value | Notes |
|-----------|----------|-------|-------|
| Ultrasonic Trigger | BCM (GPIO) | 23 | Output, requires pinmux fix after reboot |
| Ultrasonic Echo | BCM (GPIO) | 24 | Input |
| LiDAR | USB/Serial | `/dev/ttyUSB0` | Baud 230400 |
| Camera | CSI | Ribbon cable | Requires libargus daemon |
| Speaker | USB Audio | `/dev/snd` | PyAudio + USB speaker |
| Microphone | USB Audio | `/dev/snd` | PyAudio + USB mic |
| ESP32 | USB/Serial | `/dev/ttyUSB1` (or `/dev/ttyUSB0` if only one) | Baud 115200 |

### Critical Jetson Settings

| Setting | Value | Reason |
|---------|-------|--------|
| `privileged: true` (Docker) | Required | GPIO, I2C, libargus access |
| `/proc/device-tree` mount | Required | Jetson.GPIO 2.1.12 |
| Config overlay path | `/etc/mousedroid/jetson_production.yaml` | Read-only bind-mount |
| RSSM weights location | `/opt/mousedroid/weights/` | 17 HuggingFace .pt files |
| Phi-3 model location | `/opt/mousedroid/models/Phi-3-mini-4k-instruct-q4.gguf` | Runtime inference |

---

## Logs & Debugging

### Tail Container Logs

```bash
ssh ian@ "docker logs -f mousedroid"
```

### View Structured Logs (JSON)

```bash
ssh ian@ "docker logs mousedroid 2>&1 | grep 'event='"
```

### Run Pytest Hardware Suite Inside Docker

```bash
docker exec -w /opt/mousedroid mousedroid python3 -m pytest -m hardware tests/hardware/ -v
```

### Check Available Test Markers

```bash
docker exec mousedroid python3 -m pytest --markers
```

Expected markers: `slow`, `hardware`, `smoke`, `arm`, `integration`.

---

## Next Steps

- If smoke test reports voice failures, refer to `NEXT_STEPS.md` § Voice Engine Quality
- If GPIO continues to fail, check systemd service post-reboot automation in `mousedroid-docker.service`
- For bootstrap issues, see `scripts/jetson_bootstrap.sh`
- For E2E debugging, enable `debug: true` in `jetson_production.yaml` for verbose logs
