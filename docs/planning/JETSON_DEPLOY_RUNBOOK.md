# Jetson Orin Nano Deployment Runbook

> **Branch:** `feat/test-validation-and-jetson-deploy-prep`
> **Target:** Jetson Orin Nano 8GB (JetPack 6.x / L4T R36.4)
> **Status:** Pre-deploy (hardware not yet attached)
> **Prerequisite:** All software gates passed on Windows dev host

---

## Pre-Connect Checklist (do before plugging in the Jetson)

- [ ] Verify SSH key exists: `~/.ssh/id_ed25519` (or generate with `ssh-keygen`)
- [ ] Confirm network: Jetson connects via Ethernet or USB-C networking
- [ ] Confirm the Windows host has `ssh` client available
- [ ] Review `NEXT_STEPS.md` items #1 (API key rotation) and #2 (ESP32 status)
- [ ] Have the new `ANTHROPIC_API_KEY` ready for rotation (P0 security)

---

## Phase 1: Physical Connection & Discovery

### Step 1 — Connect hardware

1. Physically attach the Jetson Orin Nano to power supply
2. Connect Ethernet cable (or USB-C for serial console + networking)
3. Wait 30–60 seconds for boot

### Step 2 — Discover the Jetson

```bash
# From the Windows host (or use nmap)
ping mousedroid.local
# Or use the discovery script if available:
bash scripts/jetson_discover.sh
```

### Step 3 — SSH in

```bash
ssh ian@<jetson-ip>
# Verify the system:
cat /etc/nv_tegra_release   # Should show L4T R36.4
nvidia-smi                   # Should show Orin Nano iGPU
docker --version             # Should be available
```

---

## Phase 2: Source Code Sync

### Step 4 — Fix ownership drift

The bind-mount at `/opt/mousedroid` may have root-owned files from container
writes. Fix before any git operations:

```bash
sudo chown -R ian:ian /opt/mousedroid
cd /opt/mousedroid
git status  # Should now work without permission errors
```

### Step 5 — Sync to trunk

```bash
cd /opt/mousedroid
git fetch origin
git checkout feat/test-validation-and-jetson-deploy-prep
git pull origin feat/test-validation-and-jetson-deploy-prep
```

### Step 6 — Verify source matches

```bash
git log -n 1 --oneline
# Should match the commit from the Windows dev host
python3 -c "from mousedroid.config.schema import Settings; print(Settings().model_fields.keys())"
```

---

## Phase 3: Environment Configuration

### Step 7 — Install/update docker.env

```bash
# Check if the env file exists:
ls -la /etc/mousedroid/docker.env

# If missing, create from template:
sudo mkdir -p /etc/mousedroid
sudo cp config/docker.env.example /etc/mousedroid/docker.env
sudo chown ian:ian /etc/mousedroid/docker.env

# Edit with the correct overrides:
nano /etc/mousedroid/docker.env
```

**Required overrides in `docker.env`:**
```env
MOUSEDROID_LLM__ENABLED=true
MOUSEDROID_LLM__N_GPU_LAYERS=0          # CPU fallback (iGPU shared with world model)
MOUSEDROID_ESP32__ENABLED=false          # ESP32 is dead (F-008 blocker)
# MOUSEDROID_ANTHROPIC_API_KEY=<NEW_KEY>  # Set after rotation (Step 12)
```

### Step 8 — Run host bootstrap

```bash
cd /opt/mousedroid
bash scripts/host_bootstrap.sh
# This installs systemd units and configures the host environment
# It has dry-run/backup/rollback safety
```

---

## Phase 4: Docker Build & Start

### Step 9 — Build the container image

```bash
cd /opt/mousedroid
docker compose -f docker-compose.jetson.yml build
# This builds from Dockerfile.jetson using dustynv/l4t-pytorch:r36.4.0
# Expected time: 10-20 minutes on first build
```

### Step 10 — Start the container

```bash
docker compose -f docker-compose.jetson.yml up -d
# Verify it's running:
docker ps
docker logs mousedroid --tail 50
# Check health:
docker exec mousedroid python3 -m mousedroid.main --health-check
```

---

## Phase 5: Cold Hardware Validation

### Step 11 — USB-C endpoint discovery

```bash
# Check what devices are connected:
python3 scripts/check_usbc_devices.py
# Expected: LiDAR, camera (CSI), possibly ESP32
```

### Step 12 — Individual sensor probes

Run these OUTSIDE the container (exclusive device access):

```bash
# Stop container for cold probes:
docker compose -f docker-compose.jetson.yml stop

# LiDAR probe:
python3 scripts/verify_sensors.py --sensor lidar

# Camera probe (CSI):
python3 scripts/verify_sensors.py --sensor camera

# ESP32 probe (expected: FAIL — known dead):
python3 scripts/verify_sensors.py --sensor esp32
# If this fails, confirm ESP32__ENABLED=false in docker.env

# GPIO probe:
python3 scripts/verify_sensors.py --sensor gpio

# OLED face display:
python3 scripts/verify_sensors.py --sensor display

# ALWAYS restart the container after cold probes:
docker compose -f docker-compose.jetson.yml start
```

---

## Phase 6: Warm Validation (Container Running)

### Step 13 — Full on-device validation

```bash
# This is the comprehensive 3-phase validation script:
bash scripts/jetson_full_validation.sh
# Phase 1: Static CI (lint, typecheck, tests inside container)
# Phase 2: Cold hardware probes (container stopped temporarily)
# Phase 3: Warm server validation (API endpoints, WebSocket, metrics)
```

### Step 14 — Hardware test suite

```bash
# Run the hardware-specific pytest suite:
cd /opt/mousedroid
pytest tests/hardware/ -v --tb=short
# Expected: ~25 test files, most should PASS
# ESP32 tests will skip/fail (known)
# Camera tests depend on CSI cable connection
```

---

## Phase 7: Security & Monitoring

### Step 15 — Rotate ANTHROPIC_API_KEY (P0)

```bash
# 1. Generate new key at https://console.anthropic.com/
# 2. Update on the Jetson:
sudo nano /etc/mousedroid/docker.env
# Set: MOUSEDROID_ANTHROPIC_API_KEY=<new_key>

# 3. Restart container:
sudo systemctl restart mousedroid-docker

# 4. Verify cloud tier:
docker exec mousedroid python3 tools/llm_latency_probe.py --iterations 3

# 5. Revoke the old key in the Anthropic console
# 6. Update GitHub Actions secrets if applicable
```

### Step 16 — Wire monitoring stack

```bash
# Import Grafana dashboard:
# Upload docs/grafana_dashboard.json to Grafana instance

# Load Prometheus alerts:
# Copy config/prometheus/alerts.yml to Prometheus config dir
# Verify: promtool check rules config/prometheus/alerts.yml

# Start monitoring stack:
docker compose -f docker-compose.monitoring.yml up -d
```

---

## Phase 8: Post-Deploy Verification

### Step 17 — Endurance test

```bash
# Run a 5-minute endurance test:
docker exec mousedroid python3 scripts/endurance_test.py --duration 300
```

### Step 18 — Update deployment record

After successful validation, update the deployment record:

```bash
cd /opt/mousedroid
# Update deployments/jetson-image.json with:
# - New SHA from current HEAD
# - Current timestamp
# - Notes about this deployment
git add deployments/jetson-image.json
git commit -m "chore(deploy): update Jetson deployment record to $(git rev-parse --short HEAD)"
```

### Step 19 — Smoke the dashboard

```bash
# From the Windows host, open browser to:
# http://<jetson-ip>:8080/
# Verify:
# - LiDAR polar plot loads
# - Camera feed displays (if CSI connected)
# - Metrics endpoint responds: curl http://<jetson-ip>:8080/metrics
# - WebSocket stream: ws://<jetson-ip>:8080/ws/v1/lidar/raw
```

---

## Known Issues & Workarounds

| Issue | Status | Workaround |
|---|---|---|
| ESP32 dead | P0 blocker (F-008) | `MOUSEDROID_ESP32__ENABLED=false` — no motor control |
| iGPU memory contention | Known | `n_gpu_layers=0` for LLM (CPU fallback) |
| `/opt/mousedroid` ownership drift | Recurring | `sudo chown -R ian:ian /opt/mousedroid` before git ops |
| CSI camera requires `argus_socket` | By design | Docker volume mount handles this |
| LLM inference slow on CPU | Known (260s) | GPU offload blocked by world model memory |

---

## Rollback Plan

If the deployment fails:

```bash
# 1. Stop the container:
docker compose -f docker-compose.jetson.yml down

# 2. Revert to previous source:
cd /opt/mousedroid
git checkout <previous-branch>

# 3. Restore previous docker.env:
# (host_bootstrap.sh creates backups)
sudo cp /etc/mousedroid/docker.env.bak /etc/mousedroid/docker.env

# 4. Restart:
docker compose -f docker-compose.jetson.yml up -d
```
