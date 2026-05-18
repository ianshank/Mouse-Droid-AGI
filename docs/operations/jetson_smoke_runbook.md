# Jetson Hardware Smoke Runbook

Operator-side playbook for verifying the rover end-to-end after a hardware change. Covers the IMX500 camera (post-ribbon/lens/refocus adjustment), the NVMe SSD on PCIe, and the Hailo-8 accelerator.

## When to use this

Run this whenever you've physically touched the rover:

- IMX500 camera ribbon-cable re-seat or lens refocus
- NVMe SSD reseat (M.2 slot)
- Hailo-8 M.2 card reseat
- JetPack upgrade — kernel module names and `/dev/` paths drift between releases

## Pre-flight (from the workstation)

Run these BEFORE pushing the branch. Bail and re-deploy if any pre-flight fails — running smokes against a half-broken rover wastes time chasing the wrong signal.

```bash
# 1. Link check + SSH key check (BatchMode=yes forces non-interactive).
ping -c 2 192.168.55.1
ssh -o BatchMode=yes jetson@192.168.55.1 'echo ssh_ok'

# 2. Confirm the deploy directory + venv exist on the rover.
ssh jetson@192.168.55.1 \
  'test -d /opt/mousedroid && test -x /opt/mousedroid/venv/bin/python && echo deploy_ok'

# 3. Check whether a mousedroid systemd service is running. If yes, STOP IT
#    before the smoke — the orchestrator holds exclusive locks on /dev/video0
#    (camera) and /dev/hailo0 (Hailo PCIe), so the smoke would FAIL on those
#    sensors with "Device or resource busy" / "device locked" errors.
ssh jetson@192.168.55.1 'systemctl is-active mousedroid 2>/dev/null || echo no-service'
# If output is "active":
ssh jetson@192.168.55.1 'sudo systemctl stop mousedroid'
# Remember to restart it after the smoke (see end of this doc).
```

## Push the branch to the rover

Use rsync with the repo's `.gitignore` as the filter so the workstation's venv, build artifacts, coverage data, and caches don't pollute the rover. The rover's runtime venv lives at `/opt/mousedroid/venv` (NOT `.venv`) — the `--filter`/`--exclude` combo below preserves it.

```bash
# From the new worktree on the workstation:
rsync -av --delete \
    --filter=':- .gitignore' \
    --exclude='.git/' \
    --exclude='venv/' --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.egg-info/' \
    --exclude='.coverage' --exclude='.coverage.*' \
    --exclude='coverage-branch.json' \
    --exclude='.benchmarks/' --exclude='.hypothesis/' \
    --exclude='.pytest_cache/' --exclude='.mypy_cache/' --exclude='.ruff_cache/' \
    ./ jetson@192.168.55.1:/opt/mousedroid/
```

## Run the three smokes

The bash harness self-resolves Python via `VENV_DIR=/opt/mousedroid/venv` — no `source venv/bin/activate` needed.

```bash
ssh jetson@192.168.55.1 \
  'cd /opt/mousedroid && bash scripts/jetson_smoke_test.sh camera' | tee ~/smoke-camera-$(date +%s).log
ssh jetson@192.168.55.1 \
  'cd /opt/mousedroid && bash scripts/jetson_smoke_test.sh pcie_ssd' | tee ~/smoke-pcie_ssd-$(date +%s).log
ssh jetson@192.168.55.1 \
  'cd /opt/mousedroid && bash scripts/jetson_smoke_test.sh hailo' | tee ~/smoke-hailo-$(date +%s).log
```

The `tee` calls land transcripts in the workstation home dir so you can grep them post-hoc. The harness `exit "${FAILURES}"` — non-zero exit on partial failure is expected; check the in-line `FAIL:` lines for specifics.

## Capture + review the camera snapshot

The bash harness camera test exits PASS/FAIL but does NOT save a JPEG. To get the visual evidence after a ribbon-cable / lens adjustment, invoke the sensor script directly with `--save-frame`. Use the rover's venv-pinned interpreter (NOT bare `python`, which is the system Python without project deps). Write into `/opt/mousedroid/snapshots/` (deploy-stable) rather than `/tmp` (volatile across reboots).

```bash
# Ensure the snapshots dir exists + is writable.
ssh jetson@192.168.55.1 'mkdir -p /opt/mousedroid/snapshots'

# Capture three frames; the LAST one lands as the JPEG.
STAMP="$(date +%Y%m%d-%H%M%S)"
ssh jetson@192.168.55.1 \
  "/opt/mousedroid/venv/bin/python /opt/mousedroid/scripts/verify_sensors.py \
     --config /opt/mousedroid/config/jetson_production.yaml \
     --sensor camera --save-frame /opt/mousedroid/snapshots/snap-${STAMP}.jpg --frames 3"

# Pull it back to the workstation for visual inspection.
scp "jetson@192.168.55.1:/opt/mousedroid/snapshots/snap-${STAMP}.jpg" "./post_adjust_snapshot-${STAMP}.jpg"
```

What to look for in the JPEG:

- **Focus sharpness** on a known-distance target (a 1 m test chart is the standard reference).
- **Exposure** — neither blown highlights nor crushed shadows.
- **Framing** — center subject is actually centered. Off-axis crop is the signature of a mis-seated ribbon.
- **No rainbow / banding** at the edges. EMI banding is the signature of a partially-seated ribbon.

## Interpret PASS / SKIP / FAIL

| Output line | Meaning | Operator action |
|-------------|---------|-----------------|
| `[PASS] frame capture` | Camera capture succeeded at the expected resolution | None — proceed |
| `[PASS] snapshot saved` | JPEG landed on disk at the printed path | SCP back to the workstation |
| `[FAIL] frame shape` | Resolution mismatch | Check `camera.resolution_width/height` in YAML overlay |
| `[PASS] nvme device found` | `lspci` enumerated the NVMe drive | None |
| `[SKIP] nvme device found` | `lspci` returned no NVMe device OR `lspci` not installed | If `lspci` is installed, the drive isn't seen by PCIe — see "Safe reseat protocol" below |
| `[PASS] mount + capacity` | NVMe is mounted with enough free space | None |
| `[FAIL] capacity below required` | Free space is below `cfg.experience.map_size_gb` (default 20 GB) | See "Common failure modes" below |
| `[PASS] smartctl health` | SMART health check PASSED | None |
| `[SKIP] hailo accelerator` | `cfg.hailo.enabled=False` (the default) | Set `enabled: true` in YAML overlay to opt in |
| `[SKIP] hailo device` | `/dev/hailo0` not present | See "Common failure modes" below |
| `[SKIP] hailo SDK` | `hailo_platform` not importable | `pip install -e ".[hailo]"` on the rover |
| `[FAIL] inference latency` | One inference exceeded `cfg.hailo.timeout_ms` | See "Common failure modes" below |

## Common failure modes

### `/dev/hailo0 not present` (SKIP on Hailo)

Find the actual module name first — it varies between JetPack / Hailo-SW releases (`hailo_pci`, `hailo`, or `hailort` depending on the DKMS package):

```bash
ssh jetson@192.168.55.1 'lsmod | grep -i hailo'
ssh jetson@192.168.55.1 'dmesg | grep -i hailo | tail -20'
```

- **Module loaded but `/dev/hailo0` absent:** kernel saw the PCIe device but couldn't enumerate it — hardware-side, see the safe reseat protocol below.
- **Module missing entirely:** check the installed package (`apt list --installed 2>/dev/null | grep hailort`) and `sudo modprobe <module-name-from-lsmod>`.

### `nvme device found` SKIP

The NVMe drive isn't enumerated on PCIe. Confirm via `lspci`:

```bash
ssh jetson@192.168.55.1 'lspci -nn | grep -iE "nvme|non-volatile"'
```

If empty, the drive isn't seen by the PCIe controller — reseat (see safety steps below).

### `mount target` SKIP — non-standard mount location

The PCIe smoke resolves the SSD mount via this chain (highest priority first):

1. `$MOUSEDROID_SSD_MOUNT` environment variable — explicit operator override
2. `findmnt -no TARGET /dev/nvme0n1p1` — auto-discovery via the configured partition

If your SSD is mounted at a non-standard location (e.g. `/data` instead of `/mnt/ssd`), set the env override so the smoke can find it:

```bash
ssh jetson@192.168.55.1 \
  'MOUSEDROID_SSD_MOUNT=/data bash scripts/jetson_smoke_test.sh pcie_ssd'
```

To persist the override across runs, add it to `/etc/mousedroid/docker.env` (the env-file sourced by `docker-compose.jetson.yml`) or `/etc/environment`.

If your NVMe partition layout differs (e.g. ESP first, ext4 second), override the device paths in your YAML overlay:

```yaml
experience:
  nvme_device: /dev/nvme0n1       # smartctl target
  nvme_partition: /dev/nvme0n1p2  # findmnt target (non-canonical partition)
```

### `frame shape` FAIL — running with mock hardware

When `MOUSEDROID_MOCK_HARDWARE=true` (developer host without the IMX500), the `MockCamera` procedurally generates 320×240 frames. The default config expects 640×480, so the shape check FAILs. Either:

- Run the smoke on the actual Jetson (the common case — this runbook is intended for on-rover verification).
- Or override the resolution in your dev overlay to match the mock: `camera.resolution_width=320`, `camera.resolution_height=240`.

### `capacity below required` FAIL

Free space is below `cfg.experience.map_size_gb` (default 20 GB). Diagnose before deleting:

```bash
ssh jetson@192.168.55.1 \
  'du -sh /opt/mousedroid/weights/cloud_updates /opt/mousedroid/var/harness/journal /mnt/ssd/* 2>/dev/null | sort -h'
```

Remediations:

- Archive old `cloud_updates/<revision>/` directories to GCS via `training/upload_weights.py` rather than deleting blindly (those weights may be needed for rollback).
- `sudo apt clean` to free APT cache.
- Expand the partition (last resort — requires a reboot).

Do **NOT** `rm -rf` weight directories without confirming the active revision is preserved.

### `inference latency exceeded timeout` FAIL

Either `cfg.hailo.timeout_ms` is too tight OR the Hailo card is thermally throttled. Check thermals:

```bash
ssh jetson@192.168.55.1 'cat /sys/class/thermal/thermal_zone*/temp'
```

Each value is in millidegrees Celsius. If any zone is > 80 000, the card is throttling — improve airflow or reseat the heatsink. Otherwise tune `cfg.hailo.timeout_ms` upward in the YAML overlay.

## Safe reseat protocol (HARDWARE — power off first)

```bash
ssh jetson@192.168.55.1 'sudo shutdown -h now'
# Wait ~30 s for the green power LED to go solid amber / off.
# Unplug the barrel-jack power cable.
# Touch a grounded metal surface to discharge static.
# Unscrew the M.2 retention screw, lift the card at ~30°, reseat fully.
# Replace the screw, plug power back in, boot.
```

**NEVER hot-reseat** an M.2 card — the PCIe controller on the Jetson Orin Nano does not support hotplug, and you will permanently kill the slot or the card.

## Post-smoke cleanup

Restart the orchestrator service (if you stopped it in pre-flight):

```bash
ssh jetson@192.168.55.1 'sudo systemctl start mousedroid && systemctl is-active mousedroid'
```

Archive the smoke logs:

```bash
mv ~/smoke-*.log ./smoke-logs-$(date +%Y%m%d)/
```
