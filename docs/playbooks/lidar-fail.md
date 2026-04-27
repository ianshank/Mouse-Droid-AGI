# LiDAR Failure Playbook

Use this playbook when the LD19 stage fails in smoke, validation, or runtime verification.

## First Checks

1. Verify the expected serial device exists on the Jetson host:
   `ls -l /dev/serial/by-id`
   `ls -l /dev/ttyUSB*`
2. Confirm the configured LiDAR block:
   `sudo sed -n '/^lidar:/,/^camera:/p' /etc/mousedroid/jetson_production.yaml`
3. Review the recent runtime logs:
   `docker logs mousedroid --tail 200 | grep -i "lidar\|ld19\|serial"`

## Common Causes

- Wrong serial path in `config/jetson_production.yaml`
- USB adapter missing after reboot or cable reseat
- Baud-rate mismatch
- Partial scans because the runtime timeout or coverage values are too strict for the bench setup

## Remediation Steps

1. Confirm the production config is synced:
   `systemctl status mousedroid-docker`
   `sudo /opt/mousedroid/scripts/sync_jetson_overlay.sh`
2. Check the configured values:
   - `lidar.serial_port`
   - `lidar.baud_rate`
   - `lidar.scan_acquisition_timeout_s`
   - `lidar.min_scan_coverage_deg`
3. Re-seat the LD19 USB adapter and restart the service:
   `systemctl restart mousedroid-docker`
4. Re-run the narrow checks:
   `python scripts/verify_sensors.py --sensor lidar`
   `bash scripts/jetson_smoke_test.sh lidar`

## Exit Criteria

- `verify_sensors.py --sensor lidar` passes
- `jetson_smoke_test.sh lidar` passes
- Container logs show stable scan acquisition without repeated timeout or stale-scan errors