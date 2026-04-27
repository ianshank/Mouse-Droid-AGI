# Camera Failure Playbook

Use this playbook when the ribbon camera fails runtime validation or the camera smoke stage.

## First Checks

1. Review the camera failure detail from `scripts/verify_sensors.py`; the runtime helper now adds
   an operator-facing diagnosis when no `/dev/video*` node is present.
2. Inspect the Jetson-side camera nodes:
   `ls -l /dev/video* /dev/media*`
3. Check the latest Argus logs:
   `journalctl -u nvargus-daemon --no-pager -n 60`

## Common Causes

- Ribbon cable seated incorrectly
- Wrong device-tree overlay for the attached camera
- `camera.device_path` drift in the production overlay
- Container restarted before the camera stack was ready

## Remediation Steps

1. Confirm the active production config:
   `sudo sed -n '/^camera:/,/^voice:/p' /etc/mousedroid/jetson_production.yaml`
2. Verify the deployment path is service-managed, or resync manually if running ad hoc:
   `sudo /opt/mousedroid/scripts/sync_jetson_overlay.sh`
3. Re-seat the ribbon cable and verify the correct device-tree overlay in `/boot/extlinux/extlinux.conf`.
4. Restart the service after the camera stack is ready:
   `systemctl restart mousedroid-docker`
5. Re-run the focused checks:
   `python scripts/verify_sensors.py --sensor camera`
   `bash scripts/jetson_smoke_test.sh camera`

## Exit Criteria

- A `/dev/video*` or Jetson CSI camera path is visible again
- `verify_sensors.py --sensor camera` passes
- `jetson_smoke_test.sh camera` passes