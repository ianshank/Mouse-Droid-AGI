# On-Device Full Validation Surface (PR #116 Discipline)

> Consolidated entry point for validating MouseDroid on NVIDIA Jetson Orin Nano hardware.

## Entry Point

```bash
bash scripts/jetson_full_validation.sh
```

## Validation Discipline

1. **Composition Over Reimplementation**: The script composes existing tooling (`ci.sh`,
   `verify_sensors.py`, `jetson_smoke_test.sh`, `translate_mission.py`, `lidar_telemetry_probe.py`,
   `validate_pillars`).
2. **Cold-Then-Warm Execution**:
   - *Cold Phase*: Exclusive-device checks (LiDAR, CSI camera, GPIO) execute with the Docker
     container stopped. A shell `trap` ensures the container is restarted upon exit or failure.
   - *Warm Phase*: Service-level checks (`/api/v1/health`, live `/metrics`, mission ingress)
     execute against the running container.
3. **Validate-Around Inoperative Hardware**: Subsystems with offline hardware emit non-blocking
   warnings (`WARN`) rather than hard failures (`FAIL`). Motor motion is gated behind both
   `MOUSEDROID_SMOKE_ALLOW_MOTION` and `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION`.
4. **Environment-Driven Configuration**: Ports, timeouts, memory caps, and metric namespaces
   are env-overridable (`MOUSEDROID_VALIDATION_*`, `MOUSEDROID_METRICS__NAMESPACE`).
5. **Memory-Guarded Execution**: `scripts/ci.sh` runs with `ulimit -v` and automatic retry
   in slim mode (`MOUSEDROID_CI_SLIM=1`) upon OOM (rc=137).
