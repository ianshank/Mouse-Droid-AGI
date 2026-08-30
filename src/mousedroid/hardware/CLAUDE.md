# Hardware Subsystem — Surface Contract

> Physical sensor and actuator drivers for NVIDIA Jetson Orin Nano, CSI/USB cameras,
> LiDAR, IMU, ultrasonic, and ESP32 motor controller bridge.

## Invariants & Hardware Rules

1. **Factory DI Only**: Hardware drivers are imported solely in `src/mousedroid/factory/hardware.py`.
   Business logic depends only on `@runtime_checkable Protocol` types.
2. **Two-Level Hardware Gates**:
   - `Settings.mock_hardware: bool` (env `MOUSEDROID_MOCK_HARDWARE`) — Global mock toggle.
   - Subsystem `enabled: bool` (e.g. `esp32.enabled`, `lidar.enabled`) — Fine-grained dev escape hatches.
3. **USB-C Endpoint Discovery**: New USB-C devices are registered in `config/jetson_production.yaml`
   under `usbc_discovery.required_endpoints`.
4. **Ring Buffers**: Sensor historical buffers use `collections.deque(maxlen=N)` where `N` is
   read from schema config.
5. **Hardware Test Tier**: Tests under `tests/hardware/` run against real hardware when
   `tests._jetson_hardware.is_jetson_host` is true. Never mock in hardware tests.
6. **Robust Sysfs Reading**: All sysfs file operations must use `encoding="utf-8", errors="replace"`.

## Key Files

- `camera/` — CSI and USB camera drivers (`jetson_csi.py`, `opencv_camera.py`).
- `lidar/` — RPLiDAR serial driver.
- `ultrasonic/` — Ultrasonic distance sensor driver.
- `tests/unit/hardware/` & `tests/hardware/` — Unit and live hardware verification.
