---
name: edge-hardware-health
description: Audit edge hardware devices, USB-C dynamic discovery, LiDAR scan buffers, CSI camera frames, and motor controller health.
---

# Edge Hardware Health Skill

Workflow for diagnosing physical and simulated edge hardware components.

## Target Paths

- Motor Controller: `src/mousedroid/hardware/motor_controller.py`
- LiDAR Driver: `src/mousedroid/hardware/lidar_driver.py`
- CSI Camera: `src/mousedroid/hardware/camera_csi.py`
- Hardware Schema: `src/mousedroid/config/schema/hardware.py`

## Execution Steps

1. Audit USB-C discovery resolution via `mousedroid.factory.build_motor_controller`.
2. Check LiDAR health status and scan ingestion via `lidar.get_latest_scan()`.
3. Check CSI camera frame capture via `camera.capture_frame()`.
4. Check motor velocity clamping bounds and e-stop responsiveness.
