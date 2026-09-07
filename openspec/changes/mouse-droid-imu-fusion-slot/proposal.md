# Proposal — Optional IMU fusion slot (default off)

- change_id: mouse-droid-imu-fusion-slot
- project: mouse-droid
- status: active
- feature_id: F-040
- epic: World model
- owner: ianshank
- created: 2026-09-07
- rev: A

## Why

F-036 parsed stock IMU r/p/y onto `EncoderReading` without an RSSM slot.
A sixth fusion slot is an ONNX mask-width change, not a schema toggle.
This change adds the lidar-shaped slot with `imu_dim=0` so the default
path does not resize `motor_proj` or `encoder.fusion`.

## What Changes

- `SENSOR_SLOT_MAP["imu"]=5`, packed `valid_mask` width 6.
- `ModelConfig.imu_dim` default 0, `imu_proj_dim` default 32.
- Encoder / packer / ONNX I/O / RSSM decoders / checkpoint migration
  grow only when `imu_dim>0`.
- Sensing copies F-036 attitude into `imu_features` when `imu_valid`.

## Impact

ONNX `valid_mask` input is `(1, 6)` even with IMU off (padded zeros).
Fusion weights are unchanged at `imu_dim=0`. Existing 4- and 5-wide
masks pad. `motor_state_dim` stays 4.

## Charter

No CHARTER §3 carve-out: default-OFF modality; no new actuation.

## Validation

`bash scripts/validations/F-040.sh`
