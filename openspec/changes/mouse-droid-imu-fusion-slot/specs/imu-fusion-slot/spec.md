# IMU fusion slot

## Purpose

Add an optional last RSSM modality for IMU attitude without changing
default fusion weights.

## Requirements

### Requirement: `imu_dim` SHALL default to 0

Existing YAML SHALL load. `MultimodalEncoder` SHALL omit `imu_proj`
when `imu_dim==0` or `imu_proj_dim==0`.

### Requirement: packed `valid_mask` SHALL be width 6

The packer SHALL pad or truncate to
`N_SENSOR_MODALITIES_WITH_IMU`. Slot 5 SHALL be IMU.

### Requirement: `motor_state_dim` SHALL remain 4

IMU attitude SHALL NOT be concatenated onto the motor vector.
