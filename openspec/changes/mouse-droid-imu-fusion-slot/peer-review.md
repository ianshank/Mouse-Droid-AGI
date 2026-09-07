# Peer review — Optional IMU fusion slot

## Verdict table

| Claim | Verdict |
|---|---|
| Adding a map key without bumping packer width drops the slot | **CONFIRMED** — packer now width 6 |
| `imu_dim=0` keeps fusion in_features | **CONFIRMED** — AQA + backwards-compat |
| Widening `motor_state_dim` 4→7 is checkpoint-incompatible | **CONFIRMED** — not done |
| Hardware validation still waits on F-008 | **CONFIRMED** |

## Load-bearing pins

1. `ModelConfig.imu_dim` FieldInfo default 0.
2. `SENSOR_SLOT_MAP["imu"] == 5`.
3. Default encoder has no `imu_proj`.
