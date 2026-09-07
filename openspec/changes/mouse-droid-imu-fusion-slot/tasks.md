# Tasks — Optional IMU fusion slot

Quality gate:

```
python -m ruff check src/mousedroid/world_model src/mousedroid/constants.py src/mousedroid/config/schema/world_model.py src/mousedroid/sensing tests/regression/test_f040_aqa.py
bash scripts/validations/F-040.sh
```

**Phase 1 — Schema + constants**

- [x] 1.1 `imu_dim=0`, `imu_proj_dim=32`, slot 5, packed width 6.
- [x] 1.2 `motor_state_dim` stays 4.

**Phase 2 — Encoder / packer / ONNX**

- [x] 2.1 Encoder IMU branch after LiDAR; default has no `imu_proj`.
- [x] 2.2 Packer emits `imu=None` when `imu_dim=0`.
- [x] 2.3 ONNX optional input `imu` only when `imu_dim>0`.

**Phase 3 — Sensing consumer**

- [x] 3.1 Bundle `imu_features`; manager copies F-036 r/p/y when `imu_valid`.
