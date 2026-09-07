# Tasks — Stock FEEDBACK_BASE_INFO IMU parse

Quality gate for every task below, run before it is ticked:

```
python -m ruff check src/mousedroid/comms src/mousedroid/sensing src/mousedroid/common/tools/motor_tools.py tests/unit/comms tests/unit/sensing tests/unit/common/test_motor_tools.py tests/regression/test_f036_aqa.py tests/regression/test_f036_backwards_compat.py
bash scripts/validations/F-036.sh
```

Task ordering is binding.

**Phase 1 — Dataclass + codec**

- [x] 1.1 `EncoderReading` attitude fields default inert + `heading_for_motor`.
- [x] 1.2 Stock `parse_encoders` copies `r`/`p`/`y`; wrong T keeps `imu_valid=False`.
- [x] 1.3 Legacy codec / `parse_encoder_reading` stay IMU-inert.

**Phase 2 — Sensing consumer**

- [x] 2.1 `SensorManager` motor vector uses `heading_for_motor()`.
- [x] 2.2 Motor MCP payload includes attitude + `imu_valid`.

**Phase 3 — Tests + catalog**

- [x] 3.1 Fake T=1001 unit tests.
- [x] 3.2 Regression pair `test_f036_aqa.py` + `test_f036_backwards_compat.py`.
- [x] 3.3 `scripts/validations/F-036.sh` + `features.yaml` F-036.
