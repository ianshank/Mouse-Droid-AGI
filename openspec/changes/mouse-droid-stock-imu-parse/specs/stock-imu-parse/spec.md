# Stock IMU parse (F-036)

## Requirement: stock T=1001 frames expose IMU attitude

The WAVE ROVER stock `FEEDBACK_BASE_INFO` frame carries `r`/`p`/`y`. The
rover MUST parse those keys into `EncoderReading` attitude fields without
widening `SENSOR_SLOT_MAP`.

### Scenario: stock frame with yaw

- **GIVEN** `command_set="waveshare_stock"` and a T=1001 frame with `y=1.57`
- **WHEN** `parse_encoders` runs
- **THEN** `yaw_rad==1.57`, `imu_valid is True`, `heading_rad==0.0`, and
  `heading_for_motor()==1.57`

### Scenario: legacy path

- **GIVEN** `command_set="legacy"`
- **WHEN** a frame is parsed
- **THEN** `imu_valid is False` and `heading_for_motor()` equals `heading_rad`

### Scenario: no fusion slot

- **GIVEN** F-036 has landed
- **THEN** `"imu" not in SENSOR_SLOT_MAP` and `N_SENSOR_MODALITIES_WITH_LIDAR==5`
