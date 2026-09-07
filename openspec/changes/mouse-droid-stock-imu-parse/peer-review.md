# Peer review — Stock FEEDBACK_BASE_INFO IMU parse

## Verdict table

| Claim | Verdict |
|---|---|
| Stock codec previously zeroed heading and skipped `r`/`p`/`y` | **CONFIRMED** — `WaveshareStockCodec.parse_encoders` |
| Adding `SENSOR_SLOT_MAP["imu"]=5` is an ONNX width change | **CONFIRMED** — packer mask width 5 |
| Default `command_set="legacy"` stays IMU-inert | **CONFIRMED** — `parse_encoder_reading` does not set `imu_valid` |
| Filling `heading_rad=y` inside the codec conflates quantities | **CONFIRMED** — explicit fields + `heading_for_motor` is the rebuttal |

## What survives review unchanged

- F-025 codec/sensing split.
- No production YAML battery-threshold flip.
- No CHARTER §3 carve-out (default path unchanged).

## Load-bearing pins

1. `EncoderReading()` has `imu_valid is False` and `heading_for_motor() == 0.0`.
2. Legacy frames with extra `y` keys still report odometry heading.
3. Stock T=1001 with `y=1.57` yields `heading_for_motor() == 1.57` and `heading_rad == 0.0`.
4. `"imu" not in SENSOR_SLOT_MAP`.
