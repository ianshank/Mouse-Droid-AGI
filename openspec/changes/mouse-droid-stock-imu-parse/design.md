# Design — Stock FEEDBACK_BASE_INFO IMU parse

## D-1. Explicit attitude fields, not `heading_rad = y`

Odometry heading and IMU yaw are different quantities. Stuffing yaw into
`heading_rad` inside the codec would make a later odometry source
unrepresentable. `heading_for_motor()` is the single sensing seam.

## D-2. `imu_valid` rather than "non-zero yaw"

A rover sitting at yaw 0.0 is a legal attitude. Treating 0 as "missing"
would drop real headings. `imu_valid=True` on any T-gated 1001 frame.

## D-3. No SENSOR_SLOT_MAP key

Lidar already occupies optional slot 4; packer width is 5 for ONNX.
Adding `"imu": 5` without bumping packer width drops the slot. Fusion
is F-040.

## D-4. F-025 split stands

The codec parses; sensing consumes. Motor MCP `read_encoders` reports
both the raw attitude fields and `heading_rad` as `heading_for_motor()`
so operators see the same heading the world model gets.

## D-5. Units

Stored as radians to match `heading_rad`. Vendor `r`/`p`/`y` are passed
through `_coerce_float` without a degrees conversion — a wrong
conversion is worse than documenting the assumption.
