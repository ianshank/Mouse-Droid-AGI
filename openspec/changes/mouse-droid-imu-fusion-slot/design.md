# Design — Optional IMU fusion slot

## D-1. Lidar-shaped, last slot

LiDAR already occupies optional slot 4 with `lidar_dim=0` dropping the
branch. IMU occupies slot 5 the same way. Concat order stays
vision → ultrasonic → motor → audio → lidar → imu so default fusion
columns are a prefix of the IMU-on layout.

## D-2. Packer width is always 6

ORT needs a stable `valid_mask` shape. Width 6 with a trailing zero is
the IMU-off contract. 4-wide and 5-wide observations right-pad.

## D-3. Do not widen motor_state_dim

Heading vs IMU yaw are different quantities (F-036). Attitude lives in
the IMU slot, not `[vx, vy, omega, battery, roll, pitch, yaw]`.
