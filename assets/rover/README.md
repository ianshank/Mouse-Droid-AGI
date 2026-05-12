# Rover assets

Sim-to-real assets for the MSE-6 4WD rover platform. Consumed by the Isaac
Lab environment stub in `src/mousedroid/sim/isaaclab/rover_env.py` and by
any downstream MuJoCo / RViz / URDF viewer.

## Files

- `mse6_4wd.urdf` — 4WD chassis with an MSE-6 shell modelled as a single
  rigid body. Four `continuous` wheel joints (no angle limits) so the
  differential-drive policy can command unbounded wheel rotation. Three
  fixed sensor frames (`imu_link`, `lidar_link`, `camera_link`) are
  pre-wired so the env stub can attach sensors without modifying the
  URDF.
- `meshes/` — placeholder; the URDF currently ships primitive `<box>` and
  `<cylinder>` visuals so it loads in any viewer without external
  meshes. Drop real OBJ/STL files here and switch the URDF visuals to
  `<mesh filename="meshes/..."/>` when CAD is ready.

## Units & conventions

- SI units throughout: meters, kilograms, radians.
- Body frame: **x forward, y left, z up** (REP-103 / ROS convention).
- Wheel order in code: front-left, front-right, rear-left, rear-right.
  Left-side wheels (`wheel_fl`, `wheel_rl`) share the left differential
  drive command; right-side wheels (`wheel_fr`, `wheel_rr`) share the
  right. The `RoverIsaacLabEnv` maps the 2-D action vector
  `[left_wheel_rad_s, right_wheel_rad_s]` to all four joints.

## Inertial defaults

The URDF carries documentation-quality defaults for a 230 mm × 180 mm
chassis with a 0.85 kg hollow MSE-6 shell (3 mm wall, 20% infill). At
runtime, `RoverInertialConfig` (see
`src/mousedroid/config/schema.py`) overrides:

| Field                  | Effect                                            |
|------------------------|---------------------------------------------------|
| `shell_mass_kg`        | Total shell mass added to `base_link`             |
| `shell_thickness_m`    | Inertia tensor recomputed for a hollow shell      |
| `shell_infill`         | Density scaling for the shell mass distribution   |
| `com_offset_xyz_m`     | Vertical bias for the top-heavy roll dynamics     |
| `wheel_mass_kg`        | Per-wheel mass + inertia                          |

A top-heavy COM is intentional — the policy needs to experience roll
during cornering in sim so it generalises to the physical droid.

## Regeneration

There is no generator script yet. When real CAD (Fusion 360 / Onshape /
FreeCAD) is exported:

1. Export each link as STL or OBJ into `meshes/`.
2. Re-derive the inertia tensors with the CAD package's mass-properties
   tool (use the same `shell_*` parameters as `RoverInertialConfig`).
3. Replace the primitive `<box>` / `<cylinder>` visuals in the URDF
   with `<mesh filename="meshes/<name>.obj"/>` blocks. Keep the
   collision geometry as primitives unless the physics engine needs
   convex decomposition.
4. Validate with `check_urdf assets/rover/mse6_4wd.urdf` — all joints
   should report type `continuous` and inertia tensors should be
   positive-definite.
