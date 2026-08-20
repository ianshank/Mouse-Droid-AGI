# Robot Arm Subsystem — Surface Contract (Parked Platform)

> Hierarchical robot-arm training platform (MuJoCo Gymnasium, SAC+HER reinforcement learning,
> PDDL symbolic replanner, and SO-ARM100 hardware driver).

## Governance Notice & Freeze Status

> **F-008 Capability Freeze**: Modifications under `src/mousedroid/arm/**` are gated by
> PreToolUse hook `freeze_gate.py` while feature F-008 is not `done`. Rover bring-up and
> base platform stability take strict priority.

## Invariants & Architecture Rules

1. **Dual Cadence Planning**: Symbolic PDDL task planning operating above continuous
   SAC+HER trajectory controllers.
2. **Deterministic Simulation Seed**: MuJoCo Gymnasium environments initialize with explicit
   random seeds configured in `ArmConfig`.
3. **Hardware Driver Isolation**: Real SO-ARM100 hardware drivers remain in `hardware/` and are
   mocked via `mock_arm.py` during simulation training.

## Key Files

- `perception/` — Depth camera, YOLO detection, 6-DoF pose estimation.
- `planning/` — PDDL symbolic planner and LLM replanner.
- `control/` — SAC+HER policy and trajectory generation.
- `environments/` — MuJoCo Gymnasium environments.
