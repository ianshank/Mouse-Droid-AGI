---
description: Run simulation tests for the robot arm platform
status: frozen
unfreeze: "F-008 done on the Jetson + 30-day Phase-3b production soak (see NEXT_STEPS.md)"
---

# Simulation Test

Run the robot arm simulation test suite to validate environment, perception, planning, and control.

## Arguments

- `$ARGUMENTS` — test scope: `all`, `env`, `perception`, `planning`, `control`, `e2e` (default: `all`)

## Test Categories

### Environment Tests (`env`)
- MuJoCo scene loads and physics are stable (no drift)
- Robot URDF renders with correct DOF
- Gymnasium `reset()`/`step()` produce valid observations
- Domain randomization applies within configured ranges
- Deterministic rollouts with fixed seed

### Perception Tests (`perception`)
- Mock depth images produce correct object detections
- Symbolic state extraction matches expected disk-to-peg mapping
- Pose estimation error < 5mm on synthetic data

### Planning Tests (`planning`)
- PDDL generates optimal move sequence (2^n - 1 moves)
- No illegal moves in plan (smaller-on-larger constraint)
- Replanner generates valid recovery plan from error states

### Control Tests (`control`)
- Reward function returns correct values for grasp/place/collision
- Trajectory stays within joint limits
- Mock arm driver executes commanded trajectories

### End-to-End Tests (`e2e`)
- Full pipeline: perception -> planning -> control in simulation
- 3-disk Tower of Hanoi completes within step budget

## Commands

```bash
# All simulation tests
pytest tests/unit/arm/ tests/integration/arm/ -v

# Specific category
pytest tests/unit/arm/ -k "env" -v
pytest tests/unit/arm/ -k "planning" -v

# With coverage
pytest tests/unit/arm/ --cov=src/mousedroid/arm --cov-report=term-missing
```
