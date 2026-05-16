# ADR-009 — Isaac Lab Phase B (Real-Env Wiring for the Rover)

**Status:** Accepted (foundation laid; full wiring iterates on operator-side
Linux/Isaac Sim host)
**Date:** 2026-05-16
**Sprint:** Tier B Track B3 (`docs/planning/sprint_plan_tier_b.md`)

> **Note on numbering:** original sprint plan called this ADR-008. That
> number was claimed by the [World-Model ONNX Engine](./ADR-008-world-model-onnx-engine.md)
> ADR (Track B2). Isaac Lab Phase B is filed as ADR-009.

---

## Context

The rover (MSE-6 4WD chassis, `assets/rover/mse6_4wd.urdf`) trains in a
sim-to-real loop. Phase A (already merged) shipped a Phase-A stub at
[`src/mousedroid/sim/isaaclab/rover_env.py`](../../src/mousedroid/sim/isaaclab/rover_env.py)
with three `TODO(Phase B)` markers — `build()`, `reset()`, `step()` — all
returning mock data. The orchestrator can swap between the mock backend
([`MockRoverEnv`](../../src/mousedroid/sim/mock_rover_env.py)) and the
Isaac Lab backend via factory dispatch on `cfg.rover.sim.backend`.

Phase B's job is to replace those three stubs with a real
`ManagerBasedRLEnv` so the rover gets a photorealistic simulator
suitable for sim-to-real training — the gating dependency for Phase 5
(real-physics sim) and Phase 6 (on-device co-training).

## Decision

Adopt **Isaac Lab >= 0.20** as the Phase B real-physics simulator.

### Why Isaac Lab (not MuJoCo)?

| Concern | Isaac Lab | MuJoCo |
|---|---|---|
| Photorealistic camera rendering | ✅ Native | ⚠️ External (mujoco-viewer / dm_control) |
| RTX-accelerated LiDAR simulation | ✅ Native | ⚠️ External raycaster |
| URDF → simulator asset | URDF → USD (one-shot via Isaac Sim Launcher) | URDF → MJCF (script + manual tuning) |
| Domain randomization helpers | ✅ Built-in `RandomEnvCfg` | Hand-rolled |
| Production trajectory: Phase 5 → real Orin Nano | ⏯ Aligned (NVIDIA tooling) | ⚠️ Less direct |

Rover-specific decisive factor: **camera + LiDAR fidelity**. The rover
ships with an IMX500 camera and an LD19 LiDAR in production; the Phase 5
sim-to-real validation needs visually-plausible sensor data, which the
RTX rendering stack delivers natively.

### Asset pipeline

```
assets/rover/mse6_4wd.urdf  ← committed (human-readable source of truth)
        │
        ▼ scripts/convert_urdf_to_usd.py (one-shot, requires Isaac Sim)
        │
assets/rover/mse6_4wd.usd   ← committed (binary, ~few-MB, regenerable)
        │
        ▼ ArticulationCfg(usd_path=...) in RoverIsaacLabEnv.build()
        │
        ▼
   Isaac Lab ManagerBasedRLEnv scene
```

The `.usd` is committed (not generated on the fly) so CI runs without
Isaac Sim don't need to re-import. Operators re-run
`scripts/convert_urdf_to_usd.py` after any URDF change.

### Action space

4-wheel velocity control. The `ROVER_WHEEL_JOINT_NAMES` constants
(B3 Story 0, [`src/mousedroid/sim/isaaclab/constants.py`](../../src/mousedroid/sim/isaaclab/constants.py))
nail the joint name order: `(joint_wheel_fl, joint_wheel_fr,
joint_wheel_rl, joint_wheel_rr)`. The action-vector layout matches this
order.

For the **differential drive** mode (default in
`RoverActionConfig.mode = "differential"`), the policy outputs
`[left_wheel_rad_s, right_wheel_rad_s]`. The env's `step()` body
duplicates the value across each side's two wheels before forwarding to
the articulation actuators. This keeps Phase B compatible with the
mock backend's 2-D action contract without adding a new policy head.

Action clipping: `RoverActionConfig.max_wheel_rad_s` (default 25 rad/s)
clamps before forwarding — single source of truth shared with the
URDF, the safety layer, and the mock backend.

### Observation space

Mirrors `MockRoverEnv.observation_keys` exactly:
`("imu", "chassis_pose", "wheel_vel", "lidar")`. The
[cross-backend contract test](../../tests/unit/sim/isaaclab/test_constants.py)
asserts this contract holds in both directions.

**Explicitly excluded from the observation space:**

- HC-SR04 ultrasonic — parked sensor modality, forbidden from the rover
  baseline (`CLAUDE.md` invariant).
- Arm joint encoders — parked robot-arm platform.

### Reward function (Phase B baseline)

Simple forward-progress baseline:

```python
reward = (
    cfg.isaac_lab.reward.forward_velocity_weight * forward_velocity_mps
    - cfg.isaac_lab.reward.collision_weight * is_colliding
)
```

Defaults:
- `forward_velocity_weight: 0.01` (rewards ~1 m/s motion at +0.01 r/step)
- `collision_weight: 0.1` (firm penalty for collision)

Rationale: Phase B's reward must be **cheap** (no learned VLM signals)
so curriculum training can iterate fast. Richer reward shaping
(instruction following, multi-objective) lands in Phase 5 via the
existing `mousedroid.training.rover_reward` module.

Both weights are operator-tunable Pydantic fields. **No hardcoded
reward weights.**

### Domain randomization

Phase B reuses the existing `cfg.training.domain_randomization` Phase 1
baseline. The env's `reset()` calls into
`mousedroid.training.domain_randomization` helpers — no duplication.

Defaults: `domain_randomization.enabled: False` — preserves byte-identical
pre-B3 behavior. Operators flip the toggle when ready to train.

## What's in this PR (Foundation)

Limited to what's verifiable on a development workstation without
Isaac Sim. Operator-side validation on Ubuntu 22.04 + Isaac Sim 4.5+
closes out the remaining sub-tasks.

| Component | Status | File |
|---|---|---|
| `ROVER_WHEEL_JOINT_NAMES` + cross-backend constants | ✅ Story 0 | `src/mousedroid/sim/isaaclab/constants.py` |
| URDF -> USD conversion script | ✅ Story 1 | `scripts/convert_urdf_to_usd.py` |
| Conversion script smoke test (CI-skippable) | ✅ Story 1 | `tests/unit/sim/isaaclab/test_urdf_to_usd.py` |
| ADR-009 documenting decisions | ✅ Story 6 | _this file_ |

## What's deferred to a follow-up (operator-on-Linux PR)

Each remaining sub-task requires a live Isaac Sim installation for
end-to-end validation; writing the code on a Windows host without a way
to test the Isaac Lab API calls is unsafe.

| Sub-task | Story | Owner |
|---|---|---|
| Commit `assets/rover/mse6_4wd.usd` (run conversion script once) | 1 | Operator |
| `RoverIsaacLabEnv.build()` — scene + articulation + sensors | 2 | Operator |
| `RoverIsaacLabEnv.reset()` — domain randomization integration | 3 | Operator |
| `RoverIsaacLabEnv.step()` — action/observation mapping + reward | 4 | Operator |
| 9 unit tests under `pytest.importorskip("isaaclab")` | 5 | Operator |
| `IsaacLabRewardConfig` Pydantic block (defaults documented above) | 4 | Operator |
| 50-step random-rollout liveness smoke check | 5 | Operator |

The constants module + conversion script give the operator a
production-ready starting point. The smoke test under
`pytest.importorskip("isaaclab")` cleanly skips on CI runners without
Isaac Lab, so the wiring can land incrementally without breaking the
default test stage.

## Migration playbook (operator)

```bash
# 1. Install Isaac Sim via Omniverse Launcher (Linux only)
#    https://developer.nvidia.com/isaac-sim
# 2. Confirm import
python -c "import isaaclab; print(isaaclab.__version__)"
# 3. Convert the URDF once
python scripts/convert_urdf_to_usd.py \
    --urdf assets/rover/mse6_4wd.urdf \
    --output assets/rover/mse6_4wd.usd
# 4. Commit the produced .usd
git add assets/rover/mse6_4wd.usd
git commit -m "feat(sim): commit converted mse6_4wd.usd (B3 Story 1)"
# 5. Run the smoke test (will no longer skip)
pytest tests/unit/sim/isaaclab/test_urdf_to_usd.py -m slow -v
```

## Consequences

### Positive

- **Unblocks Phase 5 / Phase 6** by landing the photorealistic
  simulator the sim-to-real loop depends on.
- **Cross-backend contract guaranteed** — the same observation keys +
  action shape work against either `MockRoverEnv` or
  `RoverIsaacLabEnv` (verified by the existing Story 0 tests).
- **No CLAUDE.md invariant violations** — no arm references, no
  ultrasonic, all params from Pydantic config.

### Negative / out of scope

- **Linux-only path.** Windows users (the primary dev workstation in
  this repo) cannot validate end-to-end. Mitigated by the
  `pytest.importorskip` skip semantics so CI passes everywhere.
- **`.usd` binary in git history.** Single-file, ~few MB. Accepted
  trade-off; revisit if asset library grows (Git LFS).
- **Phase B reward is a baseline.** Sufficient for sim-to-real
  validation but not for solving complex tasks. Phase 5 ships the
  richer reward shaper.
- **Multi-rover scenes / arm integration / ultrasonic** — explicitly
  out of scope per the active production baseline.

## References

- Tier B sprint plan: [`docs/planning/sprint_plan_tier_b.md`](../planning/sprint_plan_tier_b.md) (Track B3)
- Phase A stub: [`src/mousedroid/sim/isaaclab/rover_env.py`](../../src/mousedroid/sim/isaaclab/rover_env.py)
  (3 `TODO(Phase B)` markers at lines 108, 146, 180)
- Mock backend (reference contract):
  [`src/mousedroid/sim/mock_rover_env.py`](../../src/mousedroid/sim/mock_rover_env.py)
- Constants module (joint / link names):
  [`src/mousedroid/sim/isaaclab/constants.py`](../../src/mousedroid/sim/isaaclab/constants.py)
- Conversion script:
  [`scripts/convert_urdf_to_usd.py`](../../scripts/convert_urdf_to_usd.py)
- Sister ADR (Track B2): [`ADR-008-world-model-onnx-engine.md`](./ADR-008-world-model-onnx-engine.md)
- Isaac Lab docs: https://isaac-sim.github.io/IsaacLab/
