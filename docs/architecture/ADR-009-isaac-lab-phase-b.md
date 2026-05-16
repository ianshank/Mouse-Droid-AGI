# ADR-009 — Isaac Lab Phase B (Real-Env Wiring for the Rover)

**Status:** Accepted (foundation + Tier C4 body wired; operator-on-Linux
validation remains the Phase 5 gate)
**Date:** 2026-05-16 (Tier C4 amendment)
**Sprint:** Tier B Track B3 (foundation), Tier C Track C4 (body wiring)

> **Note on numbering:** original sprint plan called this ADR-008. That
> number was claimed by the World-Model ONNX Engine ADR being introduced
> on the parallel Tier B2 branch (`claude/tier-b2-onnx-world-model`,
> `docs/architecture/ADR-008-world-model-onnx-engine.md` — not on this
> B3 branch). Isaac Lab Phase B is filed as ADR-009 so the two ADRs can
> land independently without a numbering collision.

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

The **policy action vector** is 2-D for both supported modes
(`RoverActionConfig.mode = "differential"` or `"body_velocity"`) per
`_ROVER_ACTION_DIM_BY_MODE` in the schema. The 2-D policy output is
fanned out onto **4 wheel actuators** by Phase B's `step()` body,
mirroring `MockRoverEnv`'s existing fan-out logic.

The actuator fan-out order — i.e. the order in which the env wires
articulation joints to the wheel commands — is pinned by
`ROVER_WHEEL_JOINT_NAMES` constant (B3 Story 0,
[`src/mousedroid/sim/isaaclab/constants.py`](../../src/mousedroid/sim/isaaclab/constants.py)):
`(joint_wheel_fl, joint_wheel_fr, joint_wheel_rl, joint_wheel_rr)`.
**This tuple is the actuator fan-out order, not the incoming action
shape.** The 2-D differential-drive action `[left, right]` maps onto
the 4 wheels by duplicating `left` into `(FL, RL)` and `right` into
`(FR, RR)`.

Action clipping: `RoverActionConfig.max_wheel_rad_s` (default 25 rad/s)
is consumed by the **mock backend** today. Phase B's `step()` body
will use it as the wheel-velocity clip before forwarding to the
articulation actuators, giving both backends a single source of truth.
The URDF does **not** encode this limit yet (Isaac Lab's
articulation joint limits are populated from the URDF itself), nor is
a separate safety-layer enforcement currently wired — those are
follow-ups for Phase 5's safety-monitor expansion.

### Observation space

Under the **default** `RoverObservationConfig` (all four `include_*`
toggles on), both backends emit
`("imu", "chassis_pose", "wheel_vel", "lidar")` — matching
`ROVER_OBSERVATION_KEYS` exactly. The
[cross-backend contract test](../../tests/unit/sim/isaaclab/test_constants.py)
asserts this exact equality for the default toggles.

Non-default toggle combinations (operator disables LiDAR, etc.) are
NOT pinned by `ROVER_OBSERVATION_KEYS` — the authoritative source of
truth at runtime is `RoverObservationConfig.enabled_keys()`, which both
backends read at `__init__`. Phase B's `RoverIsaacLabEnv` must honour
the same toggles so the configured observation set stays consistent
across backends; the constant is documentation + a default-config
contract test, not a runtime gate.

**Explicitly excluded from the observation space:**

- HC-SR04 ultrasonic — parked sensor modality, forbidden from the rover
  baseline (`CLAUDE.md` invariant).
- Arm joint encoders — parked robot-arm platform.

### Reward function (Phase B baseline — implemented in Tier C4)

Simple forward-progress baseline:

```python
reward = (
    cfg.rover.reward.forward_velocity_weight * forward_velocity_mps
    - cfg.rover.reward.collision_weight * is_colliding
)
```

> **Status: IMPLEMENTED in Tier C4.** The
> [`RoverRewardConfig`](../../src/mousedroid/config/schema.py)
> Pydantic block ships alongside the `build/reset/step` body. The
> field path is
> [`cfg.rover.reward.forward_velocity_weight`](../../src/mousedroid/config/schema.py)
> and
> [`cfg.rover.reward.collision_weight`](../../src/mousedroid/config/schema.py)
> as documented below.

Implemented defaults (Tier C4):
- `forward_velocity_weight: float = 0.01` (rewards ~1 m/s motion at +0.01 r/step)
- `collision_weight: float = 0.1` (firm penalty for collision frames)

The reward block is **optional on `RoverConfig`** with a default of
`None` so existing YAML files load byte-identically. The Isaac Lab
env raises a clear `ValueError` at `build()` time when
`cfg.rover.reward is None`, so operators set the block explicitly
when they opt into the Isaac Lab backend.

Rationale: Phase B's reward must be **cheap** (no learned VLM signals)
so curriculum training can iterate fast. Richer reward shaping
(instruction following, multi-objective) lands in Phase 5 via the
existing `mousedroid.training.rover_reward` module.

Both weights are operator-tunable Pydantic fields — no hardcoded
reward weights live inside `src/mousedroid/sim/isaaclab/`.

### Domain randomization

Phase B reuses the existing top-level `cfg.domain_randomization`
Phase 1 baseline (Pydantic block on `Settings`, not nested under
`training`). The env's `reset()` calls into
`mousedroid.training.domain_randomization` helpers — no duplication.

> **Amendment (Tier C4):** earlier drafts of this ADR documented the
> path as `cfg.sim.domain_randomization.enabled`. That is incorrect —
> the runtime path threaded into `RoverIsaacLabEnv` by the factory is
> `cfg.domain_randomization.enabled` (top-level on `Settings`). The
> Isaac Lab env constructor takes the
> `DomainRandomizationConfig` as a keyword-only argument; the factory
> passes `cfg.domain_randomization` directly.

Current defaults (per `Settings.domain_randomization` + `config/default.yaml`):
`enabled: true`. Operators on hosts without Isaac Sim should set this
to `false` for byte-identical pre-B3 mock-only runs; the env wiring
respects the toggle so existing CI on mock hardware is unaffected.

## What's in this PR (Foundation + Tier C4 body)

Limited to what's verifiable on a development workstation without
Isaac Sim. Operator-side validation on Ubuntu 22.04 + Isaac Sim 4.5+
closes out the remaining sub-tasks.

| Component | Status | File |
|---|---|---|
| `ROVER_WHEEL_JOINT_NAMES` + cross-backend constants | ✅ Story 0 | `src/mousedroid/sim/isaaclab/constants.py` |
| URDF -> USD conversion script | ✅ Story 1 | `scripts/convert_urdf_to_usd.py` |
| Conversion script smoke test (CI-skippable) | ✅ Story 1 | `tests/unit/sim/isaaclab/test_urdf_to_usd.py` |
| ADR-009 documenting decisions | ✅ Story 6 | _this file_ |
| `RoverIsaacLabEnv.build()` — scene + articulation + sensors | ✅ Tier C4 | `src/mousedroid/sim/isaaclab/rover_env.py` |
| `RoverIsaacLabEnv.reset()` — domain randomization integration | ✅ Tier C4 | `src/mousedroid/sim/isaaclab/rover_env.py` |
| `RoverIsaacLabEnv.step()` — action fan-out + reward + obs | ✅ Tier C4 | `src/mousedroid/sim/isaaclab/rover_env.py` |
| `RoverRewardConfig` Pydantic block (defaults documented above) | ✅ Tier C4 | `src/mousedroid/config/schema.py` |
| 9 unit tests under `pytest.importorskip("isaaclab")` | ✅ Tier C4 | `tests/unit/sim/isaaclab/test_rover_env.py` |
| 50-step random-rollout liveness smoke check | ✅ Tier C4 (test 9/9) | `tests/unit/sim/isaaclab/test_rover_env.py::test_random_rollout_produces_finite_observations` |

## What's deferred to a follow-up (operator-on-Linux validation)

Each remaining sub-task requires a live Isaac Sim installation for
end-to-end validation; the Tier C4 PR lands the code body but
operator-on-Linux validation remains the gate for Phase 5.

| Sub-task | Owner |
|---|---|
| Commit `assets/rover/mse6_4wd.usd` (run conversion script once) | Operator |
| Run all 9 unit tests under live `isaaclab` on Linux + Isaac Sim 4.5+ | Operator |
| Confirm `test_random_rollout_produces_finite_observations` PASS (no NaN/Inf in obs over 50 steps) | Operator |

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

## Operator validation playbook (Tier C4 post-merge)

Once the C4 PR lands, the operator runs the 9 unit tests on a Linux
workstation with Isaac Sim installed. The tests are guarded by
`pytest.importorskip("isaaclab")` so they SKIP cleanly on CI hosts
(Windows / Jetson) where Isaac Lab is not installed.

```bash
# On Linux workstation with isaaclab installed:
pip install -e ".[isaac]"
# Confirm the .usd asset is present (operator commits it once per URDF
# revision via scripts/convert_urdf_to_usd.py).
ls assets/rover/mse6_4wd.usd

# Run all 9 C4 tests. Expected: 9 PASSED.
python -m pytest tests/unit/sim/isaaclab/test_rover_env.py -v \
    --import-mode=importlib --no-cov

# Run the 50-step random-rollout smoke check in isolation.
python -m pytest \
    tests/unit/sim/isaaclab/test_rover_env.py::TestRoverIsaacLabEnv::test_random_rollout_produces_finite_observations \
    -v --import-mode=importlib --no-cov
# Expected: PASSED. Surfaces NaN/Inf regressions in the build/reset/step body.

# Cross-backend contract: same 2-D action -> same wheel velocities under
# both MockRoverEnv and RoverIsaacLabEnv. Pins the [FL=L, FR=R, RL=L, RR=R]
# fan-out layout.
python -m pytest \
    tests/unit/sim/isaaclab/test_rover_env.py::TestRoverIsaacLabEnv::test_observation_contract_matches_mock_rover_env \
    -v --import-mode=importlib --no-cov
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

- Tier B sprint plan: tracked in the planning Linear project; high-level
  shape lives in [`docs/planning/IMPLEMENTATION_PLAN.md`](../planning/IMPLEMENTATION_PLAN.md)
  and the related rover research note at
  [`docs/planning/ISAAC_LAB_ROVER_RESEARCH.md`](../planning/ISAAC_LAB_ROVER_RESEARCH.md)
  (note: a consolidated `sprint_plan_tier_b.md` was referenced in the
  original draft of this ADR but the artifact lives outside this repo —
  the linked planning docs hold the current source of truth).
- Phase A stub: [`src/mousedroid/sim/isaaclab/rover_env.py`](../../src/mousedroid/sim/isaaclab/rover_env.py)
  (3 `TODO(Phase B)` markers at lines 108, 146, 180)
- Mock backend (reference contract):
  [`src/mousedroid/sim/mock_rover_env.py`](../../src/mousedroid/sim/mock_rover_env.py)
- Constants module (joint / link names):
  [`src/mousedroid/sim/isaaclab/constants.py`](../../src/mousedroid/sim/isaaclab/constants.py)
- Conversion script:
  [`scripts/convert_urdf_to_usd.py`](../../scripts/convert_urdf_to_usd.py)
- Sister ADR (Track B2): `ADR-008-world-model-onnx-engine.md` — lives on
  the parallel B2 branch (`claude/tier-b2-onnx-world-model`), will be
  cross-linkable once both branches merge into `main`.
- Isaac Lab docs: https://isaac-sim.github.io/IsaacLab/
