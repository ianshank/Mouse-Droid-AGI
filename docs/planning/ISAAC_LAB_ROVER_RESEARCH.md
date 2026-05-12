# Research: Training the MSE-6 Rover with NVIDIA Isaac Lab + Omniverse

> **Date**: 2026-05-09
> **Branch**: `claude/isaac-lab-rover-research-HXsTi`
> **Status**: Research / Proposal — Awaiting Review
> **Scope**: Mouse-droid (rover) platform only. Robot-arm platform (`src/mousedroid/arm/`) is out of scope.

---

## TL;DR

Today the rover is **inference-only on real hardware**. Every learned weight (RSSM,
BDI, Constitutional RL, distilled VLA) is bootstrapped from `SyntheticSequenceGenerator`
running a *mock* ESP32 + camera + LiDAR — no physics, no contacts, no photometric
reality, no parallelism. This is a hard ceiling on policy quality and a known
sim-to-real risk.

NVIDIA Isaac Lab is the only mature option that simultaneously gives us:

1. **GPU-parallel rigid-body physics** for the Wave Rover mecanum chassis (thousands of envs at once),
2. **First-class sensor models** that match every modality we already use — RGB,
   depth, RTX 2D LiDAR, ultrasonic, IMU, contact — with built-in noise / domain
   randomization,
3. A **policy export path (ONNX → TensorRT FP16)** that drops directly into our
   existing Jetson Orin Nano inference stack,
4. **Cosmos Transfer / MobilityGen** to close the photometric sim-to-real gap
   without us hand-authoring photoreal scenes,
5. A **published precedent** (DreamerNav, 2025) that runs DreamerV3 + RSSM in
   Isaac Sim for indoor navigation — i.e. the exact world-model architecture we
   already ship.

The recommendation is to keep the existing protocol-based architecture intact,
add Isaac Lab as a new **simulation backend** behind a new
`SimulationProtocol`, and stage adoption in four phases over ~6 weeks of effort.
No invariant in `CLAUDE.md` is violated.

---

## 1. Why Now

### 1.1 What we have today (verified in this branch)

| Component | Where | What it actually does |
|---|---|---|
| Mock data generator | `src/mousedroid/training/data_generator.py` | Drives a *Python mock* of the rover, samples actions, applies tabular DR |
| Domain randomization | `src/mousedroid/training/domain_randomization.py` | Vision/ultrasonic noise, motor latency, push perturbations — **all in numpy, no physics** |
| RSSM training | `src/mousedroid/world_model/rssm.py`, `training/train_rssm.py` | 256-d hidden, 64-d latent, 256-d obs, 3-d action |
| MCTS planner | `src/mousedroid/world_model/mcts.py` | 50–200 sims, depth 5, γ=0.97, UCB c=1.41 over 9 candidate actions |
| Drive interface | `src/mousedroid/comms/protocol.py` + `serial_driver.py` | 3-DoF mecanum cmd `(vx, vy, ω)` over 1 Mbps UART, 50 ms E-stop |
| Sensors | `src/mousedroid/sensing/manager.py` | Vision (256-d IMX500), 2D LiDAR (`LidarConfig.n_sectors`-d, default 36), ultrasonic (1-d), motor (4-d), audio (192-d) |
| Robot geometry | `RobotConfig` in `config/schema.py` | mecanum, wheelbase 0.20 m, track 0.20 m, wheel r=0.042 m |
| VLA inference | `src/mousedroid/vla/policy.py` | ONNX Runtime + TensorRT, weights pulled from HF `ianshank/mousedroid-weights` |
| Jetson runtime | `JetsonConfig` in `config/schema.py`, `Dockerfile.jetson` | CUDA 12.6, TensorRT, FP16, 0.5 GPU mem fraction |

### 1.2 The actual gap

There is **no rover physics engine in the repo**. MuJoCo is wired up only for the
arm (`src/mousedroid/arm/environments/`). The rover's training data is a tabular
reconstruction of what we *think* the robot will see. That is enough to validate
the architecture; it is not enough to ship a navigation policy.

Concretely, today we can't:

- Train against contact dynamics (curbs, carpet, door thresholds).
- Generate photorealistic egocentric video that the IMX500 feature extractor will see in the field.
- Run thousands of parallel environments on a single GPU — RSSM training is single-stream and CPU-bound on data gen.
- Validate the 50 ms E-stop budget under realistic latency / dropout statistics.
- Curriculum-train MCTS against a closed-loop physics world.

### 1.3 What Isaac Lab gives us

Isaac Lab is NVIDIA's **GPU-parallel robot-learning framework on top of Isaac Sim**
(BSD-3, Python 3.11, Linux x86_64). It ships:

- 30+ ready-to-train environments and 16+ robot models, with first-class support
  for **wheeled mobile robots / AMRs** via `isaacsim.robot.wheeled_robots`
  (HolonomicController for mecanum, DifferentialController for diff-drive).
- Native sensors: RGB / depth / segmentation cameras, **RTX cameras and RTX
  LiDAR**, IMU, contact, ray-casters, ultrasonic — with built-in noise models
  (additive, miscalibration, rolling shutter, multipath).
- Replicator-driven **physics randomization** (mass, inertia, joint friction,
  contact offset, restitution) and **visual randomization** (PBR materials,
  lighting, HDRI domes).
- Out-of-the-box adapters for **RSL-RL, SKRL, RL-Games, Stable-Baselines3**;
  PPO/SAC are one-liners.
- `play.py` automatically calls `export_policy_as_jit()` and
  `export_policy_as_onnx()`, writing artifacts to
  `{checkpoint_dir}/exported/policy.onnx` — no Isaac Lab needed at deploy time.
- **Cosmos Transfer** + **MobilityGen** can convert a single sim trajectory into
  thousands of photorealistic rollouts (varied lighting, materials, clutter)
  for closing the sim-to-real gap.
- Documented Sim-to-Real Policy Transfer page (with the recent Newton physics
  back-end) plus a NVIDIA developer-forum thread on Jetson Orin deployment.

### 1.4 Independent precedent: DreamerNav

DreamerNav (Frontiers in Robotics & AI, Oct 2025) extends **DreamerV3 with an
RSSM** — same architecture family we use — for indoor navigation, trained
**inside Isaac Sim** with curriculum learning. Their RSSM encodes egocentric
depth + a structured local occupancy map; they demonstrate cross-platform
transfer between two quadrupeds. This is direct external validation that
RSSM+Isaac Sim is a working combination, not a research bet.

---

## 2. Proposed Architecture

We add Isaac Lab **behind a new protocol**, leaving every existing invariant
intact.

```
┌────────────────────────────────────────────────────────────────────┐
│                       Existing (unchanged)                         │
│   factory.py → orchestrator → world_model (RSSM + MCTS)            │
│              → vla (ONNX + TensorRT) → comms (ESP32)               │
└────────────────────────────────────────────────────────────────────┘
                              ▲
                              │ trained weights
                              │ (.pt → .onnx → .engine)
                              │
┌─────────────────────────────┴──────────────────────────────────────┐
│                         NEW: training-time only                    │
│                                                                    │
│   src/mousedroid/sim/                                              │
│     protocol.py            SimulationProtocol  (runtime_checkable) │
│     isaac_lab/                                                     │
│       env.py               IsaacLabRoverEnv  (Gymnasium-compatible)│
│       robot_usd.py         MSE-6 USD/URDF builder from RobotConfig │
│       sensors.py           RTX-LiDAR, depth-cam, ultrasonic wrappers│
│       randomization.py     reads existing DomainRandomizationConfig│
│     mock/                  (existing data_generator.py wrapped)    │
│                                                                    │
│   src/mousedroid/training/isaac/                                   │
│     train_ppo.py           SKRL / RSL-RL PPO baseline              │
│     train_rssm_isaac.py    RSSM trained on Isaac rollouts          │
│     export_jetson.py       ONNX → TensorRT FP16 (engine cache)     │
└────────────────────────────────────────────────────────────────────┘
```

Key properties:

- **No new hardcoded values** — `IsaacLabRoverEnv` consumes `Settings` exactly
  like every other module. Wheelbase, sensor frequencies, action limits, DR
  ranges all come from the existing schema.
- **Protocol-based DI** — `SimulationProtocol` is `@runtime_checkable`. Concrete
  Isaac classes are imported only inside `factory.build_simulation()`.
- **Backwards compatible** — Isaac Lab is an *optional* dev/training extra. The
  Jetson runtime container does not need it; only the workstation does.
- **`pyproject.toml`** adds an `[isaac]` extra (heavy: `isaaclab`, `isaacsim`,
  `rsl-rl-lib`, `skrl`). CI continues to pass without it.

---

## 3. Component-by-Component Mapping

### 3.1 Robot model (USD/URDF)

| Today (`RobotConfig`) | Isaac Lab artifact |
|---|---|
| `wheel_base=0.20`, `track_width=0.20`, `wheel_type=…`, `wheel_radius=0.042` | `mse6_rover.urdf` generated by `sim/isaac_lab/robot_usd.py` from config |
| 3-DoF action `(vx, vy, ω)` with `max_velocity_mps=0.5`, `max_omega_rads=2.0` | Controller selected from `RobotConfig.wheel_type` (`HolonomicController` for mecanum, `DifferentialController` for `standard` diff-drive); action-space limits clamp to `ESP32Config` |
| 50 ms E-stop budget | Latency injected as Replicator action-delay randomization |

> **Risk noted:** the Isaac Lab community has reported sim-to-real issues with
> n-gon wheels for mecanum and tank drives. The `sim/isaac_lab/robot_usd.py`
> builder must dispatch on `RobotConfig.wheel_type` so a differential-drive
> chassis is never silently routed through the holonomic path. We must also
> validate the chosen controller against the Wave Rover's real velocity-tracking
> response curve before committing to it. Mitigation: log real wheel encoder
> traces in Phase 0 and fit motor dynamics into the URDF actuator model.

### 3.2 Sensors

| Today (`SensorManager`) | Isaac Lab equivalent | Notes |
|---|---|---|
| IMX500 vision → 256-d feature | `Camera` with RGB at 30 Hz → run IMX500 ONNX *inside* the env or offline-replay | Use Cosmos Transfer for photorealism (§4) |
| FHL-LD19 2D LiDAR → `LidarConfig.n_sectors` angular bins (default 36) | `RTX Lidar` configured as 2D scan; env wrapper bins to `LidarConfig.n_sectors` angles | Match scan FoV/range from `LidarConfig` |
| HC-SR04 ultrasonic | `RangeSensor` (built-in ultrasonic primitive) | Single-ray ray-cast at GPIO-driven sample rate |
| ESP32 motor telemetry | Read joint velocities + battery model | Battery as exponential discharge env state |
| USB mic (192-d mel) | Not simulated. Use real-recorded clips replayed deterministically | Audio is a perception input only — leaving it on the mock path is acceptable |

All five outputs are projected per modality (vision → `vision_proj_dim`,
ultrasonic/motor/audio/lidar → their respective `*_proj_dim`), concatenated,
and fused by a final `Linear` into the same `obs_dim=256` vector that
`world_model/encoder.MultimodalEncoder` already produces — no encoder changes
needed, only feeding the same shapes from the simulator.

### 3.3 World model & planner

`RSSM(hidden=256, latent=64, obs=256, action=3)` is **kept as-is**. We change
*only the data source* feeding it:

- Today: `SyntheticSequenceGenerator.run_mock()` → `sequences.pt`.
- Isaac: `train_rssm_isaac.py` collects `(obs, action, reward)` tuples from N
  parallel `IsaacLabRoverEnv` instances and writes the same `sequences.pt`
  schema.

`mcts.py` is unchanged — it operates over learned latents, agnostic to the data
source.

### 3.4 Reward, safety, curiosity

The existing modules are physics-engine-agnostic and reused unchanged:

- `reward/` — multi-objective constitutional reward. New Isaac-specific reward
  terms (collision penalty from contact sensors, progress-to-goal from base
  link pose) are added as additional `RewardComponent` entries, not replacements.
- `safety/` — runtime safety monitor in sim consumes the same joint-limit /
  E-stop config used on real hardware.
- `curiosity/` — ICM intrinsic reward stays in latent space.

### 3.5 Domain randomization

The current `DomainRandomizationConfig` already defines ranges for vision
brightness/contrast, gaussian noise, motor latency, chassis dynamics. We **keep the same Pydantic schema** and add a
`sim/isaac_lab/randomization.py` adapter that translates each field to the
appropriate Replicator call:

| Schema field | Replicator API |
|---|---|
| `vision.brightness_range`, `contrast_range`, `gaussian_noise` | Per-camera annotator + image-augmentation graph |
| `motor.latency_ms_range`, `slip_range` | Articulation joint friction + action-delay buffer |
| `chassis.mass_kg_range`, `friction_range` | `randomize_rigid_body_masses`, material `physics_randomization` |
| External pushes | Periodic external force on root link |

This is the minimum change that lets us reuse the existing config + tests.

---

## 4. Sim-to-Real Pipeline

Three lines of defense, ordered from cheapest to most powerful:

1. **Physics DR** (free, in-loop): mass/inertia/friction/restitution/contact
   offset randomization on the chassis + ground plane, plus action-delay and
   sensor-dropout randomization. This alone is what gets quadrupeds to walk
   off a treadmill.
2. **Visual DR** (cheap, in-loop): randomize PBR materials, HDRI lighting,
   floor/wall textures, and simulated camera noise (rolling shutter,
   chromatic, Bayer mosaic). Done with `isaacsim.replicator.domain_randomization`.
3. **Cosmos Transfer** (offline, GPU-heavy, optional): once the policy works
   in DR, run the captured rollouts through Cosmos Transfer to generate
   photorealistic indoor scenes (varied cabinetry, lighting, clutter). Use
   this dataset to **fine-tune the IMX500 vision feature extractor**, not the
   policy — that closes the photometric gap without invalidating the
   trained dynamics.

Optionally we can use **MobilityGen** (Isaac Sim workflow) to bootstrap the
world model with ground-truth occupancy maps + RGB/depth/segmentation, which
is a clean Phase-0 pre-training signal for the RSSM encoder.

---

## 5. Training → Deployment Pipeline (Jetson Orin Nano)

```
┌──────────────────────────────────────────────────────────────────┐
│ Workstation (x86_64 + dGPU, Linux)                               │
│                                                                  │
│  isaac_lab/IsaacLabRoverEnv ──► PPO (rsl-rl) or warm-start RSSM  │
│         │                                                        │
│         └─► policy.pt ──► policy.onnx  (export_policy_as_onnx)   │
└──────────────────────────────────────────────────────────────────┘
                              │ scp / HF Hub push
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Jetson Orin Nano (already in repo)                               │
│                                                                  │
│  policy.onnx ──► trtexec --fp16 ──► policy.engine                │
│                  (cache: /opt/mousedroid/tensorrt_cache)         │
│                                                                  │
│  vla/policy.py ──► loads .engine via existing TensorRT path      │
│  orchestrator   ──► 30 Hz tick consumes policy as today          │
└──────────────────────────────────────────────────────────────────┘
```

Everything to the right of the `policy.onnx` boundary **already exists** in
the codebase (PR #58 for distilled VLA ONNX, PR #63 for the Jetson activation
playbook, `JetsonConfig.tensorrt_*` fields). The new work is left of that
boundary.

Weight distribution stays on `ianshank/mousedroid-weights` HuggingFace repo —
we publish Isaac-trained weights under a new `isaac_lab/` subdirectory next to
the existing BDI / VLA artifacts.

---

## 6. Phased Roadmap

| Phase | Duration | Deliverable | Acceptance |
|---|---|---|---|
| **0 — Spike** | 3–5 days | Run Isaac Lab `Cartpole-v0` + a stock wheeled-robot demo on a workstation; confirm CUDA drivers, USD viewer, RSL-RL PPO export-to-ONNX round-trip | A `policy.onnx` runs on a Jetson Orin Nano dev kit and produces sane actions |
| **1 — MSE-6 USD asset** | 1 week | URDF generator (`sim/isaac_lab/robot_usd.py`) reads `RobotConfig`; mecanum chassis loads in Isaac with correct mass/wheels; teleop in the GUI matches real Wave Rover | Side-by-side video: Isaac chassis tracks the same `(vx, vy, ω)` profile as a real bench-test trace within ±10% |
| **2 — Sensor parity** | 1 week | RTX LiDAR → `LidarConfig.n_sectors`-bin scan, depth/RGB camera → `CameraConfig.feature_dim`-d feature, ultrasonic ray-cast — all flowing into the same `MouseDroidObservationBundle` | Property tests in `tests/property/` show obs distributions overlap with mock-data baselines within KL < 0.5 |
| **3 — RL baseline** | 1 week | PPO via SKRL/RSL-RL on a goal-reaching task in a 5×5 m room; export to ONNX; deploy to Jetson | ≥80% goal-reach success in sim with full DR; ≥50% in a controlled real-world room |
| **4 — RSSM/MCTS integration** | 1–2 weeks | Replace `SyntheticSequenceGenerator` with `IsaacLabRoverEnv` rollouts; retrain RSSM; warm-start MCTS policy from PPO | RSSM reconstruction loss ≤ current mock baseline; orchestrator passes existing `tests/integration/` |
| **5 — Cosmos / MobilityGen** | optional, 2 weeks | Photoreal sim-to-real pipeline | Improved real-world success vs Phase-3 baseline |

Phases 0–4 = **~5–6 person-weeks**. Phase 5 is opt-in and depends on Cosmos
licensing / GPU availability.

---

## 7. Risk Register

| # | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| R1 | Mecanum wheels behave poorly under Isaac's n-gon wheel model — known community pain point | Medium | High | Phase 1 validation gate; fall back to differential-drive approximation for the policy if needed (the chassis has the controller, not the policy) |
| R2 | Isaac Lab requires Python 3.11 + Linux x86_64 + dGPU; the project currently pins `requires-python = ">=3.10"` in `pyproject.toml` and our CI matrix runs 3.10/3.11/3.12 | Medium | Medium | The `[isaac]` extra documents Python 3.11 as a hard floor (no version bump to the project itself — the runtime, Jetson container, and CI matrix stay on 3.10+); stand up a single shared training rig that pins 3.11; dev workflow stays on existing mock path |
| R3 | Sim-to-real gap remains too large after Phase 3 | Medium | High | Phase 5 (Cosmos Transfer) explicitly budgets for this; keep the existing `domain_randomization.py` numpy path as a fallback training source |
| R4 | RSSM trained on Isaac data fails to encode novel modalities (e.g. real audio) | Low | Medium | Audio stays on mock/recorded path — already accepted in §3.2 |
| R5 | TensorRT FP16 export drift between workstation export and Jetson runtime | Low | Medium | Existing `tensorrt_cache` + `Dockerfile.jetson` already exercises this path for VLA; reuse |
| R6 | Isaac Lab major version churn (Sim 4.5 → 5.0 → 5.1 in <12 months) | High | Low | Pin `isaacsim`/`isaaclab` versions in `[isaac]` extra; bump deliberately, not opportunistically |
| R7 | Licensing — Isaac Sim is **proprietary**; Isaac Lab is BSD-3 | Low | Low | We never redistribute Isaac Sim binaries; we depend on it as a dev tool, identical to MuJoCo today |

---

## 8. Open Questions

1. **Do we keep MuJoCo for the arm and add Isaac for the rover, or migrate the
   arm too?** Recommend keeping MuJoCo for the arm (already working, lower
   overhead) and using Isaac only for the rover.
2. **PPO vs Dreamer-V3 as the primary trainer?** PPO is the documented Isaac
   Lab path and the easiest export-to-ONNX. Dreamer-V3 + RSSM is closer to our
   architecture but currently a research bet (DreamerNav is the only public
   Isaac+Dreamer integration, and it's a fork). Recommend PPO for Phase 3,
   Dreamer-V3 as a Phase 6 research extension.
3. **Where do Isaac-trained weights live?** Recommend new HF subdirectory
   `ianshank/mousedroid-weights/isaac_lab/<phase>/` — identical pattern to the
   current VLA/BDI weights.
4. **Do we need a CI gate?** No, not initially. Isaac Lab is too heavy for the
   GitHub-hosted runner. Add a self-hosted runner only if Phase 4 lands.

---

## 9. Recommendation

**Proceed with Phase 0 (Isaac Lab spike).** It's a 3–5 day investment, the
deliverable is a deployable ONNX policy on the existing Jetson hardware, and
no production code changes until Phase 2. Every architectural invariant in
`CLAUDE.md` is preserved by the protocol-based wrapping in §2.

The strongest single argument is composability: every component above the
simulation layer (RSSM, MCTS, VLA, ESP32 driver, safety monitor, telemetry) is
already protocol-based and platform-agnostic. Swapping the data source from
`SyntheticSequenceGenerator` to `IsaacLabRoverEnv` is the cleanest large-impact
change available to us right now.

---

## References

### NVIDIA primary sources
- [NVIDIA Isaac Lab — Open-Source Modular Framework](https://developer.nvidia.com/isaac/lab)
- [Isaac Lab GitHub (BSD-3, Python 3.11)](https://github.com/isaac-sim/IsaacLab)
- [Isaac Lab — Reference Architecture](https://isaac-sim.github.io/IsaacLab/main/source/refs/reference_architecture/index.html)
- [Isaac Sim — Wheeled Robots extension](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.robot.wheeled_robots/docs/index.html)
- [Isaac Sim — Mobile Robot Controllers (Holonomic, Differential)](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/mobile_robot_controllers.html)
- [Isaac Sim — Tutorial: Import URDF](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/import_urdf.html)
- [Isaac Sim — RTX Lidar Sensor](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/sensors/isaacsim_sensors_rtx_lidar.html)
- [Isaac Sim — Ultrasonic Sensors tutorial](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/advanced_tutorials/tutorial_advanced_range_sensor_ultrasonic.html)
- [Isaac Sim Replicator — Domain Randomization API](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.replicator.domain_randomization/docs/index.html)
- [Isaac Lab — `isaaclab.sensors` API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html)
- [Isaac Lab — `isaaclab_rl` API (RSL-RL, SKRL, RL-Games, SB3)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html)
- [Isaac Lab — `rsl_rl.exporter` (`export_policy_as_jit`, `export_policy_as_onnx`)](https://docs.robotsfan.com/isaaclab_official/main/_modules/isaaclab_rl/rsl_rl/exporter.html)
- [Isaac Lab — Policy Inference in USD Environment](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/policy_inference_in_usd.html)
- [Isaac Sim — Deploying Policies tutorial](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/isaac_lab_tutorials/tutorial_policy_deployment.html)
- [Isaac Lab — Sim-to-Real Policy Transfer (Newton physics)](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/sim-to-real.html)
- [Cosmos Synthetic Data Generation in Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_cosmos.html)
- [Cosmos Cookbook — Transfer for Robotics Navigation Tasks](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer1/inference-x-mobility/inference.html)
- [NVIDIA Developer Forum — Deployment of Isaac Lab trained RL agent to Jetson Orin](https://forums.developer.nvidia.com/t/deployment-of-isaac-lab-trained-rl-agent-to-jetson-nano-orin/318835)
- [NVIDIA Blog — Advanced Sensor Physics, Customization, and Model Benchmarking in Isaac Sim & Isaac Lab](https://developer.nvidia.com/blog/advanced-sensor-physics-customization-and-model-benchmarking-coming-to-nvidia-isaac-sim-and-nvidia-isaac-lab/)
- [NVIDIA Blog — Building Generalist Humanoid Capabilities with Isaac GR00T N1.6 (Sim-to-Real Workflow)](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow/)

### Academic / research
- [Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning (arXiv 2511.04831)](https://arxiv.org/html/2511.04831v1)
- [DreamerNav: learning-based autonomous navigation in dynamic indoor environments using world models (Frontiers in Robotics & AI, 2025)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1655171/full)
- [DreamerV3 — Mastering Diverse Domains through World Models (arXiv 2301.04104)](https://arxiv.org/pdf/2301.04104)
- [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots (arXiv 2503.14734)](https://arxiv.org/abs/2503.14734)

### Internal references
- `CLAUDE.md` — project invariants
- `src/mousedroid/world_model/rssm.py`, `mcts.py`, `protocol.py`
- `src/mousedroid/sensing/manager.py`, `bundle.py`
- `src/mousedroid/comms/protocol.py`, `serial_driver.py`
- `src/mousedroid/training/data_generator.py`, `domain_randomization.py`
- `src/mousedroid/config/schema.py` (`RobotConfig`, `JetsonConfig`, `DomainRandomizationConfig`, `LidarConfig`, `CameraConfig`, `ESP32Config`)
- `Dockerfile.jetson`, `pyproject.toml`
- PR #58 (distilled VLA ONNX), PR #60 (replay), PR #63 (Jetson activation)
