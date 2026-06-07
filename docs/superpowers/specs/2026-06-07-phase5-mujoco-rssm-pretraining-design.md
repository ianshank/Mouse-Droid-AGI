# Phase 5 — MuJoCo Skid-Steer Rover Env → RSSM Dynamics Pretraining (Design Spec)

**Date:** 2026-06-07
**Status:** Approved design (3-agent adversarial peer review incorporated)
**Branch:** `claude/phase5-mujoco-rssm-pretraining`
**Roadmap origin:** `NEXT_STEPS.md` → "Phase 5 (stretch) — Real Physics Simulator"

---

## 1. Goal

Replace the NumPy kinematic rover sim (`MockRoverEnv`) with a **MuJoCo (classic
C engine) skid-steer physics simulator**, and use the episodes it generates to
**pretrain the RSSM world model's dynamics core** end-to-end through the training
pipeline orchestrator.

This closes the last big modelling gap on the Physical-AI roadmap: the mock env
teaches the world model only frictionless kinematics; the real rover exhibits
slip, contact, and chassis dynamics the mock cannot produce.

## 2. Scope decisions (locked)

| Decision | Value | Rationale |
|---|---|---|
| Physics engine | **MuJoCo classic** (`mujoco>=3.0`, already in `[arm]` extra) | Lightweight, CPU-or-CUDA, runs on RTX 5060 Ti workstation; fills the reserved `"mujoco"` factory slot. NOT MJX (heavy JAX), NOT Isaac (heavy Omniverse). |
| Drivetrain | **4-wheel skid-steer** (differential `[L,R]`, no strafing) | Matches the real Wave Rover (`{L,R}` firmware). The RSSM learns dynamics the rover can actually execute. |
| Slice scope | **End-to-end**: env → episode generation → RSSM dynamics pretraining loop | User-selected. Larger but coherent. |
| Data path | **Approach A** — in-process streaming generator → trainer; no disk | YAGNI. LMDB/mixer persistence is a deferred follow-on. |
| Execution | **Offline** data-gen on the workstation | On-device/in-the-loop sim is a Phase 6 concern. |

## 3. Architecture & data flow

```
build_rover_env(cfg)  ──►  RoverMuJoCoEnv (RoverEnvProtocol; fills reserved "mujoco" slot)
        │                        │  obs dict: {imu, chassis_pose, wheel_vel, lidar}  (== MockRoverEnv contract)
        ▼                        ▼
   SimEpisodeGenerator  ──►  EpisodeBatch (NEW in-memory dataclass; NOT MouseDroidExperienceRecord)
        │  (seeded smoothed-uniform-random wheel cmds — Dreamer seed policy)
        ▼
   RoverObsAdapter  ──►  RSSM encoder tensors  (motor_state=[vx,vy=0,omega,batt]; vision OMITTED)
        │
        ▼
   RSSMPretrainer  ──►  RSSM.train_sequence(...)   [NEW grad-enabled forward: raw-modality recon + free-bits KL]
        │                  Adam + AMP(KL in fp32) + grad-clip; checkpoint to weights_dir
        ▼
   PipelineOrchestrator._train_rssm  delegates here (wrapped in asyncio.to_thread)
```

## 4. Peer-review findings this design closes

Three independent reviewers (ML/world-model architect; codebase-fit auditor;
MuJoCo feasibility engineer who empirically ran MuJoCo against the actual URDF).
Each fix below is tagged with the finding it resolves.

### BLOCKERS resolved
- **B1 — collapse-prone objective.** The naive ELBO reconstructed the encoder's
  *own trainable* `obs_embed` (no stop-gradient) → representation collapse. **Fix:**
  reconstruct **raw fixed modalities** (motor_state, range, lidar) via new
  per-modality decoder heads. Genuine fixed targets → no collapse.
- **B2 — vision hardwired on.** Encoder always builds `vision_proj`; "masking via
  valid_mask" trains 57% dead fusion input and leaves the vision pathway untrained
  → deploy-time distribution shift. **Fix:** add `_vision_enabled = cfg.vision_dim
  > 0` mirroring the existing audio/lidar pattern. Pretrain with `vision_dim=0`
  (vision branch absent). Default `vision_dim=256` → **deployed model byte-identical**.
- **B3 — greenfield trainer.** `_train_rssm` is a bare stub; RSSM has no trainable
  path (`observe_step`/`imagine_step` are `@torch.no_grad()`). **Fix:** new
  `RSSM.train_sequence`, `RSSMPretrainer`, `build_rssm_trainable(cfg)`, and full
  orchestrator construction/wiring.

### MAJORS resolved
- **Objective hygiene.** Drop the reward head from the pretraining loss (placeholder
  `-‖pose-goal‖` reward is learnable from position and masks collapse).
- **KL stability.** Free-bits + KL-balancing + fp32/clamped `exp()`:
  `kl = α·KL(sg(post)‖prior) + (1-α)·KL(post‖sg(prior))`, clamped at `free_nats`;
  `logvar.clamp(-10,10)`; computed under `autocast(enabled=False)`.
- **Golden test flakiness.** Replace ±1% point-wise curve with **monotonic-decrease
  + final-loss-below-threshold**, CPU-deterministic (`use_deterministic_algorithms(True)`),
  **non-gating** diagnostic.
- **`wheel_slip` has no MuJoCo home.** Map to a **documented multiplicative obs-noise
  proxy** on `wheel_vel`/pose. friction→`geom_friction[:,0]`, mass→`body_mass`+inertia
  recompute, motor_gain→`actuator_gainprm`.
- **URDF unusable as-is** (welded base, no actuators/sensors/floor). Hand-author a
  **standalone MJCF** `assets/rover/mse6_4wd.xml` + a **rest-state contact + finite-`qacc`
  assertion** (silent-NaN-garbage footgun).
- **Async blocking.** `_train_rssm` is async on the orchestrator's event loop; the
  torch loop must run in `asyncio.to_thread`, with thermal-clearance checks between
  epochs so the safety pause isn't starved.
- **Factual corrections.** Record class is `MouseDroidExperienceRecord` (no
  imu/chassis/wheel/lidar fields) → `EpisodeBatch` is a new dataclass, not a Record
  mirror. `action_dim` is **3** in RSSM but **2** in rover path → pad `[vx,0,omega]`.
  Reuse `[arm]` extra (no redundant `[sim]`). Reuse `TrainingConfig` fields
  (`kl_beta`, `sequence_length`, `n_episodes`, `learning_rate`, `epochs`,
  `batch_size`, `weights_dir`) — add only new pretrain knobs. `wheel_radius`/
  `track_width` come from `RobotConfig`.

## 5. Components

### 5.1 Encoder change — make vision optional
**File:** `src/mousedroid/world_model/encoder.py`
- Add `_vision_enabled = cfg.vision_dim > 0 and cfg.vision_proj_dim > 0`,
  conditionally build `vision_proj` and include `vision_proj_dim` in `fused_dim`,
  exactly mirroring the existing `_audio_enabled`/`_lidar_enabled` blocks.
- `forward(...)`: when vision disabled, skip the vision branch (tolerate
  `vision=None`); do not add a vision part to `parts`.
- Expose `vision_enabled` property for symmetry.
- **Backwards-compat:** default `vision_dim=256` → branch built → byte-identical.

### 5.2 RSSM trainable forward + raw decoders
**File:** `src/mousedroid/world_model/rssm.py`
- New decoder heads (only used in training; deployment path untouched):
  `decode_motor(h,z)→motor_state_dim`, `decode_range(h,z)→1`,
  `decode_lidar(h,z)→lidar_dim` (lidar head built only when `lidar_dim>0`).
- New `train_sequence(obs, actions) -> dict[str, Tensor]` (NO `no_grad`):
  - Inputs are batched `(B, T, …)` tensors already adapted (5.4).
  - Roll posterior/prior dynamics over T; single `.backward()` by caller.
  - Loss = Σ recon-MSE(raw modalities) + `kl_beta` · free-bits-balanced KL.
  - Returns per-term losses for logging + a `posterior_std` probe (collapse guard).
- KL helper: extend `latent_utils` with a balanced/free-bits variant (or compute
  inline) — fp32, `logvar.clamp(-10,10)`.
- Existing `observe_step`/`imagine_step` untouched (invariant #7 preserved).

### 5.3 RoverMuJoCoEnv + MJCF
**Files:** `src/mousedroid/sim/mujoco_rover_env.py`, `assets/rover/mse6_4wd.xml`
- `RoverMuJoCoEnv(cfg: RoverConfig, wheel_radius_m, track_width_m)` (from `RobotConfig`),
  implements `RoverEnvProtocol`. `mujoco` imported lazily inside `__init__`.
- **Identical obs-dict contract** to `MockRoverEnv`: keys/shapes/order
  (`imu`(6), `chassis_pose`(4), `wheel_vel`(4, FL/FR/RL/RR), `lidar`(N)).
- MJCF: chassis box + freejoint + 4 hinge wheels + velocity actuators + plane +
  4 perimeter walls (so `<rangefinder>` lidar carries signal; normalize the `-1`
  no-hit sentinel). IMU from `<accelerometer>`+`<gyro>` (document gravity inclusion).
- DR-param consumption (documented mapping; `wheel_slip` = obs-noise proxy).
- **Rest-state assertion** at init: 4 wheels in contact + finite `qacc` at `mj_forward`.
- Path resolution via the repo's `_REPO_ROOT` convention (no `os.getcwd()`).

### 5.4 RoverObsAdapter
**File:** `src/mousedroid/training/rover_obs_adapter.py` (or `sim/`)
- Maps rover obs dict + `step` `info` → RSSM encoder inputs:
  `motor_state=[vx, vy=0, omega, battery_const]`, `distance_m`=min-forward range,
  `lidar_features`=normalized sectors, **vision omitted**, `valid_mask` in
  `SENSOR_SLOT_MAP` order. Pure, deterministic, no hidden state.

### 5.5 SimEpisodeGenerator + EpisodeBatch
**File:** `src/mousedroid/training/sim_episode_generator.py`
- Seeded (`np.random.Generator`) rollout: N episodes × T steps under a smoothed
  uniform-random wheel-command policy. Emits `EpisodeBatch` (new dataclass; fields
  for the rover modalities + action + reward, in-memory only).

### 5.6 RSSMPretrainer
**File:** `src/mousedroid/training/rssm_pretrainer.py`
- Adam loop over batched sequences, AMP (KL in fp32), grad-clip, structlog
  per-epoch loss + `posterior_std` probe, checkpoint to `weights_dir`.

### 5.7 Factory + orchestrator wiring
**Files:** `src/mousedroid/factory.py`, `src/mousedroid/training/pipeline_orchestrator.py`
- `build_rover_env`: replace the `"mujoco"` `NotImplementedError` branch with
  `RoverMuJoCoEnv(...)` (concrete import inside the factory — invariant #1).
- New `build_rssm_trainable(cfg) -> RSSM` returning the concrete nn.Module.
- `_train_rssm`: build trainable RSSM + device + env + generator + pretrainer;
  run the loop in `asyncio.to_thread`; thermal checks between epochs. Inert
  (stub-equivalent) unless rover backend is `mujoco` AND pretrain enabled.

### 5.8 Config (all additive; invariant #9)
**File:** `src/mousedroid/config/schema.py`
- `ModelConfig`: ensure `vision_dim=0` is a valid disable path (no new field needed
  if `_vision_enabled` keys off `vision_dim`; pretrain uses a model-config variant).
- New `MujocoSimConfig` sub-block (consumed only when `backend=="mujoco"`): mjcf
  path, sim_dt, friction/solref defaults, arena size, lidar sectors, battery const.
- New pretrain knobs under `TrainingConfig` (reusing existing fields): `free_nats`,
  `kl_balance_alpha`, `grad_clip`, `rssm_pretrain_enabled`, checkpoint name.
- Reuse `[arm]` extra (`mujoco>=3.0`). No `[sim]` extra.

## 6. Invariants honored
Protocol-DI (`RoverEnvProtocol`); factory the only concrete-import site; **no
hardcoded values** (MJCF path, friction, β, lr, arena, battery const all from
config); structlog throughout; `torch.no_grad()` preserved on inference paths
(training path deliberately grad-enabled); deterministic seeding;
**backwards-compatible** config + byte-identical deployed model + unchanged `mock`
backend.

## 7. Testing (mirrors repo tier split)
- **Unit:** encoder vision-optional (built/absent by `vision_dim`); `RoverMuJoCoEnv`
  satisfies `RoverEnvProtocol`; obs-dict parity vs `MockRoverEnv`; DR-param plumbing;
  rest-state NaN guard; `RoverObsAdapter` shapes + mask order; `RSSM.train_sequence`
  finite-decreasing loss + grad flow + **no-collapse probe** (`posterior_std` > floor).
- **Integration:** generator → adapter → pretrainer → checkpoint round-trip (≤4 eps).
- **Regression:** pre-feature YAML loads; `mock` backend unchanged; `vision_dim=256`
  model byte-identical; **golden loss** (monotonic + threshold, CPU-deterministic, non-gating).
- **Smoke:** sub-second import; `build_rover_env(backend="mujoco")` returns right type.
- All MuJoCo-gated tests use `pytest.importorskip("mujoco")`.
- Full local CI (`ruff check` + `ruff format --check` + `mypy --strict` +
  `pytest --cov --cov-fail-under=85`) green before finish.

## 8. Out of scope (explicit follow-ons)
- MuJoCo camera rendering → 256-d `vision_features` (vision-on RSSM fine-tune).
- Sim→LMDB persistence + Phase 2 sim:real mixer ingestion.
- Reward-head training against the real constitutional reward.
- On-device / in-the-loop sim (Phase 6).
- Mecanum drivetrain variant.

## 9. Risks
- MJCF wheel-grounding discipline (silent NaN) — mitigated by the rest-state assertion.
- Pretraining on proprio+range only teaches a dynamics-core model, not the full
  perception stack — accepted and documented; vision fine-tune is the follow-on.
- MuJoCo determinism is platform/version-bound — golden test is tolerance-based + non-gating.
