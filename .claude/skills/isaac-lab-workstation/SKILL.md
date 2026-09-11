---
description: Operate the opt-in Isaac Lab rover training backend (schema, fakes, RSSM gates, no YAML keys, no USD commits)
status: active
---

# Isaac Lab workstation

Use this when touching `src/mousedroid/sim/isaaclab/`, nested
`RoverIsaacSimConfig`, RSSM physics-backend gates, or workstation Isaac docs.

Isaac Lab is **not** on the Jetson, not in hosted CI, and not the runtime
harness (`Settings.harness` stays `None`). CHARTER M5 CI physics remains MuJoCo.
Frozen arm skills (`robot-arm-trainer`, `sim-test`, `train-policy`) stay
MuJoCo-only.

## Do

- Tune scene knobs on nested `RoverIsaacSimConfig` in
  `src/mousedroid/config/schema/sim.py` (defaults only). Do **not** add keys
  to `config/*.yaml`.
- Build the env through `src/mousedroid/factory/world_model.py`
  (`build_rover_env`).
- Keep IMU/LiDAR on duck-typed readers; live `build()` in
  `src/mousedroid/sim/isaaclab/rover_env.py` wires the chassis contact sensor
  only.
- Set `rover.reward` before RSSM pretrain. Missing reward skips with
  `reason=isaac_reward_block_required` in
  `src/mousedroid/training/pipeline_orchestrator.py`.
- Convert URDF locally; keep `*.usd` gitignored (`scripts/convert_urdf_to_usd.py`).
- Dispatch `config-guardian` + `test-engineer` on Isaac diffs.

## Do not

- Add `MetricsConfig.track_isaac_sim` or enable `Settings.harness`.
- Map 6-DoF IMU into RSSM `imu_dim`. Do not use hardware `LidarConfig` for
  sim bins.
- Hot-load PPO into `src/mousedroid/vla/policy.py` (catalog F-047 deferred).
- Register `@pytest.mark.isaac` (not in `pyproject.toml` `--strict-markers`).
- Treat skip-all `importorskip("isaaclab")` as Golden Rule `done`.
- Freeze `src/mousedroid/sim/isaaclab/**` in `.claude/workforce.yaml`.

## Commands

```bash
make install          # CI extras only — does not install [isaac]
make install-isaac    # workstation only (Linux + Isaac Sim)
python3 tools/validate_skill_commands.py
bash scripts/validations/F-043.sh
bash scripts/validations/F-044.sh
bash scripts/validations/F-045.sh
bash scripts/validations/F-046.sh
bash scripts/validations/F-048.sh
```

## Docs

- `HARNESS_SPEC.md` §16
- `docs/architecture/c4-rssm-sim-pretraining.md`
- `docs/architecture/ADR-009-isaac-lab-phase-b.md`
- `features.yaml` (F-043–F-049)
- `pyproject.toml` (`[isaac]` extra)
