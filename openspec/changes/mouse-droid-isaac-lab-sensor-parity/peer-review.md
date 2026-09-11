# Peer review — Isaac Lab sensor-parity

## Verdict table

| Claim | Verdict |
|---|---|
| `_train_rssm` was already env-agnostic | **REJECTED** — gated on `backend == mujoco` |
| PPO ONNX can reuse DistilledVLAOnnx / TRT | **REJECTED** — CHARTER 30 Hz contract break; F-047 deferred |
| Use `LidarConfig.n_sectors` for sim bins | **REJECTED** — hardware 36 / 12 m vs sim 16 / 4.0 m |
| Skip-all pytest can close `done` | **REJECTED** — validation scripts assert `passed > 0` |
| Prometheus `track_isaac_sim` for training | **REJECTED** — wrong bus; use MLflow |

## Load-bearing pins

1. `RoverIsaacSimConfig.device_headless` FieldInfo default `"cuda:0"`.
2. `ROVER_RSSM_PHYSICS_BACKENDS == {mujoco, isaac_lab}`.
3. `Settings.harness is None`; no `MetricsConfig.track_isaac_sim`.
4. `RoverObsAdapter` still zeros `SENSOR_SLOT_MAP["imu"]`.
