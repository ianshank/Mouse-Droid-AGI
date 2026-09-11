# Design — Isaac Lab sensor-parity

## D-1. Nested schema, not LidarConfig

Isaac bins come from `RoverObservationConfig.lidar_num_sectors` and
`lidar_max_range_m` (sim 4.0 m). Hardware `LidarConfig` stays at 36 sectors /
12 m. `RoverIsaacSimConfig.device_headless` / `device_gui` preserve the
existing `headless` branch. `usd_path=None` keeps `.urdf` → `.usd`.

## D-2. Optional seams, not a new trainer

`SimEpisodeGenerator` already `getattr`s `to_body_action` and
`apply_domain_params`. Isaac grows those methods and shared kinematics in
`src/mousedroid/sim/kinematics.py`. No Dreamer-V3, no PPO trainer.

## D-3. Lift gates, keep vision MuJoCo-only

`PipelineOrchestrator._train_rssm` accepts `backend in {mujoco, isaac_lab}`
when `rssm_pretrain_enabled`. `_run_vision_finetune` stays mujoco-only until
Isaac `render_rgb` exists. `build_rssm_trainable` sizes `lidar_dim` from
observation sectors for both physics backends.

## D-4. No 30 Hz contract change

F-047 PPO ONNX is deferred. Deployed RSSM/VLA/MCTS action is 3-DoF body
velocity; rover policy is 2-D differential. Do not load a PPO graph through
`vla/policy.py`. Training metrics use existing MLflow, not
`MetricsConfig.track_isaac_sim`.
