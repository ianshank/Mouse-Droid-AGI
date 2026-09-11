# C4 Component — Sim → RSSM Dynamics Pretraining (MuJoCo + optional Isaac Lab)

> Sim-first RSSM world-model training for the MSE-6 rover. A physics simulator
> (MuJoCo on CI; optional Isaac Lab on a workstation GPU) generates episodes
> that pretrain the RSSM dynamics core (vision OFF). A follow-on phase renders an
> RGB camera on **MuJoCo only** to fine-tune with vision ON. Everything is opt-in
> and runs OUTSIDE the 30 Hz reactive control loop.

## Component Diagram

```mermaid
C4Component
title Sim → RSSM Pretraining + Vision Fine-Tune — Component Diagram

Container_Boundary(train, "Offline training (PipelineOrchestrator rssm phase)") {

    Component_Boundary(sim, "Simulation (sim/)") {
        Component(env, "RoverEnvProtocol", "mujoco | isaac_lab", "Skid-steer physics; obs-parity with MockRoverEnv; Isaac optional")
        Component(mjcf, "mse6_4wd.xml", "MJCF asset", "MuJoCo chassis + 4 wheels + walls + rangefinder + camera")
        Component(usd, "mse6_4wd.usd", "USD (operator-local)", "Isaac Lab articulation; generated, gitignored")
        Component(render, "render_rgb()", "mujoco.Renderer", "Lazy offscreen RGB (vision fine-tune only)")
        Component(dr, "DomainRandomizer", "config ranges", "Per-episode friction/slip/mass/motor_gain")
    }

    Component_Boundary(data, "Data path (training/)") {
        Component(gen, "SimEpisodeGenerator", "seeded policy", "Smoothed-random rollouts → EpisodeBatch (B,T,…)")
        Component(adapt, "RoverObsAdapter", "pure fn", "obs dict + info → encoder tensors (vision slot gated)")
        Component(feat, "MeanPoolExtractor", "non-learned", "RGB → 256-d vision_features (== deployed mean_pool)")
    }

    Component_Boundary(model, "World model (world_model/)") {
        Component(rssm, "RSSM.train_sequence", "torch.nn", "Grad rollout: raw-modality recon + balanced free-bits KL (fp32)")
        Component(enc, "MultimodalEncoder", "torch.nn", "Vision optional (vision_dim=0); fuses motor/range/lidar/(vision)")
        Component(dec, "RawModalityDecoders", "torch.nn", "Pretraining-only recon heads (NOT on the deployed RSSM)")
        Component(mig, "checkpoint_migration", "state-dict", "vision-OFF ckpt → vision-ON RSSM (dynamics copied verbatim)")
    }

    Component(pre, "RSSMPretrainer", "Adam + AMP", "Optimizer loop over EpisodeBatch; writes checkpoint")
}

Component(factory, "factory/world_model.py", "DI", "build_rover_env / build_rssm_trainable / build_rssm_vision_finetune / build_vision_feature_extractor")
ComponentDb(ckpt, "Checkpoints", "weights_dir", "rssm_pretrained.pt / rssm_vision_finetuned.pt")

Rel(factory, env, "builds (backend=mujoco or isaac_lab)")
Rel(env, mjcf, "loads when backend=mujoco")
Rel(env, usd, "loads when backend=isaac_lab")
Rel(env, render, "exposes")
Rel(gen, env, "reset/step")
Rel(gen, render, "renders (vision)")
Rel(render, feat, "RGB →")
Rel(gen, adapt, "per step")
Rel(gen, dr, "per episode → env.apply_domain_params")
Rel(gen, rssm, "EpisodeBatch")
Rel(pre, rssm, "train_sequence(batch, decoders)")
Rel(rssm, enc, "encode")
Rel(rssm, dec, "reconstruct")
Rel(mig, ckpt, "load vision-OFF")
Rel(factory, mig, "build_rssm_vision_finetune")
Rel(pre, ckpt, "save")
```

## Key contracts

- **Backwards-compat (invariant #9).** `rover.sim.backend` default `mock`,
  `MujocoSimConfig.render_vision` default `False`, and
  `TrainingConfig.rssm_pretrain_enabled` / `rssm_vision_finetune_enabled` default
  `False` — pre-feature YAML, existing checkpoints, and the deployed world model
  are byte-identical. `MultimodalEncoder` default (`vision_dim=256`) is unchanged.
- **No representation collapse.** `train_sequence` reconstructs the RAW sim
  observations (fixed targets), NOT the encoder's own trainable `obs_embed`. The
  reconstruction heads (`RawModalityDecoders`) live OFF the RSSM so the deployment
  `state_dict` and seeded init never shift.
- **No CNN trained.** Sim `vision_features` use the same non-learned
  `MeanPoolExtractor` as the deployed `mean_pool` camera path, so sim/real feature
  distributions match by construction.
- **Weight transfer is reuse, not new code.** `checkpoint_migration` (extended for
  the vision modality) copies the dynamics core verbatim and Kaiming-inits the new
  vision fusion columns + `vision_proj`.
- **Outside the hot loop.** All of this is offline training; the 30 Hz reactive
  loop (RSSM `observe_step`/`imagine_step` → MCTS → ESP32) stays `@torch.no_grad`
  and LLM-free. The blocking torch loop runs in `asyncio.to_thread`.
- **No hardcoded values (invariant #3).** MJCF path, arena size, lidar
  sectors/range/geometry, render resolution, exploration bounds + smoothing, KL
  knobs, seeds, and DR ranges all come from Pydantic config.

## Operator usage

```bash
# 1) Pretrain the dynamics core (vision OFF) on MuJoCo episodes:
#    rover.sim.backend: mujoco ; training.rssm_pretrain_enabled: true
# 2) Vision-on fine-tune of the pretrained checkpoint:
#    training.rssm_vision_finetune_enabled: true
#    training.rssm_finetune_checkpoint: weights/rssm_pretrained.pt
python -m mousedroid.training.pipeline_orchestrator --config <training.yaml>
```

Plan + spec: `docs/superpowers/plans/2026-06-07-phase5-mujoco-rssm-pretraining.md`,
`docs/superpowers/plans/2026-06-08-vision-on-rssm-finetune.md`.

## F-043+ addendum — Isaac Lab is a second physics backend (not MuJoCo-only)

The pipeline is **no longer MuJoCo-only**. `factory.build_rover_env` already
dispatches `mock` | `mujoco` | `isaac_lab`. `PipelineOrchestrator._train_rssm`
runs when `training.rssm_pretrain_enabled` and
`rover.sim.backend in {mujoco, isaac_lab}`. The default `mock` backend still
skips (byte-identical). Vision-on fine-tune stays MuJoCo until Isaac grows
`render_rgb`.

Isaac `lidar_dim` is sized from `RoverObservationConfig.lidar_num_sectors`,
not hardware `LidarConfig`. Battery voltage for the adapter is
`rover.sim.battery_voltage_const_v` (parent field; the nested MuJoCo
`battery_voltage_const_v` stays at 12.0 for YAML compat and is unused by
the adapter). Isaac RSSM skips when `rover.reward is None`
(`reason=isaac_reward_block_required`). Training metrics stay on MLflow;
there is no Prometheus `track_isaac_sim` family. Live `build()` wires
contact only; IMU/LiDAR are duck-typed readers. Replicator is a
present-check, not a write.

