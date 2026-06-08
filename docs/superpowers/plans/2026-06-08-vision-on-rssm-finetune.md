# Vision-On RSSM Fine-Tune — Design + Implementation Plan

> **For agentic workers:** TDD, targeted tests per task, `--no-verify` intermediate commits (CI is the authoritative gate), full `pytest`+coverage gate at the end. Steps use `- [ ]`.

**Goal:** Add MuJoCo camera rendering → `MeanPoolExtractor` → 256-d `vision_features`, then a fine-tune phase that loads a vision-OFF Phase-5-pretrained RSSM, turns vision ON via the existing checkpoint-migration machinery, and fine-tunes — all opt-in and backwards-compatible.

**Architecture:** Reuse two existing systems so no new model is trained for perception and no new weight-transfer code is written: (1) the deployed `MeanPoolExtractor` (non-learned mean-pool→L2) runs on rendered RGB so sim/deploy vision-feature distributions match by construction; (2) `checkpoint_migration.load_rssm_with_migration` already transfers a vision-OFF checkpoint into a vision-ON RSSM (retained-modality fusion columns copied, vision columns Kaiming-init, `vision_proj` created). New work is wiring vision through the Phase-5 seams (env render, adapter, generator, `RawModalityDecoders`, `train_sequence`, pretrainer, orchestrator).

**Tech stack:** Python 3.10+, MuJoCo `>=3.0` `mujoco.Renderer` (offscreen), PyTorch, Pydantic v2, structlog, pytest+hypothesis, ruff 0.8.0, mypy --strict.

## Backwards-compat invariant
`MujocoSimConfig.render_vision` defaults `False` and `TrainingConfig.rssm_vision_finetune_enabled` defaults `False` → Phase 5 (and all existing configs/checkpoints) byte-identical. The deployment RSSM is unaffected (vision was already a supported modality).

## File map
| File | Status | Responsibility |
|---|---|---|
| `src/mousedroid/config/schema.py` | MODIFY | `MujocoSimConfig` render fields; `TrainingConfig` finetune fields |
| `assets/rover/mse6_4wd.xml` | MODIFY | forward-facing `<camera>` |
| `src/mousedroid/sim/mujoco_rover_env.py` | MODIFY | lazy `mujoco.Renderer` + `render_rgb()` |
| `src/mousedroid/factory.py` | MODIFY | `build_vision_feature_extractor`; `build_rssm_vision_finetune` |
| `src/mousedroid/training/rover_obs_adapter.py` | MODIFY | optional `vision_features` → vision slot on |
| `src/mousedroid/training/sim_episode_generator.py` | MODIFY | optional extractor; render+extract; `EpisodeBatch.vision` |
| `src/mousedroid/world_model/rssm.py` | MODIFY | `RawModalityDecoders.decode_vision`; `train_sequence` vision recon |
| `src/mousedroid/training/rssm_pretrainer.py` | MODIFY | thread `vision` through `_to_device` |
| `src/mousedroid/training/pipeline_orchestrator.py` | MODIFY | opt-in vision-finetune in `_train_rssm` |
| `tests/...` | CREATE | unit/integration/regression/smoke per task |

## Tasks

### Task 1 — Config: render + finetune fields
- `MujocoSimConfig`: `render_vision: bool=False`, `render_width: int=64 (gt=0)`, `render_height: int=64 (gt=0)`, `camera_name: str="rover_cam"`.
- `TrainingConfig`: `rssm_vision_finetune_enabled: bool=False`, `rssm_finetune_checkpoint: str=""` (path to the vision-off pretrained ckpt), `rssm_finetune_epochs: int=50 (gt=0)`.
- Regression test: defaults present; pre-feature YAML loads; opt-in overlay parses.

### Task 2 — MJCF camera
- Add `<camera name="rover_cam" mode="fixed" pos="0.11 0 0.05" xyaxes="0 -1 0 0 0 1"/>` (forward-facing) inside chassis body. Test: model loads, `mujoco.mj_name2id(..., mjOBJ_CAMERA, "rover_cam") >= 0`.

### Task 3 — `RoverMuJoCoEnv.render_rgb()`
- Lazy `self._renderer = mujoco.Renderer(model, height, width)` built only when `render_vision`. `render_rgb() -> NDArray[uint8] (H,W,3)` via `update_scene(data, camera=camera_name)` + `render()`. Guard `_require_open`. `close()` frees renderer. Test (importorskip mujoco): shape/dtype; raises clear error when `render_vision=False`.

### Task 4 — `build_vision_feature_extractor(cfg)` factory
- Returns `FeatureExtractorProtocol` = `MeanPoolExtractor(cfg.camera.feature_dim, l2_normalize=cfg.camera.l2_normalize)` (concrete import inside factory). Test: returns extractor; `extract(rgb)` → `(feature_dim,)` float32.

### Task 5 — `RoverObsAdapter` vision-on
- `adapt(obs, info, vision_features=None)`: when `vision_features` is not None, set mask vision slot=1 and include `"vision"`. Default (None) unchanged (vision slot=0). Tests: vision present→slot 1 + key; absent→slot 0, no key (byte-identical to current).

### Task 6 — `SimEpisodeGenerator` vision
- Optional `feature_extractor: FeatureExtractorProtocol | None`. When provided, each step: `rgb = env.render_rgb(); vf = extractor.extract(rgb)`; pass to adapter; stack into `EpisodeBatch.vision` `(B,T,feature_dim)`. When None, `vision` is `(B,T,0)`. Tests (importorskip mujoco): shapes with/without extractor; deterministic.

### Task 7 — `RawModalityDecoders` + `train_sequence` vision recon
- `RawModalityDecoders`: add `self.vision_enabled = cfg.vision_dim>0; if vision_enabled: self.decode_vision = nn.Linear(feat, cfg.vision_dim)`.
- `train_sequence`: `vision = batch["vision"][:, step] if self.encoder.vision_enabled else None`; pass to encoder; if `decoders.vision_enabled and vision is not None`: `recon += mse(decoders.decode_vision(hz), vision)`. Tests: vision-on model+decoders → finite decreasing loss, decode_vision gets grad; vision-off path unchanged.

### Task 8 — Pretrainer threads vision
- `_to_device` includes `vision` (when present on the batch). Test: pretrainer trains a vision-on model+batch → checkpoint; loss decreases.

### Task 9 — `build_rssm_vision_finetune(cfg, checkpoint)` factory
- `load_rssm_with_migration(checkpoint, vision_on_cfg, device)` where `vision_on_cfg = cfg.model` with `vision_dim=cfg.camera.feature_dim`, `vision_proj_dim=cfg.model.vision_proj_dim or 128`, lidar from rover, kl knobs. Returns RSSM with transferred core. Test (write a vision-off ckpt, migrate to vision-on): `encoder.vision_enabled is True`; dynamics weights (gru/posterior/prior) byte-equal to the source.

### Task 10 — Orchestrator opt-in fine-tune
- In `_train_rssm`: when `rssm_vision_finetune_enabled` (and mujoco backend + checkpoint exists), build the vision-finetune RSSM + a `build_vision_feature_extractor`, generator with `render_vision` env + extractor, run the pretrainer for `rssm_finetune_epochs`, save to a `rssm_vision_finetuned.pt` checkpoint. Inert by default. Test: inert when disabled; runs end-to-end when enabled+mujoco (importorskip).

### Task 11 — Integration + regression + smoke + full gate
- Integration: render→extract→generate(vision)→finetune→checkpoint round-trip (loss decreases).
- Regression: `render_vision=False`/`finetune disabled` → byte-identical; deployed RSSM + checkpoint-migration still pass.
- Smoke: lazy import; factory surface.
- Full `ruff`+`format`+`mypy --strict src/mousedroid/`+`pytest tests/ --cov --cov-fail-under=85`.
