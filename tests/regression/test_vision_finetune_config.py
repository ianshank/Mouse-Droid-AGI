"""Vision-finetune config fields are additive and default OFF."""

from __future__ import annotations

import yaml

from mousedroid.config.schema import MujocoSimConfig, Settings, TrainingConfig


def test_render_defaults_off() -> None:
    m = MujocoSimConfig()
    assert m.render_vision is False
    assert m.render_width > 0
    assert m.render_height > 0
    assert m.camera_name == "rover_cam"


def test_training_finetune_defaults_off() -> None:
    t = TrainingConfig()
    assert t.rssm_vision_finetune_enabled is False
    assert t.rssm_finetune_checkpoint == ""
    assert t.rssm_finetune_epochs > 0
    assert t.rssm_vision_checkpoint_name.endswith(".pt")


def test_pre_feature_yaml_still_loads() -> None:
    raw = yaml.safe_load("mock_hardware: true\nplatform: mouse_droid\n")
    cfg = Settings.model_validate(raw)
    assert cfg.training.rssm_vision_finetune_enabled is False


def test_opt_in_overlay_parses() -> None:
    raw = yaml.safe_load(
        """
        mock_hardware: true
        rover:
          sim:
            backend: mujoco
            mujoco:
              render_vision: true
              render_width: 84
              render_height: 84
        training:
          rssm_vision_finetune_enabled: true
          rssm_finetune_checkpoint: weights/rssm_pretrained.pt
        """
    )
    cfg = Settings.model_validate(raw)
    assert cfg.rover.sim.mujoco.render_vision is True
    assert cfg.rover.sim.mujoco.render_width == 84
    assert cfg.training.rssm_vision_finetune_enabled is True
    assert cfg.training.rssm_finetune_checkpoint.endswith("rssm_pretrained.pt")
