"""build_rssm_trainable returns a concrete trainable RSSM with vision off."""

from __future__ import annotations

from mousedroid.config.schema import RoverConfig, RoverSimConfig, Settings
from mousedroid.factory import build_rssm_trainable
from mousedroid.world_model.rssm import RSSM


def test_returns_trainable_rssm_vision_off() -> None:
    cfg = Settings(mock_hardware=True)
    model = build_rssm_trainable(cfg)
    assert isinstance(model, RSSM)
    assert model.encoder.vision_enabled is False  # pretraining drops vision
    assert any(p.requires_grad for p in model.parameters())


def test_no_rover_keeps_default_lidar_dim() -> None:
    cfg = Settings(mock_hardware=True)  # rover is None
    model = build_rssm_trainable(cfg)
    assert model.cfg.lidar_dim == cfg.model.lidar_dim  # unchanged default


def test_mujoco_rover_enables_lidar_matching_sectors() -> None:
    sectors = 12
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")),
    )
    assert cfg.rover is not None
    cfg = cfg.model_copy(
        update={
            "rover": cfg.rover.model_copy(
                update={
                    "observation": cfg.rover.observation.model_copy(
                        update={"lidar_num_sectors": sectors}
                    ),
                    "sim": cfg.rover.sim.model_copy(
                        update={
                            "mujoco": cfg.rover.sim.mujoco.model_copy(
                                update={"lidar_num_sectors": sectors}
                            )
                        }
                    ),
                }
            )
        }
    )
    model = build_rssm_trainable(cfg)
    assert model.cfg.lidar_dim == sectors  # model reconstructs the full lidar signal
    assert model.encoder.lidar_enabled is True


def test_isaac_lab_rover_uses_observation_lidar_sectors() -> None:
    sectors = 12
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="isaac_lab")),
    )
    assert cfg.rover is not None
    cfg = cfg.model_copy(
        update={
            "rover": cfg.rover.model_copy(
                update={
                    "observation": cfg.rover.observation.model_copy(
                        update={"lidar_num_sectors": sectors}
                    ),
                    "sim": cfg.rover.sim.model_copy(
                        update={
                            "mujoco": cfg.rover.sim.mujoco.model_copy(
                                update={"lidar_num_sectors": 16}
                            )
                        }
                    ),
                }
            )
        }
    )
    model = build_rssm_trainable(cfg)
    assert model.cfg.lidar_dim == sectors
    assert model.encoder.lidar_enabled is True


def test_lidar_dim_zero_when_include_lidar_sectors_false() -> None:
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="isaac_lab")),
    )
    assert cfg.rover is not None
    cfg = cfg.model_copy(
        update={
            "rover": cfg.rover.model_copy(
                update={
                    "observation": cfg.rover.observation.model_copy(
                        update={"include_lidar_sectors": False}
                    )
                }
            )
        }
    )
    model = build_rssm_trainable(cfg)
    assert model.cfg.lidar_dim == 0
    assert model.cfg.lidar_proj_dim == 0
    assert model.encoder.lidar_enabled is False


def test_mujoco_lidar_dim_uses_nested_sectors_not_observation() -> None:
    nested = 12
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")),
    )
    assert cfg.rover is not None
    cfg = cfg.model_copy(
        update={
            "rover": cfg.rover.model_copy(
                update={
                    "observation": cfg.rover.observation.model_copy(
                        update={"lidar_num_sectors": 16}
                    ),
                    "sim": cfg.rover.sim.model_copy(
                        update={
                            "mujoco": cfg.rover.sim.mujoco.model_copy(
                                update={"lidar_num_sectors": nested}
                            )
                        }
                    ),
                }
            )
        }
    )
    model = build_rssm_trainable(cfg)
    assert model.cfg.lidar_dim == nested


def test_overrides_pretrain_knobs_from_training_config() -> None:
    cfg = Settings(mock_hardware=True)
    model = build_rssm_trainable(cfg)
    assert model.cfg.kl_beta == cfg.training.kl_beta
    assert model.cfg.kl_free_nats == cfg.training.rssm_free_nats
    assert model.cfg.kl_balance_alpha == cfg.training.rssm_kl_balance_alpha
