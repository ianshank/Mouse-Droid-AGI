"""F-046 backwards-compat: mock + disabled flags still skip RSSM pretrain."""

from __future__ import annotations

from mousedroid.config.schema import RoverConfig, Settings, TrainingConfig


def test_rssm_pretrain_still_default_off() -> None:
    cfg = Settings(mock_hardware=True)
    assert cfg.training.rssm_pretrain_enabled is False
    assert cfg.training.rssm_vision_finetune_enabled is False


def test_default_rover_backend_is_mock() -> None:
    cfg = Settings(mock_hardware=True, rover=RoverConfig())
    assert cfg.rover is not None
    assert cfg.rover.sim.backend == "mock"
    assert TrainingConfig().rssm_pretrain_enabled is False
