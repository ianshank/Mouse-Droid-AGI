from __future__ import annotations

import pytest
import torch

from mousedroid.config.schema import ModelConfig, RewardConfig
from mousedroid.reward.model import MultiObjectiveRewardModel


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig()


@pytest.fixture
def reward_cfg() -> RewardConfig:
    return RewardConfig()


@pytest.fixture
def reward_model(model_cfg: ModelConfig, reward_cfg: RewardConfig) -> MultiObjectiveRewardModel:
    return MultiObjectiveRewardModel(model_cfg, reward_cfg)


def test_constructor(reward_model: MultiObjectiveRewardModel) -> None:
    assert len(reward_model.heads) == 4


def test_constructor_head_names(reward_model: MultiObjectiveRewardModel) -> None:
    expected = {"truthfulness", "helpfulness", "safety", "engagement"}
    assert set(reward_model.heads.keys()) == expected


def test_compute_reward_returns_dict_with_4_keys(
    reward_model: MultiObjectiveRewardModel, model_cfg: ModelConfig,
) -> None:
    state = torch.randn(2, model_cfg.obs_dim)
    scores = reward_model.compute_reward(state)
    assert len(scores) == 4
    for name in ("truthfulness", "helpfulness", "safety", "engagement"):
        assert name in scores


def test_compute_reward_output_shapes(
    reward_model: MultiObjectiveRewardModel, model_cfg: ModelConfig,
) -> None:
    state = torch.randn(3, model_cfg.obs_dim)
    scores = reward_model.compute_reward(state)
    for tensor in scores.values():
        assert tensor.shape == (3, 1)


def test_aggregate_with_known_weights() -> None:
    cfg_m = ModelConfig(obs_dim=4)
    cfg_r = RewardConfig(
        weight_truthfulness=1.0,
        weight_helpfulness=0.0,
        weight_safety=0.0,
        weight_engagement=0.0,
    )
    model = MultiObjectiveRewardModel(cfg_m, cfg_r)
    scores = {
        "truthfulness": torch.tensor([[5.0]]),
        "helpfulness": torch.tensor([[3.0]]),
        "safety": torch.tensor([[2.0]]),
        "engagement": torch.tensor([[1.0]]),
    }
    result = model.aggregate(scores)
    assert result.item() == pytest.approx(5.0)


def test_aggregate_with_all_zero_inputs(reward_model: MultiObjectiveRewardModel) -> None:
    head_names = ("truthfulness", "helpfulness", "safety", "engagement")
    scores = {name: torch.zeros(1, 1) for name in head_names}
    result = reward_model.aggregate(scores)
    assert result.item() == pytest.approx(0.0)


def test_forward_returns_scalar(
    reward_model: MultiObjectiveRewardModel, model_cfg: ModelConfig,
) -> None:
    state = torch.randn(2, model_cfg.obs_dim)
    result = reward_model(state)
    assert result.shape == (2, 1)


def test_forward_is_differentiable(
    reward_model: MultiObjectiveRewardModel, model_cfg: ModelConfig,
) -> None:
    state = torch.randn(1, model_cfg.obs_dim, requires_grad=True)
    result = reward_model(state)
    result.sum().backward()
    assert state.grad is not None
