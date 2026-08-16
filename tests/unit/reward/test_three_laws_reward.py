"""Tests for Three Laws reward integration in MultiObjectiveRewardModel."""

from __future__ import annotations

import torch

from mousedroid.config.schema import ModelConfig, RewardConfig, ThreeLawsConfig
from mousedroid.reward.model import MultiObjectiveRewardModel, ThreeLawsRewardHead


def _make_small_cfg() -> tuple[ModelConfig, RewardConfig, ThreeLawsConfig]:
    model_cfg = ModelConfig(
        vision_dim=16,
        ultrasonic_dim=1,
        motor_state_dim=4,
        hidden_dim=32,
        latent_dim=8,
        action_dim=3,
        obs_dim=16,
        vision_proj_dim=8,
        ultrasonic_proj_dim=4,
        motor_proj_dim=4,
    )
    reward_cfg = RewardConfig()
    law_cfg = ThreeLawsConfig()
    return model_cfg, reward_cfg, law_cfg


class TestThreeLawsRewardHead:
    def test_head_output_shapes(self) -> None:
        _, _, law_cfg = _make_small_cfg()
        head = ThreeLawsRewardHead(input_dim=16, law_cfg=law_cfg)
        state = torch.randn(4, 16)
        scores = head.compute_scores(state)
        assert "law1_harm" in scores
        assert "law2_obedience" in scores
        assert "law3_preservation" in scores
        for v in scores.values():
            assert v.shape == (4, 1)

    def test_aggregate_shape(self) -> None:
        _, _, law_cfg = _make_small_cfg()
        head = ThreeLawsRewardHead(input_dim=16, law_cfg=law_cfg)
        state = torch.randn(2, 16)
        scores = head.compute_scores(state)
        agg = head.aggregate(scores)
        assert agg.shape == (2, 1)

    def test_law1_negative_suppresses_reward(self) -> None:
        _, _, law_cfg = _make_small_cfg()
        head = ThreeLawsRewardHead(input_dim=16, law_cfg=law_cfg)
        # Force harm head to output very negative
        with torch.no_grad():
            head.heads["law1_harm"].weight.fill_(0.0)
            head.heads["law1_harm"].bias.fill_(-10.0)
            head.heads["law2_obedience"].bias.fill_(1.0)
            head.heads["law3_preservation"].bias.fill_(1.0)

        state = torch.randn(1, 16)
        scores = head.compute_scores(state)
        agg = head.aggregate(scores)
        # Sigmoid(-10) ≈ 0, so output should be near zero
        assert abs(agg.item()) < 0.1


class TestMultiObjectiveWithLaws:
    def test_no_law_cfg_backward_compatible(self) -> None:
        model_cfg, reward_cfg, _ = _make_small_cfg()
        model = MultiObjectiveRewardModel(model_cfg, reward_cfg)
        assert model.law_head is None
        state = torch.randn(2, 16)
        out = model(state)
        assert out.shape == (2, 1)

    def test_with_law_cfg_includes_law_heads(self) -> None:
        model_cfg, reward_cfg, law_cfg = _make_small_cfg()
        model = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=law_cfg)
        assert model.law_head is not None
        state = torch.randn(2, 16)
        out = model(state)
        assert out.shape == (2, 1)

    def test_compute_reward_includes_law_scores(self) -> None:
        model_cfg, reward_cfg, law_cfg = _make_small_cfg()
        model = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=law_cfg)
        state = torch.randn(1, 16)
        scores = model.compute_reward(state)
        assert "law1_harm" in scores
        assert "law2_obedience" in scores
        assert "law3_preservation" in scores
        # Plus the standard 4 heads
        assert "truthfulness" in scores
        assert "safety" in scores

    def test_law_weights_configurable(self) -> None:
        model_cfg, reward_cfg, _ = _make_small_cfg()
        law_cfg1 = ThreeLawsConfig(law1_reward_weight=0.9)
        law_cfg2 = ThreeLawsConfig(law1_reward_weight=0.1)
        m1 = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=law_cfg1)
        m2 = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=law_cfg2)
        assert m1.law_head._weights["law1_harm"] == 0.9
        assert m2.law_head._weights["law1_harm"] == 0.1

    def test_disabled_law_cfg_no_head(self) -> None:
        model_cfg, reward_cfg, _ = _make_small_cfg()
        law_cfg = ThreeLawsConfig(enabled=False)
        model = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=law_cfg)
        assert model.law_head is None

    def test_no_violations_normal_reward(self) -> None:
        model_cfg, reward_cfg, law_cfg = _make_small_cfg()
        model = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=law_cfg)
        state = torch.randn(1, 16)
        out = model(state)
        assert torch.isfinite(out).all()

    def test_gradient_flows_through_law_heads(self) -> None:
        model_cfg, reward_cfg, law_cfg = _make_small_cfg()
        model = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=law_cfg)
        state = torch.randn(1, 16, requires_grad=True)
        out = model(state)
        out.sum().backward()
        assert state.grad is not None
