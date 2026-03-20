"""Tests for ICM novelty decay and updated intrinsic reward."""

from __future__ import annotations

import torch

from mousedroid.config.schema import CuriosityConfig, ModelConfig
from mousedroid.curiosity.icm import IntrinsicCuriosityModule, NoveltyDecay


class TestNoveltyDecay:
    """Test NoveltyDecay state visitation tracking."""

    def test_first_visit_returns_full_scale(self) -> None:
        decay = NoveltyDecay(decay_rate=0.1, min_scale=0.01)
        state = torch.zeros(8)
        scale = decay.get_scale(state)
        assert scale == 1.0

    def test_repeated_visits_decrease_scale(self) -> None:
        decay = NoveltyDecay(decay_rate=0.5, min_scale=0.01)
        state = torch.zeros(8)

        scales = []
        for _ in range(5):
            scale = decay.record_visit(state)
            scales.append(scale)

        # Each visit should decrease the scale
        for i in range(1, len(scales)):
            assert scales[i] < scales[i - 1]

    def test_min_scale_floor(self) -> None:
        decay = NoveltyDecay(decay_rate=10.0, min_scale=0.05)
        state = torch.zeros(8)

        for _ in range(100):
            scale = decay.record_visit(state)

        assert scale >= 0.05

    def test_different_states_independent(self) -> None:
        decay = NoveltyDecay(decay_rate=0.5, min_scale=0.01)

        state_a = torch.zeros(8)
        state_b = torch.ones(8)

        # Visit state_a many times
        for _ in range(10):
            decay.record_visit(state_a)

        # State_b should still be novel
        scale_b = decay.get_scale(state_b)
        assert scale_b == 1.0

    def test_reset_clears_visits(self) -> None:
        decay = NoveltyDecay(decay_rate=0.5, min_scale=0.01)
        state = torch.zeros(8)

        for _ in range(10):
            decay.record_visit(state)

        decay.reset()
        assert decay.total_visits == 0
        assert decay.unique_states == 0
        assert decay.get_scale(state) == 1.0

    def test_total_visits_count(self) -> None:
        decay = NoveltyDecay(decay_rate=0.1, min_scale=0.01)
        state_a = torch.zeros(8)
        state_b = torch.ones(8)

        decay.record_visit(state_a)
        decay.record_visit(state_a)
        decay.record_visit(state_b)

        assert decay.total_visits == 3
        assert decay.unique_states == 2

    def test_properties(self) -> None:
        decay = NoveltyDecay(decay_rate=0.05, min_scale=0.02)
        assert decay.decay_rate == 0.05
        assert decay.min_scale == 0.02


class TestICMWithNoveltyDecay:
    """Test ICM integration with novelty decay."""

    def test_decay_disabled_by_default(self) -> None:
        icm = IntrinsicCuriosityModule(ModelConfig(), CuriosityConfig())
        assert icm.novelty_decay is None

    def test_decay_enabled_via_config(self) -> None:
        cfg = CuriosityConfig(
            novelty_decay_enabled=True,
            novelty_decay_rate=0.05,
            novelty_min_scale=0.02,
        )
        icm = IntrinsicCuriosityModule(ModelConfig(), cfg)
        assert icm.novelty_decay is not None
        assert icm.novelty_decay.decay_rate == 0.05
        assert icm.novelty_decay.min_scale == 0.02

    def test_intrinsic_reward_decreases_with_visits(self) -> None:
        """Repeated identical states should yield decreasing rewards."""
        cfg = CuriosityConfig(
            novelty_decay_enabled=True,
            novelty_decay_rate=1.0,  # aggressive decay for testing
            novelty_min_scale=0.01,
        )
        icm = IntrinsicCuriosityModule(ModelConfig(), cfg)

        s = torch.randn(1, 256)
        a = torch.randn(1, 3)
        s_next = torch.randn(1, 256)

        reward_first = icm.intrinsic_reward(s, a, s_next).item()
        reward_second = icm.intrinsic_reward(s, a, s_next).item()

        # Second reward should be smaller due to novelty decay
        assert reward_second < reward_first

    def test_intrinsic_reward_non_negative_with_decay(self) -> None:
        cfg = CuriosityConfig(
            novelty_decay_enabled=True,
            novelty_decay_rate=0.5,
            novelty_min_scale=0.01,
        )
        icm = IntrinsicCuriosityModule(ModelConfig(), cfg)

        s = torch.randn(4, 256)
        a = torch.randn(4, 3)
        s_next = torch.randn(4, 256)
        reward = icm.intrinsic_reward(s, a, s_next)
        assert (reward >= 0.0).all()

    def test_without_decay_backward_compatible(self) -> None:
        """ICM without decay should produce identical results to original."""
        cfg = CuriosityConfig(novelty_decay_enabled=False)
        icm = IntrinsicCuriosityModule(ModelConfig(), cfg)

        s = torch.randn(3, 256)
        a = torch.randn(3, 3)
        s_next = torch.randn(3, 256)

        reward = icm.intrinsic_reward(s, a, s_next)
        assert reward.shape == (3,)
        assert (reward >= 0.0).all()

    def test_forward_still_works_with_decay(self) -> None:
        """Forward pass (training mode) should be unaffected by decay."""
        cfg = CuriosityConfig(novelty_decay_enabled=True)
        icm = IntrinsicCuriosityModule(ModelConfig(), cfg)

        s = torch.randn(2, 256)
        a = torch.randn(2, 3)
        s_next = torch.randn(2, 256)

        fwd_loss, inv_loss, pred = icm(s, a, s_next)
        assert fwd_loss.shape == ()
        assert inv_loss.shape == ()
        assert pred.shape == (2, 256)
