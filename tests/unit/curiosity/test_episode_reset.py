"""Tests for CuriosityProtocol.reset_episode() and ICM implementation."""

from __future__ import annotations

import torch

from mousedroid.config.schema import CuriosityConfig, ModelConfig
from mousedroid.curiosity.icm import IntrinsicCuriosityModule
from mousedroid.curiosity.protocol import CuriosityProtocol


def _make_icm(novelty_decay: bool = True) -> IntrinsicCuriosityModule:
    """Build a small ICM for testing."""
    model_cfg = ModelConfig(obs_dim=8, action_dim=3)
    curiosity_cfg = CuriosityConfig(novelty_decay_enabled=novelty_decay)
    return IntrinsicCuriosityModule(model_cfg, curiosity_cfg)


class TestICMProtocolConformance:
    """IntrinsicCuriosityModule satisfies CuriosityProtocol."""

    def test_isinstance_check(self) -> None:
        icm = _make_icm()
        assert isinstance(icm, CuriosityProtocol)

    def test_has_reset_episode(self) -> None:
        icm = _make_icm()
        assert callable(icm.reset_episode)


class TestICMResetEpisode:
    """reset_episode() clears novelty-decay visit counts."""

    def test_reset_clears_visit_counts(self) -> None:
        icm = _make_icm(novelty_decay=True)
        assert icm.novelty_decay is not None

        s = torch.randn(1, 8)
        a = torch.zeros(1, 3)
        icm.intrinsic_reward(s, a, s)  # populates visit counts

        assert icm.novelty_decay.total_visits > 0
        icm.reset_episode()
        assert icm.novelty_decay.total_visits == 0

    def test_reset_without_novelty_decay_is_noop(self) -> None:
        """reset_episode() is safe when novelty_decay is disabled."""
        icm = _make_icm(novelty_decay=False)
        assert icm.novelty_decay is None
        icm.reset_episode()  # must not raise

    def test_reset_idempotent(self) -> None:
        """Calling reset_episode() twice does not raise."""
        icm = _make_icm(novelty_decay=True)
        icm.reset_episode()
        icm.reset_episode()
        assert icm.novelty_decay is not None
        assert icm.novelty_decay.total_visits == 0
