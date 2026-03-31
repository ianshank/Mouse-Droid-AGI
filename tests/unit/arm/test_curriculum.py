"""Tests for curriculum learning manager."""

from __future__ import annotations

from mousedroid.arm.environments.curriculum import CurriculumManager
from mousedroid.config.schema import ArmCurriculumConfig


def _make_manager(
    stages: list[int] | None = None,
    threshold: float = 0.8,
    eval_episodes: int = 5,
) -> CurriculumManager:
    """Create curriculum manager with test-friendly defaults."""
    cfg = ArmCurriculumConfig(
        stages=stages or [1, 3, 5],
        promotion_threshold=threshold,
        promotion_eval_episodes=eval_episodes,
    )
    return CurriculumManager(cfg)


class TestCurriculumManager:
    """Test CurriculumManager stage progression."""

    def test_initial_stage(self) -> None:
        mgr = _make_manager()
        assert mgr.current_difficulty == 1
        assert mgr.current_stage_index == 0

    def test_not_final_stage_initially(self) -> None:
        mgr = _make_manager()
        assert not mgr.is_final_stage

    def test_success_rate_zero_initially(self) -> None:
        mgr = _make_manager()
        assert mgr.success_rate == 0.0

    def test_no_promotion_before_enough_episodes(self) -> None:
        mgr = _make_manager(eval_episodes=5)
        for _ in range(4):
            promoted = mgr.record_episode(True)
            assert not promoted

    def test_promotion_on_threshold_met(self) -> None:
        mgr = _make_manager(eval_episodes=5, threshold=0.8)
        # 5 successes -> 100% rate -> promote
        for _i in range(5):
            promoted = mgr.record_episode(True)
        assert promoted
        assert mgr.current_difficulty == 3
        assert mgr.current_stage_index == 1

    def test_no_promotion_below_threshold(self) -> None:
        mgr = _make_manager(eval_episodes=5, threshold=0.8)
        # 3 success + 2 failure = 60% < 80%
        for success in [True, True, True, False, False]:
            promoted = mgr.record_episode(success)
        assert not promoted
        assert mgr.current_difficulty == 1

    def test_final_stage_no_promotion(self) -> None:
        mgr = _make_manager(stages=[1, 3], eval_episodes=3, threshold=0.5)
        # Promote to stage 2
        for _ in range(3):
            mgr.record_episode(True)
        assert mgr.current_difficulty == 3
        assert mgr.is_final_stage

        # Try to promote again — should not
        for _ in range(3):
            promoted = mgr.record_episode(True)
        assert not promoted
        assert mgr.current_difficulty == 3

    def test_reset(self) -> None:
        mgr = _make_manager(eval_episodes=3, threshold=0.5)
        for _ in range(3):
            mgr.record_episode(True)
        assert mgr.current_stage_index == 1

        mgr.reset()
        assert mgr.current_stage_index == 0
        assert mgr.success_rate == 0.0
