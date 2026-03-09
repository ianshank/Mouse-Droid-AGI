"""Tests for Three Laws integration in the training pipeline."""

from __future__ import annotations

import numpy as np
from training.collect_annotations import INTENTION_LABELS, label_intention
from training.train_bdi import _INTENTION_CLASSES

from mousedroid.cognitive.constitutional_rl import ConstitutionalChecker, ConstitutionalRLConfig
from mousedroid.safety.three_laws import RoboticsLawChecker


class TestExpandedIntentionLabels:
    def test_10_classes(self) -> None:
        assert len(INTENTION_LABELS) == 10

    def test_protect_human_label_exists(self) -> None:
        assert "protect_human" in INTENTION_LABELS
        assert INTENTION_LABELS.index("protect_human") == 8

    def test_obey_command_label_exists(self) -> None:
        assert "obey_command" in INTENTION_LABELS
        assert INTENTION_LABELS.index("obey_command") == 9

    def test_bdi_matches_labels(self) -> None:
        assert len(INTENTION_LABELS) == _INTENTION_CLASSES


class TestProtectHumanIntention:
    def test_human_proximity_labels_protect(self) -> None:
        from mousedroid.sensing.bundle import MouseDroidObservationBundle

        obs = MouseDroidObservationBundle()
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        label = label_intention(
            action,
            obs,
            human_detected=True,
            human_dist_m=0.2,
        )
        assert label == 8  # protect_human

    def test_no_human_not_protect(self) -> None:
        from mousedroid.sensing.bundle import MouseDroidObservationBundle

        obs = MouseDroidObservationBundle()
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        label = label_intention(action, obs, human_detected=False)
        assert label != 8


class TestObeyCommandIntention:
    def test_command_labels_obey(self) -> None:
        from mousedroid.sensing.bundle import MouseDroidObservationBundle

        obs = MouseDroidObservationBundle()
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        label = label_intention(action, obs, commanded_action=cmd)
        assert label == 9  # obey_command

    def test_no_command_not_obey(self) -> None:
        from mousedroid.sensing.bundle import MouseDroidObservationBundle

        obs = MouseDroidObservationBundle()
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        label = label_intention(action, obs)
        assert label != 9


class TestConstitutionalCheckerWithLawChecker:
    def test_law_checker_runs_first(self) -> None:
        law_checker = RoboticsLawChecker(human_safety_radius_m=0.5)
        checker = ConstitutionalChecker(
            ConstitutionalRLConfig(),
            law_checker=law_checker,
        )
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {"human_detected": True, "human_dist_m": 0.1, "mcts_sims": 50}
        safe, violations = checker.check(action, ctx)
        # Should have Law 1 violations from law checker
        assert any("[Law 1]" in v for v in violations)
        assert np.allclose(safe, 0.0, atol=0.01)

    def test_no_law_checker_backward_compatible(self) -> None:
        checker = ConstitutionalChecker(ConstitutionalRLConfig())
        action = np.array([0.1, 0.0], dtype=np.float64)
        ctx = {"battery_v": 12.0, "obstacle_dist_m": 2.0, "mcts_sims": 50}
        _safe, violations = checker.check(action, ctx)
        assert violations == []

    def test_law1_negative_reward_in_rollout(self) -> None:
        """Verify that law1 violations produce negative reward label."""
        law_checker = RoboticsLawChecker(human_safety_radius_m=0.5)
        checker = ConstitutionalChecker(
            ConstitutionalRLConfig(),
            law_checker=law_checker,
        )
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {"human_detected": True, "human_dist_m": 0.1, "mcts_sims": 50}
        _, violations = checker.check(action, ctx)

        law1_violations = [v for v in violations if v.startswith("[Law 1]")]
        assert len(law1_violations) > 0
        # Per train_constitutional_rl.py: law1 → reward = -1.0
        reward = -1.0 if law1_violations else 0.0
        assert reward == -1.0

    def test_validation_zero_law1_safe_policy(self) -> None:
        """A zero-output policy should have no law violations."""
        from mousedroid.cognitive.constitutional_rl import PolicyMLP

        law_checker = RoboticsLawChecker()
        checker = ConstitutionalChecker(
            ConstitutionalRLConfig(),
            law_checker=law_checker,
        )
        policy = PolicyMLP(input_dim=8, action_dim=2)
        policy._w1 *= 0.0
        policy._w2 *= 0.0

        total_law1 = 0
        for _ in range(50):
            state = np.random.randn(8).astype(np.float32)
            action = policy.forward(state)
            ctx = {"battery_v": 12.0, "obstacle_dist_m": 2.0, "mcts_sims": 50}
            _, violations = checker.check(action, ctx)
            total_law1 += sum(1 for v in violations if v.startswith("[Law 1]"))

        assert total_law1 == 0
