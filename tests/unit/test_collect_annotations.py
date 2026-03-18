"""Tests for intention annotation labelling — all paths and edge cases."""

from __future__ import annotations

import numpy as np
from training.collect_annotations import INTENTION_LABELS, label_intention

from mousedroid.sensing.bundle import MouseDroidObservationBundle


def _obs(
    distance_m: float = 4.0,
    battery_v: float = 12.0,
) -> MouseDroidObservationBundle:
    """Build a minimal observation bundle."""
    return MouseDroidObservationBundle(
        _distance_m=distance_m,
        _motor_state=np.array([0, 0, 0, battery_v], dtype=np.float32),
    )


class TestIntentionLabels:
    def test_label_count(self) -> None:
        assert len(INTENTION_LABELS) == 10

    def test_unique_labels(self) -> None:
        assert len(set(INTENTION_LABELS)) == len(INTENTION_LABELS)

    def test_three_laws_labels_present(self) -> None:
        assert "protect_human" in INTENTION_LABELS
        assert "obey_command" in INTENTION_LABELS


class TestLabelIntentionPaths:
    def test_protect_human_close(self) -> None:
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        obs = _obs()
        result = label_intention(action, obs, human_detected=True, human_dist_m=0.3)
        assert result == 8  # protect_human

    def test_protect_human_far_no_trigger(self) -> None:
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        obs = _obs()
        result = label_intention(action, obs, human_detected=True, human_dist_m=1.0)
        assert result != 8

    def test_obey_command(self) -> None:
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        obs = _obs()
        cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        result = label_intention(action, obs, commanded_action=cmd)
        assert result == 9  # obey_command

    def test_low_battery_charge(self) -> None:
        action = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        obs = _obs(battery_v=9.0)
        result = label_intention(action, obs)
        assert result == 6  # charge

    def test_close_obstacle_avoid(self) -> None:
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        obs = _obs(distance_m=0.1)
        result = label_intention(action, obs)
        assert result == 2  # avoid_obstacle

    def test_idle(self) -> None:
        action = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        obs = _obs()
        result = label_intention(action, obs)
        assert result == 7  # idle

    def test_wait_low_speed(self) -> None:
        action = np.array([0.05, 0.03, 0.02], dtype=np.float32)
        obs = _obs()
        result = label_intention(action, obs)
        assert result == 4  # wait

    def test_turn_high_omega(self) -> None:
        action = np.array([0.1, 0.0, 1.0], dtype=np.float32)
        obs = _obs()
        result = label_intention(action, obs)
        assert result == 5  # turn

    def test_backtrack_negative_speed(self) -> None:
        action = np.array([-0.5, 0.0, 0.0], dtype=np.float32)
        obs = _obs()
        result = label_intention(action, obs)
        assert result == 3  # backtrack

    def test_approach_target(self) -> None:
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        obs = _obs(distance_m=2.0)
        result = label_intention(action, obs)
        assert result == 1  # approach_target

    def test_explore_default(self) -> None:
        action = np.array([0.15, 0.15, 0.0], dtype=np.float32)
        obs = _obs(distance_m=0.5)
        result = label_intention(action, obs)
        assert result == 0  # explore

    def test_protect_human_overrides_low_battery(self) -> None:
        """Law 1 (human) takes priority over battery charge."""
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        obs = _obs(battery_v=9.0)
        result = label_intention(
            action,
            obs,
            human_detected=True,
            human_dist_m=0.2,
        )
        assert result == 8  # protect_human, not charge


# ---------------------------------------------------------------------------
# label_intention — configurable threshold params (Phase 3 refactor)
# ---------------------------------------------------------------------------


class TestLabelIntentionConfigThresholds:
    def test_custom_human_safety_radius(self) -> None:
        """A larger safety radius means human at 0.8m still triggers protect_human."""
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        obs = _obs()
        result = label_intention(
            action, obs,
            human_detected=True, human_dist_m=0.8,
            human_safety_radius_m=1.0,
        )
        assert result == 8  # protect_human

    def test_custom_human_safety_radius_miss(self) -> None:
        """A smaller safety radius means human at 0.8m does NOT trigger protect_human."""
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        obs = _obs()
        result = label_intention(
            action, obs,
            human_detected=True, human_dist_m=0.8,
            human_safety_radius_m=0.5,
        )
        assert result != 8  # not protect_human

    def test_custom_battery_warn_v_lower(self) -> None:
        """With a lower battery threshold, 9.5V triggers charge only if < new threshold."""
        action = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        obs = _obs(battery_v=9.5)
        # Default threshold is 10.8 → 9.5V triggers charge
        result_default = label_intention(action, obs, battery_warn_v=10.8)
        assert result_default == 6  # charge
        # With threshold = 9.0 → 9.5V does NOT trigger charge
        result_low = label_intention(action, obs, battery_warn_v=9.0)
        assert result_low != 6

    def test_custom_obstacle_clearance_m(self) -> None:
        """With a larger clearance, an obstacle at 0.4m triggers avoid_obstacle."""
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        obs = _obs(distance_m=0.4)
        result = label_intention(action, obs, obstacle_clearance_m=0.5)
        assert result == 2  # avoid_obstacle

    def test_custom_obstacle_clearance_m_miss(self) -> None:
        """With a smaller clearance, same 0.4m obstacle does NOT avoid."""
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        obs = _obs(distance_m=0.4)
        result = label_intention(action, obs, obstacle_clearance_m=0.2)
        assert result != 2
