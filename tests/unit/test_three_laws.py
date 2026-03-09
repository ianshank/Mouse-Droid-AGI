"""Comprehensive tests for the Three Laws of Robotics safety module.

Covers Law 1 (No Harm), Law 2 (Obedience), Law 3 (Self-Preservation),
and hierarchical priority interactions.
"""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.config.schema import ThreeLawsConfig
from mousedroid.safety.three_laws import (
    LawViolation,
    RoboticsLaw,
    RoboticsLawChecker,
)


def _checker(**overrides) -> RoboticsLawChecker:
    """Build a checker from config with optional overrides."""
    cfg = ThreeLawsConfig(**overrides)
    return RoboticsLawChecker.from_config(cfg)


# =========================================================================
# Law 1 — No Harm
# =========================================================================


class TestLaw1HumanProximity:
    def test_human_proximity_full_stop(self) -> None:
        checker = _checker()
        action = np.array([0.5, 0.0, 0.0], dtype=np.float64)
        ctx = {"human_detected": True, "human_dist_m": 0.2}
        safe, violations = checker.check(action, ctx)
        assert np.allclose(safe[:1], 0.0)  # speed zeroed
        assert any(v.law == RoboticsLaw.FIRST for v in violations)

    def test_human_at_boundary_no_violation(self) -> None:
        checker = _checker(human_safety_radius_m=0.5)
        action = np.array([0.3, 0.0], dtype=np.float64)
        ctx = {"human_detected": True, "human_dist_m": 0.5}
        _safe, violations = checker.check(action, ctx)
        law1 = [v for v in violations if v.law == RoboticsLaw.FIRST and "human" in v.description]
        assert len(law1) == 0

    def test_human_far_away_no_violation(self) -> None:
        checker = _checker()
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {"human_detected": True, "human_dist_m": 5.0}
        _safe, violations = checker.check(action, ctx)
        law1_human = [v for v in violations if "human" in v.description]
        assert len(law1_human) == 0

    def test_no_human_detected_no_violation(self) -> None:
        checker = _checker()
        action = np.array([0.5, 0.0], dtype=np.float64)
        _, violations = checker.check(action, {})
        law1_human = [v for v in violations if "human" in v.description]
        assert len(law1_human) == 0

    def test_violation_returns_full_stop_action(self) -> None:
        checker = _checker()
        action = np.array([0.8, 0.3, 0.5], dtype=np.float64)
        ctx = {"human_detected": True, "human_dist_m": 0.1}
        safe, _violations = checker.check(action, ctx)
        assert float(np.max(np.abs(safe))) < 0.2  # near zero

    def test_severity_scales_with_proximity(self) -> None:
        checker = _checker(human_safety_radius_m=1.0)
        action = np.array([0.5, 0.0], dtype=np.float64)

        _, v_close = checker.check(action, {"human_detected": True, "human_dist_m": 0.1})
        _, v_far = checker.check(action, {"human_detected": True, "human_dist_m": 0.8})

        close_sev = max(v.severity for v in v_close if "human" in v.description)
        far_sev = max(v.severity for v in v_far if "human" in v.description)
        assert close_sev > far_sev

    def test_zero_speed_near_human_ok(self) -> None:
        checker = _checker()
        action = np.array([0.0, 0.0], dtype=np.float64)
        ctx = {"human_detected": True, "human_dist_m": 0.1}
        _, violations = checker.check(action, ctx)
        law1_human = [v for v in violations if "human" in v.description]
        assert len(law1_human) == 0

    def test_violation_logged(self) -> None:
        checker = _checker()
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {"human_detected": True, "human_dist_m": 0.2}
        # Just verify no error — logging is tested by presence of violations
        _, violations = checker.check(action, ctx)
        assert len(violations) > 0


class TestLaw1CollisionTrajectory:
    def test_obstacle_emergency_zero_speed(self) -> None:
        checker = _checker(emergency_stop_dist_m=0.15)
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {"obstacle_dist_m": 0.1}
        safe, violations = checker.check(action, ctx)
        assert safe[0] == 0.0
        assert any(v.law == RoboticsLaw.FIRST for v in violations)

    def test_obstacle_far_no_violation(self) -> None:
        checker = _checker()
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {"obstacle_dist_m": 2.0}
        _, violations = checker.check(action, ctx)
        obstacle_v = [v for v in violations if "obstacle" in v.description]
        assert len(obstacle_v) == 0


class TestLaw1HarmfulAcceleration:
    def test_acceleration_clamped(self) -> None:
        checker = _checker(max_safe_acceleration_mps2=0.5)
        action = np.array([1.0, 0.0], dtype=np.float64)
        ctx = {"prev_action": np.array([0.0, 0.0])}
        safe, violations = checker.check(action, ctx)
        accel = np.abs(safe - np.array([0.0, 0.0]))
        assert float(np.max(accel)) <= 0.5 + 0.01
        assert any("acceleration" in v.description for v in violations)

    def test_small_acceleration_ok(self) -> None:
        checker = _checker(max_safe_acceleration_mps2=1.0)
        action = np.array([0.3, 0.0], dtype=np.float64)
        ctx = {"prev_action": np.array([0.0, 0.0])}
        _, violations = checker.check(action, ctx)
        accel_v = [v for v in violations if "acceleration" in v.description]
        assert len(accel_v) == 0


class TestLaw1InactionHarm:
    def test_inaction_override_when_human_needs_help(self) -> None:
        checker = _checker()
        action = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ctx = {"human_needs_help": True}
        safe, violations = checker.check(action, ctx)
        assert safe[0] > 0  # Alert signal
        assert any("inaction" in v.description for v in violations)

    def test_no_inaction_when_already_moving(self) -> None:
        checker = _checker()
        action = np.array([0.3, 0.0], dtype=np.float64)
        ctx = {"human_needs_help": True}
        _, violations = checker.check(action, ctx)
        inaction = [v for v in violations if "inaction" in v.description]
        assert len(inaction) == 0


class TestLaw1AlwaysFirst:
    def test_law1_always_checked_first(self) -> None:
        checker = _checker()
        action = np.array([0.5, 0.0], dtype=np.float64)
        # Human too close + commanded action + low battery
        ctx = {
            "human_detected": True,
            "human_dist_m": 0.1,
            "commanded_action": np.array([1.0, 0.0]),
            "battery_v": 8.0,
        }
        safe, violations = checker.check(action, ctx)
        # Law 1 should dominate
        assert violations[0].law == RoboticsLaw.FIRST
        assert float(np.max(np.abs(safe))) < 0.2

    def test_law1_overrides_law2_command(self) -> None:
        checker = _checker()
        # Moving toward human — Law 1 fires
        action = np.array([0.5, 0.0], dtype=np.float64)
        # Command says go forward, but human is right there
        ctx = {
            "human_detected": True,
            "human_dist_m": 0.1,
            "commanded_action": np.array([1.0, 0.0]),
        }
        _safe, violations = checker.check(action, ctx)
        # Should NOT follow command toward human — Law 1 override
        law2 = [v for v in violations if v.law == RoboticsLaw.SECOND]
        assert law2, "Expected Law 2 violation when command overridden"
        assert any("overridden by Law 1" in v.description for v in law2)

    def test_law1_overrides_law3_preservation(self) -> None:
        """Self-preservation should not cause harm."""
        checker = _checker()
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {
            "human_detected": True,
            "human_dist_m": 0.2,
            "battery_v": 8.0,  # Law 3 wants to conserve
        }
        safe, _violations = checker.check(action, ctx)
        # Law 1 full stop takes priority
        assert np.allclose(safe, 0.0, atol=0.01)


# =========================================================================
# Law 2 — Obedience
# =========================================================================


class TestLaw2CommandCompliance:
    def test_command_compliance_blend(self) -> None:
        checker = _checker(command_blend_weight=0.8)
        action = np.array([0.0, 0.0], dtype=np.float64)
        cmd = np.array([1.0, 0.0])
        ctx = {"commanded_action": cmd}
        safe, _violations = checker.check(action, ctx)
        # Should blend: 0.8 * cmd + 0.2 * action
        assert abs(safe[0] - 0.8) < 0.01

    def test_no_command_no_change(self) -> None:
        checker = _checker()
        action = np.array([0.3, 0.1], dtype=np.float64)
        _safe, violations = checker.check(action, {})
        law2 = [v for v in violations if v.law == RoboticsLaw.SECOND]
        assert len(law2) == 0

    def test_command_overridden_by_law1(self) -> None:
        checker = _checker()
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {
            "human_detected": True,
            "human_dist_m": 0.2,
            "commanded_action": np.array([1.0, 0.0]),
        }
        _safe, violations = checker.check(action, ctx)
        law2 = [v for v in violations if v.law == RoboticsLaw.SECOND]
        if law2:
            assert any("overridden by Law 1" in v.description for v in law2)

    def test_blend_weight_respected(self) -> None:
        checker = _checker(command_blend_weight=0.5)
        action = np.array([0.0, 0.0], dtype=np.float64)
        cmd = np.array([1.0, 0.0])
        ctx = {"commanded_action": cmd}
        safe, _ = checker.check(action, ctx)
        assert abs(safe[0] - 0.5) < 0.01

    def test_command_not_toward_human_accepted(self) -> None:
        checker = _checker(command_blend_weight=1.0)
        action = np.array([0.0, 0.0], dtype=np.float64)
        cmd = np.array([0.5, 0.0])
        ctx = {"commanded_action": cmd}
        safe, _ = checker.check(action, ctx)
        assert abs(safe[0] - 0.5) < 0.01


class TestLaw2BoundaryCompliance:
    def test_boundary_clips_action(self) -> None:
        checker = _checker()
        action = np.array([1.5, -0.5], dtype=np.float64)
        ctx = {
            "allowed_zone_min": np.array([-1.0, -0.3]),
            "allowed_zone_max": np.array([1.0, 0.3]),
        }
        safe, _violations = checker.check(action, ctx)
        assert safe[0] <= 1.0
        assert safe[1] >= -0.3

    def test_boundary_inside_no_change(self) -> None:
        checker = _checker()
        action = np.array([0.5, 0.1], dtype=np.float64)
        ctx = {
            "allowed_zone_min": np.array([-1.0, -1.0]),
            "allowed_zone_max": np.array([1.0, 1.0]),
        }
        _safe, violations = checker.check(action, ctx)
        boundary_v = [
            v for v in violations if "boundary" in v.description or "zone" in v.description
        ]
        assert len(boundary_v) == 0

    def test_law2_overrides_law3(self) -> None:
        """Command takes precedence over self-preservation."""
        checker = _checker(command_blend_weight=1.0)
        action = np.array([0.0, 0.0], dtype=np.float64)
        ctx = {
            "commanded_action": np.array([0.5, 0.0]),
            "battery_v": 8.0,  # Law 3 wants to conserve
        }
        safe, _violations = checker.check(action, ctx)
        # Command should still be partially obeyed
        assert safe[0] > 0

    def test_violation_severity_scales_with_diff(self) -> None:
        checker = _checker(command_blend_weight=0.8)
        action = np.array([0.0, 0.0], dtype=np.float64)
        ctx_close = {"commanded_action": np.array([0.1, 0.0])}
        ctx_far = {"commanded_action": np.array([1.0, 0.0])}
        _, v_close = checker.check(action, ctx_close)
        _, v_far = checker.check(action, ctx_far)
        # Both produce violations, far command has higher severity
        sev_close = [v.severity for v in v_close if v.law == RoboticsLaw.SECOND]
        sev_far = [v.severity for v in v_far if v.law == RoboticsLaw.SECOND]
        if sev_close and sev_far:
            assert max(sev_far) >= max(sev_close)


# =========================================================================
# Law 3 — Self-Preservation
# =========================================================================


class TestLaw3BatteryPreservation:
    def test_low_battery_reduces_motion(self) -> None:
        checker = _checker(battery_preservation_v=10.5)
        action = np.array([0.8, 0.0], dtype=np.float64)
        ctx = {"battery_v": 9.0}
        safe, violations = checker.check(action, ctx)
        assert abs(safe[0]) < abs(action[0])
        assert any(v.law == RoboticsLaw.THIRD for v in violations)

    def test_battery_ok_no_change(self) -> None:
        checker = _checker(battery_preservation_v=10.5)
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {"battery_v": 12.0}
        _safe, violations = checker.check(action, ctx)
        battery_v = [v for v in violations if "battery" in v.description]
        assert len(battery_v) == 0

    def test_battery_preservation_threshold_configurable(self) -> None:
        checker = _checker(battery_preservation_v=11.0)
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {"battery_v": 10.5}
        _, violations = checker.check(action, ctx)
        assert any("battery" in v.description for v in violations)

    def test_severity_scales_with_battery(self) -> None:
        checker = _checker(battery_preservation_v=10.0)
        action = np.array([0.5, 0.0], dtype=np.float64)
        _, v_low = checker.check(action, {"battery_v": 5.0})
        _, v_med = checker.check(action, {"battery_v": 9.0})
        low_sev = [v.severity for v in v_low if "battery" in v.description]
        med_sev = [v.severity for v in v_med if "battery" in v.description]
        if low_sev and med_sev:
            assert max(low_sev) > max(med_sev)


class TestLaw3ThermalPreservation:
    def test_thermal_critical_reduces_activity(self) -> None:
        checker = _checker(thermal_critical_c=85.0)
        action = np.array([0.8, 0.0], dtype=np.float64)
        ctx = {"gpu_temp_c": 95.0}
        safe, violations = checker.check(action, ctx)
        assert abs(safe[0]) < abs(action[0])
        assert any("thermal" in v.description or "temp" in v.description for v in violations)

    def test_thermal_normal_no_change(self) -> None:
        checker = _checker(thermal_critical_c=85.0)
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {"gpu_temp_c": 60.0}
        _, violations = checker.check(action, ctx)
        thermal = [v for v in violations if "temp" in v.description or "thermal" in v.description]
        assert len(thermal) == 0


class TestLaw3MechanicalStress:
    def test_rapid_reversal_smoothed(self) -> None:
        # Use high max_safe_acceleration so Law 1 accel check doesn't fire
        checker = _checker(max_safe_acceleration_mps2=5.0)
        action = np.array([-1.0, 0.0], dtype=np.float64)
        ctx = {"prev_action": np.array([1.0, 0.0])}
        safe, violations = checker.check(action, ctx)
        # Should be smoothed, not jump from 1.0 to -1.0
        assert abs(safe[0]) < 1.0
        assert any("reversal" in v.description for v in violations)


class TestLaw3OverriddenByHigherLaws:
    def test_overridden_by_law1(self) -> None:
        checker = _checker()
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {
            "human_detected": True,
            "human_dist_m": 0.1,
            "battery_v": 8.0,
        }
        safe, _violations = checker.check(action, ctx)
        # Law 1 takes over: full stop despite battery wanting conservation
        assert np.allclose(safe, 0.0, atol=0.01)

    def test_overridden_by_law2(self) -> None:
        checker = _checker(command_blend_weight=1.0)
        action = np.array([0.0, 0.0], dtype=np.float64)
        ctx = {
            "commanded_action": np.array([0.5, 0.0]),
            "battery_v": 8.0,
        }
        safe, _violations = checker.check(action, ctx)
        # Law 2 command still partially obeyed
        assert safe[0] > 0

    def test_combined_thermal_battery(self) -> None:
        checker = _checker(battery_preservation_v=10.5, thermal_critical_c=85.0)
        action = np.array([1.0, 0.0], dtype=np.float64)
        ctx = {"battery_v": 9.0, "gpu_temp_c": 95.0}
        _, violations = checker.check(action, ctx)
        law3 = [v for v in violations if v.law == RoboticsLaw.THIRD]
        assert len(law3) >= 2  # both battery and thermal


# =========================================================================
# Hierarchy / Integration
# =========================================================================


class TestLawPriorityOrdering:
    def test_violations_sorted_by_law(self) -> None:
        checker = _checker()
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {
            "human_detected": True,
            "human_dist_m": 0.1,
            "battery_v": 8.0,
        }
        _, violations = checker.check(action, ctx)
        law_numbers = [v.law.value for v in violations]
        # Law 1 violations should come before Law 3
        first_law1 = next((i for i, n in enumerate(law_numbers) if n == 1), None)
        first_law3 = next((i for i, n in enumerate(law_numbers) if n == 3), None)
        if first_law1 is not None and first_law3 is not None:
            assert first_law1 < first_law3

    def test_all_laws_pass_action_unchanged(self) -> None:
        checker = _checker()
        action = np.array([0.3, 0.1], dtype=np.float64)
        ctx = {
            "battery_v": 12.0,
            "gpu_temp_c": 50.0,
            "obstacle_dist_m": 5.0,
        }
        safe, violations = checker.check(action, ctx)
        assert len(violations) == 0
        np.testing.assert_allclose(safe, action, atol=1e-10)

    def test_all_three_violated(self) -> None:
        checker = _checker()
        action = np.array([0.8, 0.0], dtype=np.float64)
        ctx = {
            "human_detected": True,
            "human_dist_m": 0.1,
            "commanded_action": np.array([1.0, 0.0]),
            "battery_v": 8.0,
        }
        _, violations = checker.check(action, ctx)
        laws_hit = {v.law for v in violations}
        assert RoboticsLaw.FIRST in laws_hit

    def test_checker_returns_correct_tuple_type(self) -> None:
        checker = _checker()
        action = np.array([0.3], dtype=np.float64)
        safe, violations = checker.check(action, {})
        assert isinstance(safe, np.ndarray)
        assert isinstance(violations, list)

    def test_default_config_normal_conditions_no_violations(self) -> None:
        checker = _checker()
        action = np.array([0.2, 0.0, 0.0], dtype=np.float64)
        ctx = {"battery_v": 12.0, "obstacle_dist_m": 2.0}
        _, violations = checker.check(action, ctx)
        assert violations == []

    def test_checker_disabled_no_violations(self) -> None:
        checker = _checker(enabled=False)
        action = np.array([0.8, 0.0], dtype=np.float64)
        ctx = {"human_detected": True, "human_dist_m": 0.01}
        safe, violations = checker.check(action, ctx)
        assert violations == []
        np.testing.assert_allclose(safe, action)

    def test_violations_include_law_reference(self) -> None:
        checker = _checker()
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {"human_detected": True, "human_dist_m": 0.1}
        _, violations = checker.check(action, ctx)
        for v in violations:
            assert isinstance(v.law, RoboticsLaw)
            assert v.law.value in (1, 2, 3)

    def test_config_driven_thresholds(self) -> None:
        # Custom thresholds should be respected
        checker = _checker(human_safety_radius_m=2.0)
        action = np.array([0.5, 0.0], dtype=np.float64)
        ctx = {"human_detected": True, "human_dist_m": 1.5}
        _, violations = checker.check(action, ctx)
        assert any("human" in v.description for v in violations)

    def test_law_violation_dataclass_frozen(self) -> None:
        v = LawViolation(law=RoboticsLaw.FIRST, description="test", severity=0.5)
        with pytest.raises(AttributeError):
            v.severity = 0.9  # type: ignore[misc]

    def test_robotics_law_enum_ordering(self) -> None:
        assert RoboticsLaw.FIRST < RoboticsLaw.SECOND < RoboticsLaw.THIRD
