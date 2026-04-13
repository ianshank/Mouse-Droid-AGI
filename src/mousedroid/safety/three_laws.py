"""Three Laws of Robotics — hierarchical safety constraint layer.

Implements Asimov's Three Laws with strict priority ordering:
  1. A robot may not injure a human being or, through inaction, allow
     a human being to come to harm.
  2. A robot must obey the orders given it by human beings except where
     such orders would conflict with the First Law.
  3. A robot must protect its own existence as long as such protection
     does not conflict with the First or Second Law.

All computation is numpy-only. The checker clips/overrides actions that
violate any law before they reach actuators.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ThreeLawsConfig

_log = get_logger(__name__)


def _default_three_laws_config() -> ThreeLawsConfig:
    """Lazily create a default ThreeLawsConfig for constructor defaults.

    Returns:
        A ``ThreeLawsConfig`` instance with all Pydantic defaults applied.
    """
    from mousedroid.config.schema import ThreeLawsConfig as _Cfg

    return _Cfg()  # type: ignore[call-arg]  # Pydantic fields have defaults


_DEFAULT_CFG = _default_three_laws_config()


class RoboticsLaw(enum.IntEnum):
    """Asimov's Three Laws, ordered by priority."""

    FIRST = 1  # No harm to humans
    SECOND = 2  # Obey human commands
    THIRD = 3  # Self-preservation


@dataclass(frozen=True)
class LawViolation:
    """A single violation of a Robotics Law.

    Attributes:
        law: Which law was violated.
        description: Human-readable description.
        severity: Violation severity in ``[0.0, 1.0]``.
        action_override: Corrective safe action, if any.
    """

    law: RoboticsLaw
    description: str
    severity: float
    action_override: NDArray[np.floating[Any]] | None = None


class RoboticsLawChecker:
    """Check and enforce the Three Laws of Robotics on every action.

    Checks are evaluated in **strict priority order**: Law 1, then Law 2,
    then Law 3. A higher-priority law's override always takes precedence.

    Args:
        human_safety_radius_m: Law 1 — minimum distance to humans.
        emergency_stop_dist_m: Law 1 — emergency stop obstacle distance.
        max_safe_acceleration_mps2: Law 1 — max acceleration magnitude.
        command_blend_weight: Law 2 — weight for blending toward commands.
        battery_preservation_v: Law 3 — battery voltage preservation threshold.
        thermal_critical_c: Law 3 — thermal preservation threshold.
        enabled: Whether the checker is active.
    """

    def __init__(
        self,
        *,
        human_safety_radius_m: float = _DEFAULT_CFG.human_safety_radius_m,
        emergency_stop_dist_m: float = _DEFAULT_CFG.emergency_stop_dist_m,
        max_safe_acceleration_mps2: float = _DEFAULT_CFG.max_safe_acceleration_mps2,
        idle_speed_threshold: float = _DEFAULT_CFG.idle_speed_threshold,
        alert_signal_speed: float = _DEFAULT_CFG.alert_signal_speed,
        command_blend_weight: float = _DEFAULT_CFG.command_blend_weight,
        battery_preservation_v: float = _DEFAULT_CFG.battery_preservation_v,
        thermal_critical_c: float = _DEFAULT_CFG.thermal_critical_c,
        smoothing_factor: float = _DEFAULT_CFG.smoothing_factor,
        enabled: bool = _DEFAULT_CFG.enabled,
        command_diff_threshold: float = _DEFAULT_CFG.command_diff_threshold,
        thermal_severity_range_c: float = _DEFAULT_CFG.thermal_severity_range_c,
        rapid_reversal_threshold: float = _DEFAULT_CFG.rapid_reversal_threshold,
        inaction_harm_severity: float = _DEFAULT_CFG.inaction_harm_severity,
        law1_override_severity: float = _DEFAULT_CFG.law1_override_severity,
        zone_boundary_severity: float = _DEFAULT_CFG.zone_boundary_severity,
        mechanical_stress_severity: float = _DEFAULT_CFG.mechanical_stress_severity,
        battery_damping_factor: float = _DEFAULT_CFG.battery_damping_factor,
        thermal_damping_factor: float = _DEFAULT_CFG.thermal_damping_factor,
    ) -> None:
        self._human_safety_radius_m = human_safety_radius_m
        self._emergency_stop_dist_m = emergency_stop_dist_m
        self._max_safe_acceleration_mps2 = max_safe_acceleration_mps2
        self._idle_speed_threshold = idle_speed_threshold
        self._alert_signal_speed = alert_signal_speed
        self._command_blend_weight = command_blend_weight
        self._battery_preservation_v = battery_preservation_v
        self._thermal_critical_c = thermal_critical_c
        self._smoothing_factor = smoothing_factor
        self._enabled = enabled
        self._command_diff_threshold = command_diff_threshold
        self._thermal_severity_range_c = thermal_severity_range_c
        self._rapid_reversal_threshold = rapid_reversal_threshold
        self._inaction_harm_severity = inaction_harm_severity
        self._law1_override_severity = law1_override_severity
        self._zone_boundary_severity = zone_boundary_severity
        self._mechanical_stress_severity = mechanical_stress_severity
        self._battery_damping_factor = battery_damping_factor
        self._thermal_damping_factor = thermal_damping_factor

    @classmethod
    def from_config(cls, cfg: ThreeLawsConfig) -> RoboticsLawChecker:
        """Build from a ``ThreeLawsConfig`` pydantic model.

        Args:
            cfg: ThreeLawsConfig instance.

        Returns:
            Configured checker.
        """
        return cls(
            human_safety_radius_m=cfg.human_safety_radius_m,
            emergency_stop_dist_m=cfg.emergency_stop_dist_m,
            max_safe_acceleration_mps2=cfg.max_safe_acceleration_mps2,
            idle_speed_threshold=cfg.idle_speed_threshold,
            alert_signal_speed=cfg.alert_signal_speed,
            command_blend_weight=cfg.command_blend_weight,
            battery_preservation_v=cfg.battery_preservation_v,
            thermal_critical_c=cfg.thermal_critical_c,
            smoothing_factor=cfg.smoothing_factor,
            enabled=cfg.enabled,
            command_diff_threshold=cfg.command_diff_threshold,
            thermal_severity_range_c=cfg.thermal_severity_range_c,
            rapid_reversal_threshold=cfg.rapid_reversal_threshold,
            inaction_harm_severity=cfg.inaction_harm_severity,
            law1_override_severity=cfg.law1_override_severity,
            zone_boundary_severity=cfg.zone_boundary_severity,
            mechanical_stress_severity=cfg.mechanical_stress_severity,
            battery_damping_factor=cfg.battery_damping_factor,
            thermal_damping_factor=cfg.thermal_damping_factor,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        action: NDArray[np.floating[Any]],
        context: dict[str, Any],
    ) -> tuple[NDArray[np.floating[Any]], list[LawViolation]]:
        """Validate *action* against the Three Laws of Robotics.

        Args:
            action: Raw action vector (at minimum ``[speed, ...]``).
            context: Environment context with optional keys:
                ``human_detected``, ``human_dist_m``, ``human_needs_help``,
                ``obstacle_dist_m``, ``prev_action``, ``commanded_action``,
                ``allowed_zone_min``, ``allowed_zone_max``,
                ``battery_v``, ``gpu_temp_c``.

        Returns:
            Tuple of ``(safe_action, violations)`` where *violations* is
            a list of :class:`LawViolation` sorted by law priority.
        """
        if not self._enabled:
            _log.debug("three_laws_disabled")
            return action.copy().astype(np.float64), []

        safe: NDArray[np.floating[Any]] = action.copy().astype(np.float64)
        violations: list[LawViolation] = []

        _log.debug("three_laws_check_start", action_norm=float(np.linalg.norm(action)))

        # --- Law 1: No Harm (highest priority) ---
        safe, v1 = self._check_law1(safe, context)
        violations.extend(v1)

        # --- Law 2: Obedience ---
        safe, v2 = self._check_law2(safe, context, has_law1_violation=bool(v1))
        violations.extend(v2)

        # --- Law 3: Self-Preservation ---
        safe, v3 = self._check_law3(safe, context, has_higher_violation=bool(v1 or v2))
        violations.extend(v3)

        if violations:
            _log.warning(
                "three_laws_violations",
                violations=[
                    {"law": v.law.value, "desc": v.description, "severity": v.severity}
                    for v in violations
                ],
            )

        return safe, violations

    # ------------------------------------------------------------------
    # Law 1: No Harm
    # ------------------------------------------------------------------

    def _check_law1(
        self,
        action: NDArray[np.floating[Any]],
        context: dict[str, Any],
    ) -> tuple[NDArray[np.floating[Any]], list[LawViolation]]:
        safe = action.copy()
        violations: list[LawViolation] = []

        # Human proximity check
        human_detected = bool(context.get("human_detected", False))
        human_dist_m = float(context.get("human_dist_m", float("inf")))

        if human_detected and human_dist_m < self._human_safety_radius_m:
            speed = float(np.abs(safe[0])) if safe.size > 0 else 0.0
            if speed > self._idle_speed_threshold:
                severity = 1.0 - (human_dist_m / self._human_safety_radius_m)
                severity = float(np.clip(severity, 0.0, 1.0))
                violations.append(
                    LawViolation(
                        law=RoboticsLaw.FIRST,
                        description=(
                            f"human at {human_dist_m:.2f}m < safety radius "
                            f"{self._human_safety_radius_m:.2f}m"
                        ),
                        severity=severity,
                        action_override=np.zeros_like(safe),
                    )
                )
                safe[:] = 0.0  # Full stop

        # Collision trajectory check
        obstacle_dist = float(context.get("obstacle_dist_m", float("inf")))
        if safe.size > 0 and float(safe[0]) > 0 and obstacle_dist < self._emergency_stop_dist_m:
            severity = 1.0 - (obstacle_dist / self._emergency_stop_dist_m)
            severity = float(np.clip(severity, 0.0, 1.0))
            violations.append(
                LawViolation(
                    law=RoboticsLaw.FIRST,
                    description=(
                        f"obstacle at {obstacle_dist:.2f}m < emergency stop "
                        f"{self._emergency_stop_dist_m:.2f}m"
                    ),
                    severity=severity,
                )
            )
            safe[0] = 0.0  # Zero forward velocity

        # Harmful acceleration check
        prev_action = context.get("prev_action")
        if prev_action is not None:
            prev = np.asarray(prev_action, dtype=np.float64)
            accel = np.abs(safe - prev)
            max_accel = float(np.max(accel))
            if max_accel > self._max_safe_acceleration_mps2:
                max_safe = self._max_safe_acceleration_mps2
                severity = float(
                    np.clip(
                        (max_accel - max_safe) / max_safe,
                        0.0,
                        1.0,
                    )
                )
                violations.append(
                    LawViolation(
                        law=RoboticsLaw.FIRST,
                        description=(
                            f"acceleration {max_accel:.2f} > max safe "
                            f"{self._max_safe_acceleration_mps2:.2f} m/s²"
                        ),
                        severity=severity,
                    )
                )
                # Clamp acceleration
                direction = safe - prev
                scale = self._max_safe_acceleration_mps2 / (max_accel + 1e-8)
                safe = prev + direction * scale

        # Inaction harm check: human needs help but robot is idle
        human_needs_help = bool(context.get("human_needs_help", False))
        if human_needs_help:
            speed = float(np.linalg.norm(safe)) if safe.size > 0 else 0.0
            if speed < self._idle_speed_threshold:
                violations.append(
                    LawViolation(
                        law=RoboticsLaw.FIRST,
                        description="inaction while human needs help",
                        severity=self._inaction_harm_severity,
                        action_override=np.array(
                            [self._alert_signal_speed] + [0.0] * (safe.size - 1),
                            dtype=np.float64,
                        )
                        if safe.size > 0
                        else None,
                    )
                )
                if safe.size > 0:
                    safe[0] = self._alert_signal_speed  # Small alert signal

        return safe, violations

    # ------------------------------------------------------------------
    # Law 2: Obedience
    # ------------------------------------------------------------------

    def _check_law2(
        self,
        action: NDArray[np.floating[Any]],
        context: dict[str, Any],
        *,
        has_law1_violation: bool,
    ) -> tuple[NDArray[np.floating[Any]], list[LawViolation]]:
        safe = action.copy()
        violations: list[LawViolation] = []

        # Command compliance
        commanded = context.get("commanded_action")
        if commanded is not None:
            cmd = np.asarray(commanded, dtype=np.float64)
            if has_law1_violation:
                # Law 1 override: do NOT follow command
                diff = float(np.linalg.norm(safe - cmd))
                if diff > self._command_diff_threshold:
                    violations.append(
                        LawViolation(
                            law=RoboticsLaw.SECOND,
                            description="command overridden by Law 1 (no harm)",
                            severity=self._law1_override_severity,
                        )
                    )
            else:
                # Blend toward commanded action
                w = self._command_blend_weight
                blended = w * cmd + (1.0 - w) * safe
                diff = float(np.linalg.norm(safe - cmd))
                if diff > self._command_diff_threshold:
                    violations.append(
                        LawViolation(
                            law=RoboticsLaw.SECOND,
                            description=f"blending toward command (diff={diff:.3f})",
                            severity=float(np.clip(diff, 0.0, 1.0)),
                        )
                    )
                safe = blended

        # Boundary compliance
        zone_min = context.get("allowed_zone_min")
        zone_max = context.get("allowed_zone_max")
        if zone_min is not None and zone_max is not None and not has_law1_violation:
            z_min = np.asarray(zone_min, dtype=np.float64)
            z_max = np.asarray(zone_max, dtype=np.float64)
            clipped = np.clip(safe, z_min, z_max)
            if not np.allclose(safe, clipped):
                violations.append(
                    LawViolation(
                        law=RoboticsLaw.SECOND,
                        description="action clipped to allowed zone boundary",
                        severity=self._zone_boundary_severity,
                    )
                )
                safe = clipped

        return safe, violations

    # ------------------------------------------------------------------
    # Law 3: Self-Preservation
    # ------------------------------------------------------------------

    def _check_law3(
        self,
        action: NDArray[np.floating[Any]],
        context: dict[str, Any],
        *,
        has_higher_violation: bool,
    ) -> tuple[NDArray[np.floating[Any]], list[LawViolation]]:
        safe = action.copy()
        violations: list[LawViolation] = []

        # Battery preservation
        battery_v = float(context.get("battery_v", self._battery_preservation_v + 1.0))
        if battery_v < self._battery_preservation_v and not has_higher_violation:
            severity = float(
                np.clip(
                    1.0 - (battery_v / self._battery_preservation_v),
                    0.0,
                    1.0,
                )
            )
            violations.append(
                LawViolation(
                    law=RoboticsLaw.THIRD,
                    description=(
                        f"battery {battery_v:.2f}V < preservation "
                        f"{self._battery_preservation_v:.2f}V"
                    ),
                    severity=severity,
                )
            )
            # Reduce motion to conserve battery
            safe *= self._battery_damping_factor

        # Thermal preservation
        gpu_temp = float(context.get("gpu_temp_c", 0.0))
        if gpu_temp > self._thermal_critical_c and not has_higher_violation:
            # Severity scales linearly: 0.0 at critical, 1.0 at critical+range
            severity = float(
                np.clip(
                    (gpu_temp - self._thermal_critical_c) / self._thermal_severity_range_c,
                    0.0,
                    1.0,
                )
            )
            violations.append(
                LawViolation(
                    law=RoboticsLaw.THIRD,
                    description=(
                        f"GPU temp {gpu_temp:.1f}°C > critical {self._thermal_critical_c:.1f}°C"
                    ),
                    severity=severity,
                )
            )
            safe *= self._thermal_damping_factor

        # Mechanical stress: smooth rapid direction reversals
        prev_action = context.get("prev_action")
        if prev_action is not None and not has_higher_violation:
            prev = np.asarray(prev_action, dtype=np.float64)
            # Check if sign reversal on any axis
            sign_change = np.sign(safe) != np.sign(prev)
            magnitude_change = np.abs(safe - prev)
            rapid_reversal = sign_change & (magnitude_change > self._rapid_reversal_threshold)
            if np.any(rapid_reversal):
                violations.append(
                    LawViolation(
                        law=RoboticsLaw.THIRD,
                        description="rapid direction reversal smoothed",
                        severity=self._mechanical_stress_severity,
                    )
                )
                # Smooth the transition
                safe = np.where(
                    rapid_reversal,
                    prev + (safe - prev) * self._smoothing_factor,
                    safe,
                )

        return safe, violations
