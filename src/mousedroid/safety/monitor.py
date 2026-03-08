"""MouseDroid safety monitor — evaluates observations for hazardous conditions.

Implements :class:`~mousedroid.safety.protocol.SafetyMonitorProtocol`.
Each control-loop tick the monitor produces a frozen
:class:`~mousedroid.safety.context.SafetyContext` that agents consume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mousedroid.logging.setup import get_logger
from mousedroid.safety.context import SafetyContext

if TYPE_CHECKING:
    from mousedroid.config.schema import SafetyConfig
    from mousedroid.sensing.protocol import ObservationProtocol

_log = get_logger(__name__)

_BATTERY_WARN_V: float = 10.5
"""Fallback battery warning voltage when config is unavailable."""

_BATTERY_CRITICAL_V: float = 9.5
"""Fallback battery critical voltage when config is unavailable."""

_MAX_LOOP_TIME_MS: float = 200.0
"""Loop iteration time above which an emergency is triggered."""


class MouseDroidSafetyMonitor:
    """Evaluates each observation for hazardous conditions.

    Checks forward clearance, battery voltage, sensor validity, and
    loop timing.  Any critical condition sets ``is_emergency=True`` on
    the returned :class:`SafetyContext`.

    Implements :class:`~mousedroid.safety.protocol.SafetyMonitorProtocol`.

    Args:
        cfg: Safety configuration thresholds.
    """

    def __init__(self, cfg: SafetyConfig) -> None:
        self._cfg = cfg

    # -- SafetyMonitorProtocol ---------------------------------------------

    def evaluate(
        self,
        observation: ObservationProtocol,
        loop_time_ms: float,
    ) -> SafetyContext:
        """Evaluate safety state from the current observation.

        Args:
            observation: Fused sensor bundle for this tick.
            loop_time_ms: Wall-clock duration of the last loop iteration
                in milliseconds.

        Returns:
            A frozen :class:`SafetyContext` with all fields populated.
        """
        is_emergency = False

        # -- Forward clearance ---------------------------------------------
        forward_clearance_ok = observation.distance_m >= self._cfg.min_forward_clearance_m
        if not forward_clearance_ok:
            _log.warning(
                "forward_clearance_violation",
                distance_m=observation.distance_m,
                threshold_m=self._cfg.min_forward_clearance_m,
            )
            is_emergency = True

        # -- Battery voltage -----------------------------------------------
        battery_voltage: float = float(observation.motor_state[3])
        battery_warn_v = getattr(self._cfg, "battery_warn_v", _BATTERY_WARN_V)
        battery_critical_v = getattr(self._cfg, "battery_critical_v", _BATTERY_CRITICAL_V)

        if battery_voltage < battery_critical_v:
            _log.error(
                "battery_critical",
                voltage=battery_voltage,
                threshold=battery_critical_v,
            )
            is_emergency = True
        elif battery_voltage < battery_warn_v:
            _log.warning(
                "battery_low",
                voltage=battery_voltage,
                threshold=battery_warn_v,
            )

        # -- Valid sensor count --------------------------------------------
        valid_sensor_count = int(np.sum(observation.valid_mask > 0.0))
        if valid_sensor_count < self._cfg.min_valid_sensors:
            _log.error(
                "insufficient_valid_sensors",
                valid=valid_sensor_count,
                required=self._cfg.min_valid_sensors,
            )
            is_emergency = True

        # -- Loop timing ---------------------------------------------------
        if loop_time_ms > _MAX_LOOP_TIME_MS:
            _log.error(
                "loop_overrun",
                loop_time_ms=loop_time_ms,
                max_ms=_MAX_LOOP_TIME_MS,
            )
            is_emergency = True

        return SafetyContext(
            ultrasonic_dist_m=observation.distance_m,
            forward_clearance_ok=forward_clearance_ok,
            battery_voltage=battery_voltage,
            valid_sensor_count=valid_sensor_count,
            loop_time_ms=loop_time_ms,
            is_emergency=is_emergency,
        )
