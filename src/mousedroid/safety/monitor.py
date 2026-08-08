"""MouseDroid safety monitor — evaluates observations for hazardous conditions.

Implements :class:`~mousedroid.safety.protocol.SafetyMonitorProtocol`.
Each control-loop tick the monitor produces a frozen
:class:`~mousedroid.safety.context.SafetyContext` that agents consume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mousedroid.constants import MOTOR_STATE_BATTERY_INDEX
from mousedroid.logging.setup import get_logger
from mousedroid.safety.context import SafetyContext

if TYPE_CHECKING:
    from mousedroid.config.schema import SafetyConfig
    from mousedroid.sensing.protocol import ObservationProtocol

_log = get_logger(__name__)


class MouseDroidSafetyMonitor:
    """Evaluates each observation for hazardous conditions.

    Checks forward clearance, battery voltage, sensor validity, sensor
    staleness, and loop timing.  Any critical condition sets
    ``is_emergency=True`` on the returned :class:`SafetyContext`.

    Implements :class:`~mousedroid.safety.protocol.SafetyMonitorProtocol`.

    Args:
        cfg: Safety configuration thresholds.
    """

    def __init__(self, cfg: SafetyConfig) -> None:
        self._cfg = cfg
        self._last_valid_timestamps: dict[int, float] = {}
        # Latch for the implausible-battery WARNING: a comms fault persists,
        # and this runs on every 30 Hz evaluation. Warn once per episode,
        # DEBUG thereafter; cleared by the next plausible reading.
        self._battery_missing_warned = False

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
        battery_voltage: float = float(observation.motor_state[MOTOR_STATE_BATTERY_INDEX])
        battery_warn_v = self._cfg.battery_warn_v
        battery_critical_v = self._cfg.battery_critical_v
        implausible_below_v = self._cfg.battery_implausible_below_v

        if implausible_below_v > 0 and battery_voltage < implausible_below_v:
            # Missing telemetry, NOT a flat pack. The comms layer reports an
            # unavailable reading as 0.0 V to keep the protocol signature
            # (see BaseESP32Driver.get_battery_voltage); treating that as
            # `battery_critical` would latch a permanent emergency stop on
            # every tick and send the operator to swap a healthy battery.
            #
            # A comms fault persists, so this branch is entered on EVERY
            # evaluation — at 30 Hz an unconditional WARNING buries the rest
            # of the log precisely when an operator is reading it. Warn once
            # per episode, DEBUG thereafter; the latch clears below when a
            # plausible reading returns, so a second fault warns again.
            # Mirrors BaseESP32Driver._warn_battery_unavailable.
            log_missing = _log.debug if self._battery_missing_warned else _log.warning
            self._battery_missing_warned = True
            log_missing(
                "battery_reading_implausible",
                voltage=battery_voltage,
                threshold=implausible_below_v,
                hint=(
                    "treating as missing telemetry, not a flat pack — check "
                    "esp32_battery_reading_unavailable / command-set + baud"
                ),
            )
        else:
            # A plausible reading ends the episode, so the next fault warns
            # again rather than being swallowed by a stale latch.
            self._battery_missing_warned = False
            if battery_critical_v > 0 and battery_voltage < battery_critical_v:
                _log.error(
                    "battery_critical",
                    voltage=battery_voltage,
                    threshold=battery_critical_v,
                )
                is_emergency = True
            elif battery_warn_v > 0 and battery_voltage < battery_warn_v:
                _log.warning(
                    "battery_low",
                    voltage=battery_voltage,
                    threshold=battery_warn_v,
                )

        # -- Sensor staleness ----------------------------------------------
        current_time = observation.timestamp

        for i in range(len(observation.valid_mask)):
            if observation.valid_mask[i] > 0.0:
                self._last_valid_timestamps[i] = current_time
            elif i in self._last_valid_timestamps:
                elapsed = current_time - self._last_valid_timestamps[i]
                if elapsed > self._cfg.sensor_stale_s:
                    _log.warning(
                        "sensor_stale",
                        sensor_index=i,
                        elapsed_s=round(elapsed, 3),
                        threshold_s=self._cfg.sensor_stale_s,
                    )
                    is_emergency = True

        # -- Valid sensor count (uses original mask; staleness is an additional emergency trigger)
        valid_sensor_count = int(np.sum(observation.valid_mask > 0.0))
        if valid_sensor_count < self._cfg.min_valid_sensors:
            _log.error(
                "insufficient_valid_sensors",
                valid=valid_sensor_count,
                required=self._cfg.min_valid_sensors,
            )
            is_emergency = True

        # -- Loop timing ---------------------------------------------------
        max_loop_time_ms = self._cfg.max_loop_time_ms
        if loop_time_ms > max_loop_time_ms:
            _log.error(
                "loop_overrun",
                loop_time_ms=loop_time_ms,
                max_ms=max_loop_time_ms,
            )
            is_emergency = True

        # -- LiDAR 360-degree clearance ------------------------------------
        lidar_min_dist_m = float("inf")
        lidar_clearance_ok = True
        lidar_features = observation.lidar_features
        if lidar_features is not None and len(lidar_features) > 0:
            # Features are normalised distances (min_in_sector / max_range).
            # Convert to metres using the maximum observed feature range.
            lidar_max_range = self._cfg.lidar_max_range_m
            lidar_min_dist_m = float(np.min(lidar_features)) * lidar_max_range
            if lidar_min_dist_m < self._cfg.min_forward_clearance_m:
                lidar_clearance_ok = False
                _log.warning(
                    "lidar_clearance_violation",
                    lidar_min_dist_m=round(lidar_min_dist_m, 3),
                    threshold_m=self._cfg.min_forward_clearance_m,
                )
                is_emergency = True

        # -- Human detection (from observation if available) ---------------
        human_detected = bool(getattr(observation, "human_detected", False))
        human_dist_m = float(getattr(observation, "human_dist_m", float("inf")))

        if human_detected and human_dist_m < self._cfg.min_forward_clearance_m:
            is_emergency = True

        ctx = SafetyContext(
            ultrasonic_dist_m=observation.distance_m,
            forward_clearance_ok=forward_clearance_ok,
            battery_voltage=battery_voltage,
            valid_sensor_count=valid_sensor_count,
            loop_time_ms=loop_time_ms,
            is_emergency=is_emergency,
            human_detected=human_detected,
            human_dist_m=human_dist_m,
            lidar_min_dist_m=lidar_min_dist_m,
            lidar_clearance_ok=lidar_clearance_ok,
        )
        _log.debug(
            "safety_evaluate_result",
            is_emergency=is_emergency,
            valid_sensors=valid_sensor_count,
        )
        return ctx
