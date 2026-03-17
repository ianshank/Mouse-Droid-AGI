"""Drone safety monitor — evaluates drone-specific hazardous conditions.

Implements :class:`~mousedroid.safety.protocol.SafetyMonitorProtocol`.
Composes the ground-shared :class:`MouseDroidSafetyMonitor` and adds
altitude range, geofence breach, and GPS fix validation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger
from mousedroid.safety.context import SafetyContext
from mousedroid.safety.monitor import MouseDroidSafetyMonitor

if TYPE_CHECKING:
    from mousedroid.config.schema import FlightEnvelopeConfig, GeofenceConfig, SafetyConfig
    from mousedroid.sensing.protocol import ObservationProtocol

_log = get_logger(__name__)

_EARTH_RADIUS_M: float = 6_371_000.0
"""Mean Earth radius in metres for haversine distance approximation."""


class DroneSafetyMonitor:
    """Evaluates each observation for drone-specific hazardous conditions.

    Uses composition: delegates ground-common checks to
    :class:`MouseDroidSafetyMonitor`, then adds altitude, geofence,
    and GPS validation.

    Implements :class:`~mousedroid.safety.protocol.SafetyMonitorProtocol`.

    Args:
        safety_cfg: Shared safety configuration thresholds.
        envelope_cfg: Flight envelope constraints.
        geofence_cfg: Geofence boundary configuration.
    """

    def __init__(
        self,
        safety_cfg: SafetyConfig,
        envelope_cfg: FlightEnvelopeConfig | None = None,
        geofence_cfg: GeofenceConfig | None = None,
    ) -> None:
        self._ground_monitor = MouseDroidSafetyMonitor(safety_cfg)
        self._envelope = envelope_cfg
        self._geofence = geofence_cfg
        _log.info(
            "drone_safety_monitor_init",
            has_envelope=envelope_cfg is not None,
            has_geofence=geofence_cfg is not None,
        )

    # -- SafetyMonitorProtocol ---------------------------------------------

    def evaluate(
        self,
        observation: ObservationProtocol,
        loop_time_ms: float,
    ) -> SafetyContext:
        """Evaluate safety state for a drone observation.

        Args:
            observation: Fused sensor bundle for this tick.
            loop_time_ms: Wall-clock duration of the last loop iteration
                in milliseconds.

        Returns:
            A frozen :class:`SafetyContext` with drone fields populated.
        """
        # Delegate ground-common checks first.
        ground_ctx = self._ground_monitor.evaluate(observation, loop_time_ms)
        is_emergency = ground_ctx.is_emergency

        # The ground monitor reads battery from motor_state[3] which is
        # correct for ground layout [left_vel, right_vel, heading, battery_v].
        # For drone layout [vx, vy, vz, yaw_rate, altitude, battery_v, armed],
        # battery is at index 5.  Re-evaluate battery if motor_state is long enough.
        battery_voltage = ground_ctx.battery_voltage
        if len(observation.motor_state) >= 6:
            battery_voltage = float(observation.motor_state[5])
            # Re-evaluate battery emergency — ground monitor may have
            # incorrectly flagged index 3 (yaw_rate) as critical.
            battery_emergency = False
            if battery_voltage < self._ground_monitor._cfg.battery_critical_v:
                _log.error(
                    "battery_critical",
                    voltage=battery_voltage,
                    threshold=self._ground_monitor._cfg.battery_critical_v,
                )
                battery_emergency = True
            elif battery_voltage < self._ground_monitor._cfg.battery_warn_v:
                _log.warning(
                    "battery_low",
                    voltage=battery_voltage,
                    threshold=self._ground_monitor._cfg.battery_warn_v,
                )

            # If ground flagged emergency only due to battery (wrong index),
            # we need to recalculate.  Easiest: start with non-battery ground
            # emergency flags and re-add battery if truly critical.
            is_emergency = (
                not ground_ctx.forward_clearance_ok
                or ground_ctx.valid_sensor_count < self._ground_monitor._cfg.min_valid_sensors
                or ground_ctx.loop_time_ms > self._ground_monitor._cfg.max_loop_time_ms
                or battery_emergency
            )

        # -- Altitude check ------------------------------------------------
        altitude_m = float(getattr(observation, "altitude_m", 0.0))
        altitude_ok = True
        if self._envelope is not None:
            if altitude_m < self._envelope.min_altitude_m:
                _log.warning(
                    "altitude_below_minimum",
                    altitude_m=altitude_m,
                    min_altitude_m=self._envelope.min_altitude_m,
                )
                altitude_ok = False
                is_emergency = True
            elif altitude_m > self._envelope.max_altitude_m:
                _log.warning(
                    "altitude_above_maximum",
                    altitude_m=altitude_m,
                    max_altitude_m=self._envelope.max_altitude_m,
                )
                altitude_ok = False
                is_emergency = True

        # -- Geofence check ------------------------------------------------
        geofence_ok = True
        if self._geofence is not None and self._geofence.enabled:
            geofence_ok = self._check_geofence(observation, altitude_m)
            if not geofence_ok:
                is_emergency = True

        # -- GPS fix check -------------------------------------------------
        gps_fix = bool(getattr(observation, "gps_fix", True))
        if not gps_fix:
            _log.warning("gps_fix_lost")
            is_emergency = True

        # -- IMU health check ----------------------------------------------
        imu_healthy = bool(getattr(observation, "imu_healthy", True))
        if not imu_healthy:
            _log.warning("imu_unhealthy")
            is_emergency = True

        # -- Armed state ---------------------------------------------------
        armed = bool(getattr(observation, "armed", False))

        ctx = SafetyContext(
            # Ground-shared fields
            ultrasonic_dist_m=ground_ctx.ultrasonic_dist_m,
            forward_clearance_ok=ground_ctx.forward_clearance_ok,
            battery_voltage=battery_voltage,
            valid_sensor_count=ground_ctx.valid_sensor_count,
            loop_time_ms=ground_ctx.loop_time_ms,
            human_detected=ground_ctx.human_detected,
            human_dist_m=ground_ctx.human_dist_m,
            # Drone-specific fields
            altitude_m=altitude_m,
            altitude_ok=altitude_ok,
            geofence_ok=geofence_ok,
            gps_fix=gps_fix,
            imu_healthy=imu_healthy,
            armed=armed,
            # Computed
            is_emergency=is_emergency,
        )

        _log.debug(
            "drone_safety_evaluate_result",
            is_emergency=is_emergency,
            altitude_ok=altitude_ok,
            geofence_ok=geofence_ok,
            gps_fix=gps_fix,
        )
        return ctx

    # -- Private helpers ---------------------------------------------------

    def _check_geofence(
        self,
        observation: ObservationProtocol,
        altitude_m: float,
    ) -> bool:
        """Check whether the drone is within the geofence boundary.

        Args:
            observation: Current observation with optional GPS data.
            altitude_m: Current altitude AGL.

        Returns:
            True if within geofence, False if breached.
        """
        if self._geofence is None or not self._geofence.enabled:
            return True

        # Altitude ceiling
        if altitude_m > self._geofence.max_altitude_m:
            _log.warning(
                "geofence_altitude_breach",
                altitude_m=altitude_m,
                max_altitude_m=self._geofence.max_altitude_m,
            )
            return False

        # Horizontal radius (haversine approximation)
        gps_position = getattr(observation, "gps_position", None)
        if gps_position is not None:
            lat, lon, _ = gps_position
            distance = self._haversine_distance_m(
                self._geofence.center_lat,
                self._geofence.center_lon,
                lat,
                lon,
            )
            if distance > self._geofence.radius_m:
                _log.warning(
                    "geofence_radius_breach",
                    distance_m=distance,
                    radius_m=self._geofence.radius_m,
                )
                return False

        return True

    @staticmethod
    def _haversine_distance_m(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Compute haversine distance between two GPS coordinates.

        Args:
            lat1: Latitude of point 1 (degrees).
            lon1: Longitude of point 1 (degrees).
            lat2: Latitude of point 2 (degrees).
            lon2: Longitude of point 2 (degrees).

        Returns:
            Distance in metres.
        """
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return _EARTH_RADIUS_M * c
