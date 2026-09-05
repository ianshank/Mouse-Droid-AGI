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
        # Loop-overrun debounce state. ``_last_counted_tick_index`` dedupes the
        # second ``evaluate`` call the orchestrator makes on the sensor-recovery
        # path within a single tick — without it a naive streak counter would
        # double-increment and trip at half the configured threshold.
        self._loop_overrun_streak = 0
        self._last_counted_tick_index: int | None = None
        self._ticks_seen = 0

    # -- SafetyMonitorProtocol ---------------------------------------------

    def _evaluate_loop_timing(self, loop_time_ms: float, tick_index: int | None) -> bool:
        """Return ``True`` when loop timing warrants an emergency stop.

        Extracted from :meth:`evaluate` because that method sits at C901 = 14
        against a ceiling of 15: the debounce and warm-up branches below would
        not fit inside it.

        Two guards separate a genuine overrun from an expected one:

        * **Warm-up** (``loop_overrun_warmup_ticks``) — a Jetson's first ticks
          pay lazy CUDA context creation and TensorRT/ONNX kernel warm-up and
          routinely exceed the threshold. Overruns there are logged and
          counted but never escalate, so the rover does not emergency-stop at
          every boot.
        * **Debounce** (``loop_overrun_consecutive_ticks``) — an isolated GC
          pause or page fault is not a control-loop failure. Default 1
          preserves the historical single-sample trip exactly.

        ``tick_index`` dedupes repeated calls within one tick: the orchestrator
        evaluates twice when sensor recovery fires, and counting both would
        trip at half the configured streak. ``None`` means "caller does not
        track ticks", and every call is then counted — the pre-debounce
        behaviour, kept so callers predating ``tick_index`` still work.

        Args:
            loop_time_ms: Measured loop duration for the tick being judged.
            tick_index: Monotonic tick counter, or ``None``.

        Returns:
            ``True`` if this overrun should raise an emergency stop.
        """
        cfg = self._cfg
        first_call_this_tick = tick_index is None or tick_index != self._last_counted_tick_index
        if first_call_this_tick:
            self._last_counted_tick_index = tick_index
            self._ticks_seen += 1

        if loop_time_ms <= cfg.max_loop_time_ms:
            if first_call_this_tick:
                self._loop_overrun_streak = 0
            return False

        if first_call_this_tick:
            self._loop_overrun_streak += 1

        if self._ticks_seen <= cfg.loop_overrun_warmup_ticks:
            _log.info(
                "loop_overrun_warmup",
                loop_time_ms=loop_time_ms,
                max_ms=cfg.max_loop_time_ms,
                tick=self._ticks_seen,
                warmup_ticks=cfg.loop_overrun_warmup_ticks,
            )
            return False

        if self._loop_overrun_streak < cfg.loop_overrun_consecutive_ticks:
            _log.warning(
                "loop_overrun_debounced",
                loop_time_ms=loop_time_ms,
                max_ms=cfg.max_loop_time_ms,
                streak=self._loop_overrun_streak,
                required=cfg.loop_overrun_consecutive_ticks,
            )
            return False

        _log.error(
            "loop_overrun",
            loop_time_ms=loop_time_ms,
            max_ms=cfg.max_loop_time_ms,
            streak=self._loop_overrun_streak,
        )
        return True

    def evaluate(
        self,
        observation: ObservationProtocol,
        loop_time_ms: float,
        *,
        tick_index: int | None = None,
    ) -> SafetyContext:
        """Evaluate safety state from the current observation.

        Args:
            observation: Fused sensor bundle for this tick.
            loop_time_ms: Wall-clock duration of the loop iteration being
                judged, in milliseconds.
            tick_index: Monotonic tick counter, used to dedupe the repeated
                call the orchestrator makes on the sensor-recovery path.
                Keyword-only with a default so every existing caller, test
                double and ``SafetyMonitorProtocol`` implementation keeps
                working unchanged.

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
        if self._evaluate_loop_timing(loop_time_ms, tick_index):
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
