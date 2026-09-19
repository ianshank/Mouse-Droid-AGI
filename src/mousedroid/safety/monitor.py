"""MouseDroid safety monitor — evaluates observations for hazardous conditions.

Implements :class:`~mousedroid.safety.protocol.SafetyMonitorProtocol`.
Each control-loop tick the monitor produces a frozen
:class:`~mousedroid.safety.context.SafetyContext` that agents consume.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from mousedroid.constants import LOG_PRECISION_DP, MOTOR_STATE_BATTERY_INDEX
from mousedroid.logging.setup import get_logger
from mousedroid.safety.context import SafetyContext

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mousedroid.config.schema import SafetyConfig
    from mousedroid.safety.latch import EmergencyLatchProtocol
    from mousedroid.sensing.human_presence import HumanPresenceProtocol
    from mousedroid.sensing.protocol import ObservationProtocol

_log = get_logger(__name__)

LIDAR_UNAVAILABLE_DIST_M: float = 0.0
"""Clearance reported for a LiDAR-less tick under a fail-closed policy.

Deliberately NOT a config field: this is not a tunable threshold but the
definition of the worst case — "assume an obstacle is in contact". Every
clearance comparison in this module and in
:mod:`mousedroid.safety.projector` is a strict ``<`` against a ``gt=0``
threshold, so 0.0 is guaranteed to fail all of them regardless of how the
operator tunes ``min_forward_clearance_m``, ``lidar_brake_distance_m`` or
``tight_quarters_dist_m``. Making it tunable would only create a way to
configure the fail-closed path back open.
"""


class MouseDroidSafetyMonitor:
    """Evaluates each observation for hazardous conditions.

    Checks forward clearance, battery voltage, sensor validity, sensor
    staleness, and loop timing.  Any critical condition sets
    ``is_emergency=True`` on the returned :class:`SafetyContext`.

    Implements :class:`~mousedroid.safety.protocol.SafetyMonitorProtocol`.

    Args:
        cfg: Safety configuration thresholds.
        human_presence: Source of human-presence readings. Keyword-only with
            a ``None`` default so every existing caller -- the factory, the
            MCP tool bridge and ~20 test sites that construct
            ``MouseDroidSafetyMonitor(SafetyConfig())`` -- keeps working
            unchanged. ``None`` resolves to
            :class:`~mousedroid.sensing.human_presence.NullHumanPresenceDetector`.
    """

    def __init__(
        self,
        cfg: SafetyConfig,
        *,
        human_presence: HumanPresenceProtocol | None = None,
        latch: EmergencyLatchProtocol | None = None,
    ) -> None:
        self._cfg = cfg
        # ``None`` keeps the pre-latch path exactly as it was: no latch
        # object, no file, and ``_apply_emergency_latch`` returns on its
        # first line.
        self._latch = latch
        if human_presence is None:
            # Local import: keeps ``mousedroid.safety`` from growing a runtime
            # edge into ``sensing`` (whose package __init__ pulls in the full
            # SensorManager). Mirrors orchestrator.py's NullHookRegistry default.
            from mousedroid.sensing.human_presence import NullHumanPresenceDetector

            human_presence = NullHumanPresenceDetector()
        self._human_presence = human_presence
        self._log_human_presence_capability()
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
        # Observation timestamp of the last tick that carried usable LiDAR
        # features. ``None`` until the monitor has seen either a usable scan
        # or its first LiDAR-less tick under a fail-closed policy — see
        # ``_evaluate_lidar_clearance`` for why it is seeded there rather
        # than left unset.
        self._lidar_last_valid_ts: float | None = None

    def _log_human_presence_capability(self) -> None:
        """Announce whether the human-safety interlocks can actually fire.

        Peer review D-1. Three interlocks -- the Law-1 stop in
        ``MouseDroidNavigationAgent.act``, the human clamp in
        ``GeometricSafetyProjector.project`` and the human branch of
        :meth:`evaluate` -- depend on a presence reading that no source in
        this repo produces. They were silently inert. S-11 established the
        remedy for exactly this shape: a safety feature that reads as
        available while being dormant must say so at construction.

        Deliberately log-only. Failing closed here would mean assuming a
        human at zero distance, which is a permanent emergency stop -- the
        inverse of the ``lidar_unavailable_policy`` trade-off, and the
        wrong one.

        Logged from ``__init__`` rather than the factory because the
        factory is only one of several construction paths; S-11's fix
        lives in the driver for the same reason.
        """
        source = self._human_presence.source_name
        if self._human_presence.can_detect:
            _log.info("human_presence_source_bound", source=source)
            return
        _log.warning(
            "human_presence_source_unavailable",
            source=source,
            inert_interlocks=(
                "MouseDroidNavigationAgent.act law-1 stop; "
                "GeometricSafetyProjector.project human clamp; "
                "MouseDroidSafetyMonitor.evaluate human branch"
            ),
            unconsumable_budgets=(
                "safety.projector.human_keepout_m; "
                "safety.projector.human_proximity_speed_mps; "
                "three_laws.human_safety_radius_m"
            ),
            remedy=(
                "wire a HumanPresenceProtocol source through "
                "factory.safety.build_human_presence_detector"
            ),
        )

    def _apply_emergency_latch(
        self, live_emergency: bool, causes: tuple[str, ...]
    ) -> tuple[bool, bool]:
        """Merge the live verdict with the latch (peer review D-5).

        A tuple return rather than an inline ``or`` so :meth:`evaluate`
        gains no branch: it is measured at 13 against the ``ruff C901``
        ceiling of 15, and every branch here would count against it.

        Returns:
            ``(is_emergency, emergency_latched)``. When no latch is wired
            the first element is ``live_emergency`` unchanged and the
            second is ``False``, which is byte-identical to pre-latch
            behaviour.
        """
        if self._latch is None:
            return live_emergency, False
        if live_emergency and self._latch.trip(
            causes[0] if causes else "unspecified", causes=causes
        ):
            _log.error("estop_latch_tripped", causes=list(causes))
        latched = self._latch.is_latched
        return live_emergency or latched, latched

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

    def _evaluate_lidar_clearance(
        self,
        lidar_features: NDArray[np.float32] | None,
        timestamp: float,
    ) -> tuple[float, bool, bool]:
        """Judge 360-degree LiDAR clearance, failing CLOSED when data is absent.

        Extracted from :meth:`evaluate` both to keep that method under the
        ``ruff C901`` ceiling (it sat at 14 of 15 before this branch existed)
        and because "what does an ABSENT LiDAR reading mean" is a policy
        question worth isolating from the arithmetic of a present one.

        The absent case used to be unreachable in practice, and that was the
        bug. ``SensorManager._safe_lidar_read`` substituted
        ``np.ones(feature_dim)`` for a failed read; features are normalised
        range fractions, so all-ones read through here as ``1.0 *
        lidar_max_range_m`` — a dead LiDAR reported full clearance in every
        sector and ``lidar_clearance_ok`` stayed ``True``. The sensing layer
        now reports ``None``, and this method decides what that means.

        The generic ``sensor_stale_s`` sweep in :meth:`evaluate` is not a
        backstop for this. It only fires for a mask slot that was valid at
        least once, so a LiDAR that is dead *from boot* never enters
        ``_last_valid_timestamps`` and never goes stale — it fails open
        forever. Seeding ``_lidar_last_valid_ts`` on the first LiDAR-less
        tick below is what closes that hole.

        Args:
            lidar_features: Normalised sector ranges for this tick, or
                ``None``/empty when the modality produced nothing.
            timestamp: Observation timestamp, used as the grace-window clock
                so the window tracks sensor time rather than wall time.

        Returns:
            ``(lidar_min_dist_m, lidar_clearance_ok, is_emergency)``.
        """
        cfg = self._cfg

        if lidar_features is not None and len(lidar_features) > 0:
            self._lidar_last_valid_ts = timestamp
            # Features are normalised distances (min_in_sector / max_range).
            # Convert to metres using the maximum observed feature range.
            lidar_min_dist_m = float(np.min(lidar_features)) * cfg.lidar_max_range_m
            if lidar_min_dist_m < cfg.min_forward_clearance_m:
                _log.warning(
                    "lidar_clearance_violation",
                    lidar_min_dist_m=round(lidar_min_dist_m, LOG_PRECISION_DP),
                    threshold_m=cfg.min_forward_clearance_m,
                )
                return lidar_min_dist_m, False, True
            return lidar_min_dist_m, True, False

        # -- No usable LiDAR this tick ------------------------------------
        policy = cfg.lidar_unavailable_policy
        if policy == "ignore":
            # Pre-existing behaviour, kept as the default so existing YAML
            # deploys unchanged: the modality is simply not consulted.
            return math.inf, True, False

        if self._lidar_last_valid_ts is None:
            # Never seen a usable scan. Start the grace clock now so a LiDAR
            # that was dead before the first tick still trips, instead of
            # waiting forever for a "last valid" time that never arrives.
            self._lidar_last_valid_ts = timestamp

        elapsed = timestamp - self._lidar_last_valid_ts
        if elapsed < cfg.lidar_unavailable_grace_s:
            # Strict ``<``, so a grace of 0.0 really is no grace: the policy
            # fires on the very first tick without usable features rather than
            # granting a free one. That also keeps the window independent of
            # clock granularity — on a host whose monotonic clock ticks every
            # ~16 ms, two 30 Hz observations can share a timestamp, and an
            # inclusive comparison would hand a zero-grace deployment a
            # silent reprieve whenever they did.
            _log.debug(
                "lidar_unavailable_grace",
                elapsed_s=round(elapsed, LOG_PRECISION_DP),
                grace_s=cfg.lidar_unavailable_grace_s,
                policy=policy,
            )
            return math.inf, True, False

        is_emergency = policy == "emergency"
        log_unavailable = _log.error if is_emergency else _log.warning
        log_unavailable(
            "lidar_unavailable",
            elapsed_s=round(elapsed, LOG_PRECISION_DP),
            grace_s=cfg.lidar_unavailable_grace_s,
            policy=policy,
            reported_clearance_m=LIDAR_UNAVAILABLE_DIST_M,
            hint=(
                "no LiDAR features this tick — reporting worst-case clearance "
                "rather than reading through to a fabricated distance"
            ),
        )
        return LIDAR_UNAVAILABLE_DIST_M, False, is_emergency

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
        causes: list[str] = []

        # -- Forward clearance ---------------------------------------------
        forward_clearance_ok = observation.distance_m >= self._cfg.min_forward_clearance_m
        if not forward_clearance_ok:
            _log.warning(
                "forward_clearance_violation",
                distance_m=observation.distance_m,
                threshold_m=self._cfg.min_forward_clearance_m,
            )
            is_emergency = True
            causes.append("forward_clearance_violation")

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
                causes.append("battery_critical")
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
                        elapsed_s=round(elapsed, LOG_PRECISION_DP),
                        threshold_s=self._cfg.sensor_stale_s,
                    )
                    is_emergency = True
                    causes.append("sensor_stale")

        # -- Valid sensor count (uses original mask; staleness is an additional emergency trigger)
        valid_sensor_count = int(np.sum(observation.valid_mask > 0.0))
        if valid_sensor_count < self._cfg.min_valid_sensors:
            _log.error(
                "insufficient_valid_sensors",
                valid=valid_sensor_count,
                required=self._cfg.min_valid_sensors,
            )
            is_emergency = True
            causes.append("insufficient_valid_sensors")

        # -- Loop timing ---------------------------------------------------
        if self._evaluate_loop_timing(loop_time_ms, tick_index):
            is_emergency = True
            causes.append("loop_overrun")

        # -- LiDAR 360-degree clearance ------------------------------------
        lidar_min_dist_m, lidar_clearance_ok, lidar_emergency = self._evaluate_lidar_clearance(
            observation.lidar_features,
            current_time,
        )
        if lidar_emergency:
            is_emergency = True
            causes.append("lidar_emergency")

        # -- Human detection (from the injected HumanPresenceProtocol) -----
        # NOT from the observation. It used to be read off the observation
        # through a defaulting attribute lookup that no observation type
        # could ever satisfy -- see peer review D-1 and
        # sensing/human_presence.py. This comment said "from observation if
        # available" until 2026-09-19; leaving it would have invited exactly
        # the reintroduction test_human_presence_source.py guards against.
        # That pin is a source-level gate, so the old expression is described
        # here rather than quoted: a commented-out lookup is one keystroke
        # from being a live one.
        presence = self._human_presence.sample(observation)
        human_detected = presence.detected
        human_dist_m = presence.distance_m

        if human_detected and human_dist_m < self._cfg.min_forward_clearance_m:
            is_emergency = True
            causes.append("human_proximity")

        is_emergency, emergency_latched = self._apply_emergency_latch(is_emergency, tuple(causes))

        ctx = SafetyContext(
            ultrasonic_dist_m=observation.distance_m,
            forward_clearance_ok=forward_clearance_ok,
            battery_voltage=battery_voltage,
            valid_sensor_count=valid_sensor_count,
            loop_time_ms=loop_time_ms,
            is_emergency=is_emergency,
            emergency_latched=emergency_latched,
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
