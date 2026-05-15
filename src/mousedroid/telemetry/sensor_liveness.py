"""Sensor liveness tracking — distinguishes disabled / awaiting / live / stale.

The TelemetryFrame previously zeroed missing sensor fields (e.g.
``lidar_n_points = 0``), which is indistinguishable from "sensor enabled
but broken". This module gives the frame builder a small state machine
so each sensor can publish one of four explicit states:

* ``disabled`` — the sensor is not configured for this build.
* ``awaiting`` — configured but the first frame has not arrived yet.
* ``live`` — fresh data within the staleness window.
* ``stale`` — last frame is older than ``sensor_liveness_stale_s``.

The tracker is intentionally framework-free: callers feed it timestamps
and a config object and it returns a state plus an age. The frame
builder and the Prometheus broadcast loop both consume the same result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LivenessState = Literal["disabled", "awaiting", "live", "stale"]
"""Discrete liveness states. Order matches severity ascending so a sort
on the literal yields a stable rendering order."""

# Public ordering for dashboard / Prometheus label-set iteration. Kept
# as a module-level constant so callers don't depend on the implicit
# string ordering inside :data:`LivenessState`.
LIVENESS_STATES: tuple[LivenessState, ...] = ("disabled", "awaiting", "live", "stale")


@dataclass(frozen=True)
class SensorLiveness:
    """Single sensor's liveness snapshot.

    Attributes:
        state: One of :data:`LIVENESS_STATES`.
        age_s: Seconds since the last observation, or ``None`` when no
            observation has ever been recorded (``state`` is then either
            ``disabled`` or ``awaiting``).
    """

    state: LivenessState
    age_s: float | None

    def to_dict(self) -> dict[str, object]:
        """Serialise to a JSON-friendly dict.

        Returns:
            ``{"state": <str>, "age_s": <float | None>}``.
        """
        return {"state": self.state, "age_s": self.age_s}


class SensorLivenessTracker:
    """Records last-observed timestamps and reports per-sensor liveness.

    The tracker holds NO async state and is fully synchronous. The
    publisher / orchestrator records observations as they arrive; the
    frame builder polls :meth:`snapshot` once per frame.

    Sensors must be registered before :meth:`mark_observed` or
    :meth:`snapshot` calls; ``register`` declares whether the sensor is
    enabled for this deployment.
    """

    def __init__(self, *, stale_s: float) -> None:
        """Initialise the tracker.

        Args:
            stale_s: Age threshold (seconds) above which a sensor is
                reported as ``stale`` instead of ``live``. Must be > 0.

        Raises:
            ValueError: When ``stale_s`` is not strictly positive.
        """
        if stale_s <= 0:
            raise ValueError(f"stale_s must be > 0, got {stale_s}")
        self._stale_s: float = stale_s
        # value tuple = (enabled, last_observed_monotonic_s | None)
        self._sensors: dict[str, tuple[bool, float | None]] = {}

    @property
    def stale_s(self) -> float:
        """Current staleness threshold."""
        return self._stale_s

    def register(self, sensor: str, *, enabled: bool) -> None:
        """Declare a sensor.

        Repeated registration updates the enabled flag. The
        last-observed timestamp survives a re-registration so a config
        reload doesn't trash existing data.

        Args:
            sensor: Sensor identifier (e.g. ``"lidar"``, ``"camera"``).
            enabled: Whether the sensor is active in this deployment.
        """
        existing = self._sensors.get(sensor)
        last_ts = existing[1] if existing is not None else None
        self._sensors[sensor] = (enabled, last_ts)

    def mark_observed(self, sensor: str, now_s: float) -> None:
        """Record that a sensor produced data at ``now_s``.

        Auto-registers the sensor as enabled if it was not registered.

        Args:
            sensor: Sensor identifier.
            now_s: Monotonic timestamp (seconds) for the observation.
        """
        existing = self._sensors.get(sensor)
        enabled = existing[0] if existing is not None else True
        self._sensors[sensor] = (enabled, now_s)

    def snapshot(self, *, now_s: float) -> dict[str, SensorLiveness]:
        """Return the liveness state of every registered sensor.

        Args:
            now_s: Current monotonic timestamp (seconds).

        Returns:
            Mapping ``{sensor_name: SensorLiveness}`` covering every
            registered sensor. Sensors not registered are absent (use
            :meth:`register` to add them).
        """
        out: dict[str, SensorLiveness] = {}
        for sensor, (enabled, last_ts) in self._sensors.items():
            out[sensor] = self._classify(enabled=enabled, last_ts=last_ts, now_s=now_s)
        return out

    def _classify(
        self,
        *,
        enabled: bool,
        last_ts: float | None,
        now_s: float,
    ) -> SensorLiveness:
        """Map (enabled, last_ts) → ``SensorLiveness``."""
        if not enabled:
            return SensorLiveness(state="disabled", age_s=None)
        if last_ts is None:
            return SensorLiveness(state="awaiting", age_s=None)
        age = max(0.0, now_s - last_ts)
        if age > self._stale_s:
            return SensorLiveness(state="stale", age_s=age)
        return SensorLiveness(state="live", age_s=age)
