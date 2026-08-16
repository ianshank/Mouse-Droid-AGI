"""PR #4 — telemetry streaming + dashboard-liveness metrics.

Sensor liveness, mDNS registration, the real bound port, and the raw LiDAR
streaming-queue published/dropped counters. All ungated by ``cfg.track_*``
toggles — emitted conditionally on non-empty/non-zero state instead, so
deployments that never touch these APIs don't produce noisy zero-valued
series (see ``_families_streaming``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.telemetry.metrics.primitives import (
    _Counter,
    _DoubleLabeledGauge,
    _Gauge,
    _LabeledGauge,
    _render_counter,
    _render_double_labeled_gauge,
    _render_gauge,
    _render_labeled_gauge,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import MetricsConfig


class _StreamingMetricsMixin:
    """PR #4 telemetry-streaming + dashboard-liveness metric family.

    Carries no ``track_*`` toggle (every write helper is unconditional; the
    render side gates on non-empty/non-zero snapshot state instead), so this
    mixin — unlike most others — never needs to read ``self._cfg``.
    """

    def _init_streaming_metrics(self, cfg: MetricsConfig) -> None:
        """Initialise PR #4 streaming / dashboard-liveness metrics.

        Args:
            cfg: Metrics configuration with namespace and toggle flags.
        """
        ns = cfg.namespace

        # PR #4 — telemetry streaming + dashboard observability.
        # Sensor liveness gauge: labeled by (sensor, state). Values are
        # 0/1 indicators (mutually exclusive per sensor).
        self._sensor_liveness = _DoubleLabeledGauge()
        # mDNS register success indicator: 0 = not registered, 1 = OK.
        self._mdns_registered = _LabeledGauge()
        # Real bound port (useful when port_discovery_strategy=
        # fallback_range/kernel_assigned).
        self._bound_port = _Gauge()
        # Raw LiDAR streaming counters.
        self._lidar_raw_published = _Counter()
        self._lidar_raw_dropped = _Counter()

        # PR #4 metric names
        self._name_sensor_liveness = f"{ns}_telemetry_sensor_liveness"
        self._name_mdns_registered = f"{ns}_telemetry_mdns_registered"
        self._name_bound_port = f"{ns}_telemetry_bound_port"
        self._name_lidar_raw_published = f"{ns}_telemetry_lidar_raw_published_total"
        self._name_lidar_raw_dropped = f"{ns}_telemetry_lidar_raw_dropped_total"

    # ------------------------------------------------------------------
    # PR #4 — telemetry streaming + dashboard liveness helpers
    # ------------------------------------------------------------------

    def set_sensor_liveness(self, snapshot: dict[str, str]) -> None:
        """Update the sensor liveness gauge from a name → state map.

        Always emitted; observability of live/stale sensors must never
        be silenced by config toggles.

        For every sensor in ``snapshot`` the gauge is set to 1.0 for
        the currently-active state and 0.0 for every other known state
        — so a Prometheus query like ``telemetry_sensor_liveness{
        sensor="lidar"}`` returns one 1 and three 0s at every scrape.

        Args:
            snapshot: Mapping ``{sensor_name: state}`` where ``state``
                is one of ``disabled`` / ``awaiting`` / ``live`` /
                ``stale``.
        """
        from mousedroid.telemetry.sensor_liveness import LIVENESS_STATES

        for sensor, current_state in snapshot.items():
            for state in LIVENESS_STATES:
                self._sensor_liveness.set(
                    sensor,
                    state,
                    1.0 if state == current_state else 0.0,
                )

    def set_mdns_registered(self, service_name: str, ok: bool) -> None:
        """Set the mDNS registration indicator gauge.

        Args:
            service_name: mDNS service name (becomes the ``service`` label).
            ok: ``True`` when the service is registered.
        """
        self._mdns_registered.set(service_name, 1.0 if ok else 0.0)

    def set_bound_port(self, port: int) -> None:
        """Record the real bound TCP port for the telemetry server.

        Useful when ``port_discovery_strategy`` is ``fallback_range`` or
        ``kernel_assigned`` and the deployed port can differ from
        ``TelemetryConfig.port``.

        Args:
            port: The OS-assigned or fallback-discovered port number.
        """
        self._bound_port.set(float(port))

    def inc_lidar_raw_published(self, amount: int = 1) -> None:
        """Increment the raw LiDAR scans-published counter."""
        self._lidar_raw_published.inc(amount)

    def inc_lidar_raw_dropped(self, amount: int = 1) -> None:
        """Increment the raw LiDAR scans-dropped counter."""
        self._lidar_raw_dropped.inc(amount)

    # ------------------------------------------------------------------
    # Prometheus text exposition — family renderer
    # ------------------------------------------------------------------

    def _families_streaming(self) -> list[list[str]]:
        """PR-#4 streaming / dashboard-liveness families."""
        out: list[list[str]] = []
        # PR #4 — telemetry streaming + dashboard liveness metrics.
        # Emit conditionally so deployments that never touch the new
        # APIs don't produce noisy zero-valued series.
        liveness_snapshot = self._sensor_liveness.snapshot()
        if liveness_snapshot:
            out.append(
                _render_double_labeled_gauge(
                    self._name_sensor_liveness,
                    "Per-sensor liveness (labels: sensor, state)",
                    "sensor",
                    "state",
                    liveness_snapshot,
                )
            )
        mdns_snapshot = self._mdns_registered.snapshot()
        if mdns_snapshot:
            out.append(
                _render_labeled_gauge(
                    self._name_mdns_registered,
                    "mDNS service registration indicator (1=registered, 0=failed)",
                    "service",
                    mdns_snapshot,
                )
            )
        if self._bound_port.value > 0:
            out.append(
                _render_gauge(
                    self._name_bound_port,
                    "Actual TCP port the telemetry server bound to",
                    self._bound_port.value,
                )
            )
        if self._lidar_raw_published.value > 0 or self._lidar_raw_dropped.value > 0:
            out.append(
                _render_counter(
                    self._name_lidar_raw_published.removesuffix("_total"),
                    "Raw LiDAR scans published to the streaming queue",
                    self._lidar_raw_published.value,
                )
            )
            out.append(
                _render_counter(
                    self._name_lidar_raw_dropped.removesuffix("_total"),
                    "Raw LiDAR scans dropped (streaming queue full)",
                    self._lidar_raw_dropped.value,
                )
            )
        return out
