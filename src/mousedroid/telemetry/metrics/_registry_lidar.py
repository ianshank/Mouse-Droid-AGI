"""LiDAR sector-distance / min-distance / scan-point gauges.

Distinct from the raw-scan-published/dropped counters in
``_registry_streaming.py`` (PR #4 telemetry-streaming family) — the original
code renders these two LiDAR-adjacent families separately (``_families_lidar``
is gated by ``cfg.track_lidar``; the PR #4 streaming counters are ungated),
so the split preserves that boundary rather than merging them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.telemetry.metrics.primitives import (
    _Gauge,
    _LabeledGauge,
    _render_gauge,
    _render_labeled_gauge,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import MetricsConfig


class _LidarMetricsMixin:
    """Per-sector LiDAR distance, minimum distance, and scan-point count gauges."""

    # Populated by ``_CoreMetricsMixin._init_core_metrics``, which always runs
    # first from ``MetricsRegistry.__init__``.
    _cfg: MetricsConfig

    def _init_lidar_metrics(self, cfg: MetricsConfig) -> None:
        """Initialise LiDAR sector/min-distance/scan-point gauges.

        Args:
            cfg: Metrics configuration with namespace and toggle flags.
        """
        ns = cfg.namespace

        self._lidar_sector_distance_m = _LabeledGauge()
        self._lidar_min_distance_m = _Gauge()
        self._lidar_scan_points = _Gauge()

        self._name_lidar_sector_distance = f"{ns}_lidar_sector_distance_m"
        self._name_lidar_min_distance = f"{ns}_lidar_min_distance_m"
        self._name_lidar_scan_points = f"{ns}_lidar_scan_points"

    # ------------------------------------------------------------------
    # Write helpers — called by broadcast loop / orchestrator
    # ------------------------------------------------------------------

    def set_lidar_sectors(self, sectors: list[float], max_range_m: float) -> None:
        """Publish per-sector LiDAR distances as a labeled gauge.

        Args:
            sectors: Normalised sector distances in ``[0.0, 1.0]``, where
                ``1.0`` equals ``max_range_m`` (no obstacle detected).
            max_range_m: LiDAR maximum detection range in metres; used to
                convert the normalised sector value into metres for scrape.
        """
        if not self._cfg.track_lidar:
            return
        payload = {str(i): float(v) * max_range_m for i, v in enumerate(sectors)}
        self._lidar_sector_distance_m.set_many(payload)

    def set_lidar_min_distance_m(self, value: float) -> None:
        """Set the minimum LiDAR distance reading in metres."""
        if self._cfg.track_lidar:
            self._lidar_min_distance_m.set(value)

    def set_lidar_scan_points(self, count: int) -> None:
        """Set the raw LiDAR scan point count (liveness signal)."""
        if self._cfg.track_lidar:
            self._lidar_scan_points.set(float(count))

    # ------------------------------------------------------------------
    # Prometheus text exposition — family renderer
    # ------------------------------------------------------------------

    def _families_lidar(self) -> list[list[str]]:
        """LiDAR sector / min-distance / scan-point gauges."""
        cfg = self._cfg
        out: list[list[str]] = []
        if cfg.track_lidar:
            sector_snapshot = self._lidar_sector_distance_m.snapshot()
            if sector_snapshot:
                out.append(
                    _render_labeled_gauge(
                        self._name_lidar_sector_distance,
                        "Per-sector LiDAR distance in metres (label: sector)",
                        "sector",
                        sector_snapshot,
                    )
                )
            out.append(
                _render_gauge(
                    self._name_lidar_min_distance,
                    "Minimum LiDAR distance across all sectors (metres)",
                    self._lidar_min_distance_m.value,
                )
            )
            out.append(
                _render_gauge(
                    self._name_lidar_scan_points,
                    "Number of raw points in the last LiDAR scan",
                    self._lidar_scan_points.value,
                )
            )
        return out
