"""MetricsRegistry to Google Cloud Monitoring exporter.

Periodically reads gauges and counters from the existing Prometheus-style
``MetricsRegistry`` and writes them as Cloud Monitoring custom metrics.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import GCPConfig
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)


class CloudMetricsExporter:
    """Exports ``MetricsRegistry`` metrics to Cloud Monitoring.

    Args:
        cfg: GCP configuration.
        registry: The local metrics registry to export from.
    """

    def __init__(self, cfg: GCPConfig, registry: MetricsRegistry) -> None:
        self._cfg = cfg
        self._mon_cfg = cfg.monitoring
        self._registry = registry
        self._project_id = cfg.project_id
        self._prefix = self._mon_cfg.metric_prefix
        self._interval_s = self._mon_cfg.export_interval_s

        self._client: Any | None = None
        self._project_path: str = ""
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Initialise the Cloud Monitoring client and start the export loop."""
        from google.cloud import monitoring_v3

        from mousedroid.cloud._auth import resolve_credentials

        creds, _project = resolve_credentials(self._cfg)
        self._client = monitoring_v3.MetricServiceClient(credentials=creds)
        self._project_path = f"projects/{self._project_id}"
        self._running = True
        self._task = asyncio.create_task(self._export_loop())
        _log.info(
            "cloud_metrics_exporter_started",
            prefix=self._prefix,
            interval_s=self._interval_s,
        )

    async def export_once(self) -> None:
        """Run a single export cycle — read registry, write to Cloud Monitoring.

        Parses the Prometheus text exposition output from the registry to
        extract gauge values, then converts them to Cloud Monitoring
        ``TimeSeries`` objects.
        """
        if self._client is None:
            return

        metrics = self._parse_gauge_metrics()
        if not metrics:
            return

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._write_metrics,
                metrics,
            )
            _log.debug("cloud_metrics_exported", count=len(metrics))
        except Exception:
            _log.warning("cloud_metrics_export_failed", exc_info=True)

    def _parse_gauge_metrics(self) -> dict[str, float]:
        """Parse gauge values from the Prometheus text exposition output.

        Returns:
            Dictionary of ``{metric_name: value}`` for gauge-type metrics.
        """
        text = self._registry.render_prometheus()
        result: dict[str, float] = {}
        current_type: str | None = None

        for line in text.splitlines():
            if line.startswith("# TYPE "):
                parts = line.split()
                current_type = "gauge" if len(parts) >= 4 and parts[3] == "gauge" else None
            elif line.startswith("# "):
                continue
            elif current_type == "gauge" and line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        result[parts[0]] = float(parts[1])
                    except ValueError:
                        continue
        return result

    def _write_metrics(self, metrics: dict[str, float]) -> None:
        """Write metrics to Cloud Monitoring (runs in executor thread).

        Args:
            metrics: Dictionary of ``{metric_name: value}`` to export.
        """
        from google.cloud.monitoring_v3 import types
        from google.protobuf import timestamp_pb2

        now_ts = timestamp_pb2.Timestamp()
        now_ts.FromSeconds(int(time.time()))

        series_list: list[Any] = []
        for name, value in metrics.items():
            series = types.TimeSeries()
            series.metric.type = f"{self._prefix}/{name}"
            series.resource.type = "generic_node"
            series.resource.labels["project_id"] = self._project_id
            series.resource.labels["location"] = "global"
            series.resource.labels["namespace"] = "mousedroid"
            series.resource.labels["node_id"] = self._cfg.robot_id

            point = types.Point()
            point.interval.end_time = now_ts
            point.value.double_value = value
            series.points.append(point)
            series_list.append(series)

        if series_list:
            self._client.create_time_series(self._project_path, series_list)

    async def stop(self) -> None:
        """Stop the export loop and release resources."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._client = None
        _log.info("cloud_metrics_exporter_stopped")

    async def _export_loop(self) -> None:
        """Periodic export loop — runs as a background task."""
        while self._running:
            try:
                await asyncio.sleep(self._interval_s)
                await self.export_once()
            except asyncio.CancelledError:
                break
            except Exception:
                _log.warning("cloud_metrics_loop_error", exc_info=True)
