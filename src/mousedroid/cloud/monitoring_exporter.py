"""MetricsRegistry to Google Cloud Monitoring exporter.

Periodically reads gauges and counters from the existing Prometheus-style
``MetricsRegistry`` and writes them as Cloud Monitoring custom metrics.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from mousedroid.common.async_utils import cancel_and_drain, spawn_tracked
from mousedroid.logging.setup import get_logger

_TRANSIENT_MONITORING_EXCEPTIONS: tuple[type[BaseException], ...]

try:
    from google.api_core.exceptions import (
        DeadlineExceeded,
        RetryError,
        ServiceUnavailable,
    )

    _TRANSIENT_MONITORING_EXCEPTIONS = (
        RetryError,
        DeadlineExceeded,
        ServiceUnavailable,
        TimeoutError,
        ConnectionError,
        OSError,
    )
except ImportError:  # pragma: no cover - optional cloud dependency
    _TRANSIENT_MONITORING_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)

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
        # Derive the generic_node namespace label from the metric prefix
        # tail segment so it stays in lock-step with whatever namespace
        # the operator configured (e.g. "custom.googleapis.com/foo" ->
        # "foo"). Fallback to robot_id when the prefix is empty so the
        # Cloud Monitoring write never uses a hardcoded string.
        self._resource_namespace = self._prefix.rstrip("/").rsplit("/", 1)[-1] or cfg.robot_id
        # Operator-supplied labels (env, region, fleet, …) are applied
        # on top of the required generic_node labels; only non-reserved
        # keys are respected.
        self._extra_labels: dict[str, str] = dict(cfg.metrics_labels)

        self._client: Any | None = None
        self._project_path: str = ""
        self._task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._running = False

    async def start(self) -> None:
        """Initialise the Cloud Monitoring client and start the export loop."""
        from google.cloud import monitoring_v3

        from mousedroid.cloud._auth import resolve_credentials

        creds, _project = resolve_credentials(self._cfg)
        self._client = monitoring_v3.MetricServiceClient(credentials=creds)
        self._project_path = f"projects/{self._project_id}"
        self._running = True
        self._task = spawn_tracked(
            self._background_tasks,
            self._export_loop(),
            name=self._export_loop.__name__,
        )
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
        except _TRANSIENT_MONITORING_EXCEPTIONS:
            _log.warning("cloud_metrics_export_failed", transient=True, exc_info=True)
        except Exception:
            _log.warning("cloud_metrics_export_failed", transient=False, exc_info=True)

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

    @staticmethod
    def _parse_metric_name(name: str) -> tuple[str, dict[str, str]]:
        r"""Split a Prometheus-format metric name into base name and labels.

        Args:
            name: Metric name, possibly with labels (e.g. ``"foo{a=\"x\"}"``).

        Returns:
            Tuple of ``(base_name, labels_dict)``.
        """
        brace = name.find("{")
        if brace == -1:
            return name, {}
        base = name[:brace]
        label_str = name[brace + 1 :].rstrip("}")
        labels: dict[str, str] = {}
        for match in re.finditer(r'(\w+)="([^"]*)"', label_str):
            labels[match.group(1)] = match.group(2)
        return base, labels

    def _write_metrics(self, metrics: dict[str, float]) -> None:
        """Write metrics to Cloud Monitoring (runs in executor thread).

        Args:
            metrics: Dictionary of ``{metric_name: value}`` to export.
        """
        from google.cloud.monitoring_v3 import types

        timestamp_module = importlib.import_module("google.protobuf.timestamp_pb2")
        timestamp_api = cast(Any, timestamp_module)
        timestamp_ctor = cast(Callable[[], Any], timestamp_api.Timestamp)

        now_ts = timestamp_ctor()
        now_ts.FromSeconds(int(time.time()))

        series_list: list[Any] = []
        # Reserved generic_node resource-label keys that operator labels
        # must never override, to keep the Cloud Monitoring schema valid.
        reserved_resource_keys = frozenset({"project_id", "location", "namespace", "node_id"})
        for name, value in metrics.items():
            base_name, metric_labels = self._parse_metric_name(name)
            series = types.TimeSeries()
            series.metric.type = f"{self._prefix}/{base_name}"
            series.resource.type = "generic_node"
            series.resource.labels["project_id"] = self._project_id
            series.resource.labels["location"] = "global"
            series.resource.labels["namespace"] = self._resource_namespace
            series.resource.labels["node_id"] = self._cfg.robot_id
            for label_key, label_value in metric_labels.items():
                series.metric.labels[label_key] = label_value
            for label_key, label_value in self._extra_labels.items():
                if label_key in reserved_resource_keys:
                    continue
                series.metric.labels[label_key] = label_value

            point = types.Point()
            point.interval.end_time = now_ts
            point.value.double_value = value
            series.points.append(point)
            series_list.append(series)

        if series_list and self._client is not None:
            self._client.create_time_series(self._project_path, series_list)

    async def stop(self) -> None:
        """Stop the export loop and release resources."""
        self._running = False
        if self._task is not None:
            if self._task in self._background_tasks:
                await cancel_and_drain(self._background_tasks)
            elif not self._task.done():
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
            self._background_tasks.discard(self._task)
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
            except _TRANSIENT_MONITORING_EXCEPTIONS:
                _log.warning("cloud_metrics_loop_error", transient=True, exc_info=True)
            except Exception:
                _log.warning("cloud_metrics_loop_error", transient=False, exc_info=True)
