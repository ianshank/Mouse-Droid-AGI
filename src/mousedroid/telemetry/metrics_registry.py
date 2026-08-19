"""Prometheus metrics registry conforming to MetricsRegistryProtocol."""

from __future__ import annotations

import threading

from mousedroid.config.schema.telemetry import TelemetryConfig
from mousedroid.interfaces.protocols import MetricsRegistryProtocol
from mousedroid.logging.setup import get_logger

_log = get_logger("mousedroid.telemetry.metrics_registry")


class PrometheusMetricsRegistry(MetricsRegistryProtocol):
    """Zero-allocation Prometheus metrics registry implementing MetricsRegistryProtocol."""

    def __init__(self, cfg: TelemetryConfig | None = None) -> None:
        self._cfg = cfg or TelemetryConfig()
        self._counters: dict[str, dict[str, float]] = {}
        self._histograms: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def record_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Increment Prometheus counter."""
        label_key = self._format_labels(labels)
        with self._lock:
            if name not in self._counters:
                self._counters[name] = {}
            self._counters[name][label_key] = self._counters[name].get(label_key, 0.0) + value

    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record value in Prometheus histogram."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = []
            self._histograms[name].append(value)

    def _format_labels(self, labels: dict[str, str] | None) -> str:
        if not labels:
            return ""
        items = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(items) + "}"

    def render_prometheus(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            for name, labeled_vals in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                for labels, val in sorted(labeled_vals.items()):
                    lines.append(f"{name}{labels} {val}")

            for name, vals in sorted(self._histograms.items()):
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {len(vals)}")
                lines.append(f"{name}_sum {sum(vals)}")

        return "\n".join(lines) + "\n"
