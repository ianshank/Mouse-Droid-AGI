"""Prometheus-compatible metrics registry for the telemetry server.

Implements the Prometheus text exposition format (version 0.0.4) without
any external dependency on ``prometheus_client``.  All metric names are
derived from :attr:`MetricsConfig.namespace` — nothing is hardcoded in
logic.

Supported metric types:

* ``Counter`` — monotonically increasing integer (resets to 0 on restart)
* ``Gauge``   — arbitrary float that can go up or down
* ``Histogram`` — bucket-based latency/distribution tracking

Thread / async safety
---------------------
Mutating operations are protected by per-metric ``threading.Lock`` instances
(``_Counter``, ``_Gauge``, ``_Histogram``, etc.) so updates from background
threads are safe. Scrapes read each metric independently; snapshots are
best-effort and may reflect in-flight changes across different metric families.

Usage::

    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.config.schema import MetricsConfig

    cfg = MetricsConfig()
    registry = MetricsRegistry(cfg)

    registry.inc_frame_drops()
    registry.set_loop_time_ms(18.5)
    text = registry.render_prometheus()
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import MetricsConfig

_log = get_logger(__name__)

# Default histogram buckets for latency in milliseconds (control-loop context)
_DEFAULT_LATENCY_BUCKETS_MS: tuple[float, ...] = (
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    33.0,
    50.0,
    100.0,
    200.0,
    float("inf"),
)


class _Counter:
    """Thread-safe Prometheus Counter (only increments)."""

    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._value: int = 0
        self._lock = threading.Lock()

    def inc(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset for testing only — not used in production."""
        with self._lock:
            self._value = 0


class _LabeledCounter:
    """Counter with a single string label dimension."""

    __slots__ = ("_lock", "_values")

    def __init__(self) -> None:
        self._values: dict[str, int] = {}
        self._lock = threading.Lock()

    def inc(self, label: str, amount: int = 1) -> None:
        with self._lock:
            self._values[label] = self._values.get(label, 0) + amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._values)

    def reset(self) -> None:
        with self._lock:
            self._values.clear()


class _Gauge:
    """Thread-safe Prometheus Gauge (set to any float)."""

    __slots__ = ("_lock", "_value")

    def __init__(self, initial: float = 0.0) -> None:
        self._value: float = initial
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class _Histogram:
    """Thread-safe Prometheus Histogram.

    Tracks sum, count, and per-bucket counts for distributions.
    """

    __slots__ = ("_buckets", "_count", "_lock", "_sum", "_thresholds")

    def __init__(self, buckets: tuple[float, ...]) -> None:
        self._thresholds: tuple[float, ...] = buckets
        # +Inf bucket is always last; pre-allocate all bucket counters
        self._buckets: list[int] = [0] * len(buckets)
        self._sum: float = 0.0
        self._count: int = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._count += 1
            for i, threshold in enumerate(self._thresholds):
                if value <= threshold:
                    self._buckets[i] += 1
                    break  # Each observation counted in exactly one bucket; snapshot accumulates

    def snapshot(self) -> tuple[list[tuple[float, int]], float, int]:
        """Return ``([(le, count), …], sum, count)`` under the lock."""
        with self._lock:
            cumulative = 0
            result: list[tuple[float, int]] = []
            for threshold, bucket_count in zip(self._thresholds, self._buckets, strict=True):
                cumulative += bucket_count
                result.append((threshold, cumulative))
            return result, self._sum, self._count


def _fmt_float(value: float) -> str:
    """Format a float for Prometheus text output with 6 significant digits."""
    if value == float("inf"):
        return "+Inf"
    if value != value:  # NaN
        return "NaN"
    return f"{value:.6g}"


def _escape_label_value(value: str) -> str:
    """Escape a label value for Prometheus text exposition format."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _escape_help_text(help_text: str) -> str:
    """Escape HELP text for Prometheus text exposition format."""
    return help_text.replace("\\", "\\\\").replace("\n", "\\n")


def _render_counter(name: str, help_text: str, value: int) -> list[str]:
    metric_name = f"{name}_total"
    lines = [
        f"# HELP {metric_name} {_escape_help_text(help_text)}",
        f"# TYPE {metric_name} counter",
        f"{metric_name} {value}",
    ]
    return lines


def _render_labeled_counter(
    name: str,
    help_text: str,
    label_name: str,
    values: dict[str, int],
) -> list[str]:
    lines = [
        f"# HELP {name}_total {_escape_help_text(help_text)}",
        f"# TYPE {name}_total counter",
    ]
    for label_val, count in sorted(values.items()):
        escaped_label_val = _escape_label_value(label_val)
        lines.append(f'{name}_total{{{label_name}="{escaped_label_val}"}} {count}')
    return lines


def _render_gauge(name: str, help_text: str, value: float) -> list[str]:
    lines = [
        f"# HELP {name} {_escape_help_text(help_text)}",
        f"# TYPE {name} gauge",
        f"{name} {_fmt_float(value)}",
    ]
    return lines


def _render_histogram(
    name: str,
    help_text: str,
    buckets: list[tuple[float, int]],
    total_sum: float,
    total_count: int,
) -> list[str]:
    lines = [
        f"# HELP {name} {_escape_help_text(help_text)}",
        f"# TYPE {name} histogram",
    ]
    for le, count in buckets:
        lines.append(f'{name}_bucket{{le="{_fmt_float(le)}"}} {count}')
    lines.append(f"{name}_sum {_fmt_float(total_sum)}")
    lines.append(f"{name}_count {total_count}")
    return lines


class MetricsRegistry:
    """Central registry for all MouseDroid Prometheus metrics.

    All metric names are derived from ``cfg.namespace`` so they can be
    overridden without code changes.  Individual metrics can be disabled
    via toggle flags in :class:`~mousedroid.config.schema.MetricsConfig`.

    Instantiate once and pass to :class:`TelemetryServer`.  Call the
    ``set_*`` / ``inc_*`` / ``observe_*`` helpers from the broadcast loop
    or orchestrator; call :meth:`render_prometheus` from the HTTP handler.
    """

    def __init__(self, cfg: MetricsConfig) -> None:
        """Initialise registry and all metric families.

        Args:
            cfg: Metrics configuration with namespace and toggle flags.
        """
        self._cfg = cfg
        ns = cfg.namespace
        self._start_time = time.monotonic()

        # Counters
        self._frame_drops = _Counter()
        self._safety_violations = _LabeledCounter()

        # Gauges
        self._loop_time_ms = _Gauge()
        self._battery_v = _Gauge()
        self._ws_clients = _Gauge()
        self._gpu_temp_c = _Gauge()
        self._publish_hz = _Gauge()

        # Histogram (loop latency)
        self._loop_histogram = _Histogram(_DEFAULT_LATENCY_BUCKETS_MS)

        # Pre-format metric names from namespace
        self._name_frame_drops = f"{ns}_frame_drops"
        self._name_safety_violations = f"{ns}_safety_violations"
        self._name_loop_time_ms = f"{ns}_loop_time_ms"
        self._name_loop_latency = f"{ns}_loop_latency_ms"
        self._name_battery_v = f"{ns}_battery_voltage_v"
        self._name_ws_clients = f"{ns}_ws_client_count"
        self._name_gpu_temp_c = f"{ns}_gpu_temp_celsius"
        self._name_publish_hz = f"{ns}_publish_hz"
        self._name_uptime = f"{ns}_uptime_seconds"

        _log.debug("metrics_registry_initialised", namespace=ns)

    # ------------------------------------------------------------------
    # Write helpers — called by broadcast loop / orchestrator
    # ------------------------------------------------------------------

    def inc_frame_drops(self, amount: int = 1) -> None:
        """Increment the telemetry frame-drop counter."""
        if self._cfg.track_frame_drops:
            self._frame_drops.inc(amount)

    def inc_safety_violation(self, law: str) -> None:
        """Increment the safety-violation counter for a given law label.

        Args:
            law: Label string identifying the violated law (e.g. ``"law1"``).
        """
        if self._cfg.track_safety_violations:
            self._safety_violations.inc(law)

    def set_loop_time_ms(self, value: float) -> None:
        """Set the current control-loop iteration time in milliseconds."""
        if self._cfg.track_loop_time:
            self._loop_time_ms.set(value)
            self._loop_histogram.observe(value)

    def set_battery_voltage(self, value: float) -> None:
        """Set the latest battery voltage in volts."""
        if self._cfg.track_battery:
            self._battery_v.set(value)

    def set_ws_client_count(self, count: int) -> None:
        """Set the number of currently connected WebSocket clients."""
        if self._cfg.track_ws_clients:
            self._ws_clients.set(float(count))

    def set_gpu_temp_celsius(self, value: float) -> None:
        """Set the latest GPU temperature in degrees Celsius."""
        if self._cfg.track_gpu_temp:
            self._gpu_temp_c.set(value)

    def set_publish_hz(self, value: float) -> None:
        """Set the current telemetry publish rate in Hz."""
        self._publish_hz.set(value)

    # ------------------------------------------------------------------
    # Read helpers (for testing / internal queries)
    # ------------------------------------------------------------------

    @property
    def frame_drops_total(self) -> int:
        """Total frame drops since startup."""
        return self._frame_drops.value

    @property
    def safety_violations(self) -> dict[str, int]:
        """Safety violation counts per law label."""
        return self._safety_violations.snapshot()

    # ------------------------------------------------------------------
    # Prometheus text exposition format
    # ------------------------------------------------------------------

    def render_prometheus(self) -> str:
        """Render all enabled metrics in Prometheus text format 0.0.4.

        Returns:
            Plain-text Prometheus scrape payload.  Each metric family is
            separated by a blank line.  The output ends with a trailing
            newline as required by the spec.
        """
        cfg = self._cfg
        sections: list[list[str]] = []

        # Uptime (always emitted — useful for detecting restarts)
        uptime = time.monotonic() - self._start_time
        sections.append(
            _render_gauge(self._name_uptime, "Seconds since metrics registry start", uptime)
        )

        if cfg.track_frame_drops:
            sections.append(
                _render_counter(
                    self._name_frame_drops,
                    "Telemetry frames dropped due to backpressure",
                    self._frame_drops.value,
                )
            )

        if cfg.track_safety_violations:
            violations = self._safety_violations.snapshot()
            if violations:
                sections.append(
                    _render_labeled_counter(
                        self._name_safety_violations,
                        "Safety law violations (label: law)",
                        "law",
                        violations,
                    )
                )

        if cfg.track_loop_time:
            sections.append(
                _render_gauge(
                    self._name_loop_time_ms,
                    "Last control-loop iteration time (milliseconds)",
                    self._loop_time_ms.value,
                )
            )
            buckets, hsum, hcount = self._loop_histogram.snapshot()
            sections.append(
                _render_histogram(
                    self._name_loop_latency,
                    "Control-loop iteration latency histogram (milliseconds)",
                    buckets,
                    hsum,
                    hcount,
                )
            )

        if cfg.track_battery:
            sections.append(
                _render_gauge(
                    self._name_battery_v,
                    "Battery voltage (volts)",
                    self._battery_v.value,
                )
            )

        if cfg.track_ws_clients:
            sections.append(
                _render_gauge(
                    self._name_ws_clients,
                    "Number of currently connected WebSocket clients",
                    self._ws_clients.value,
                )
            )

        if cfg.track_gpu_temp:
            sections.append(
                _render_gauge(
                    self._name_gpu_temp_c,
                    "GPU temperature (degrees Celsius)",
                    self._gpu_temp_c.value,
                )
            )

        sections.append(
            _render_gauge(
                self._name_publish_hz,
                "Telemetry publisher rate (Hz)",
                self._publish_hz.value,
            )
        )

        return "\n\n".join("\n".join(section) for section in sections) + "\n"
