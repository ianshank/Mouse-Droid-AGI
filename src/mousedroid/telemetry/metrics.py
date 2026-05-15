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


class _DoubleLabeledCounter:
    """Counter keyed by a pair of string label values (e.g. tool, result)."""

    __slots__ = ("_lock", "_values")

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def inc(self, label_a: str, label_b: str, amount: int = 1) -> None:
        with self._lock:
            key = (label_a, label_b)
            self._values[key] = self._values.get(key, 0) + amount

    def snapshot(self) -> dict[tuple[str, str], int]:
        with self._lock:
            return dict(self._values)

    def reset(self) -> None:
        with self._lock:
            self._values.clear()


class _TripleLabeledCounter:
    """Counter keyed by a triple of string label values (e.g. subsystem, reason, level)."""

    __slots__ = ("_lock", "_values")

    def __init__(self) -> None:
        self._values: dict[tuple[str, str, str], int] = {}
        self._lock = threading.Lock()

    def inc(self, label_a: str, label_b: str, label_c: str, amount: int = 1) -> None:
        with self._lock:
            key = (label_a, label_b, label_c)
            self._values[key] = self._values.get(key, 0) + amount

    def snapshot(self) -> dict[tuple[str, str, str], int]:
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


class _LabeledGauge:
    """Gauge with a single string label dimension."""

    __slots__ = ("_lock", "_values")

    def __init__(self) -> None:
        self._values: dict[str, float] = {}
        self._lock = threading.Lock()

    def set(self, label: str, value: float) -> None:
        with self._lock:
            self._values[label] = value

    def set_many(self, values: dict[str, float]) -> None:
        """Replace all label → value entries atomically."""
        with self._lock:
            self._values = dict(values)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._values)

    def reset(self) -> None:
        with self._lock:
            self._values.clear()


class _DoubleLabeledGauge:
    """Gauge keyed by two string label values (e.g. sensor, state)."""

    __slots__ = ("_lock", "_values")

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def set(self, label_a: str, label_b: str, value: float) -> None:
        """Set the gauge for the (label_a, label_b) combination."""
        with self._lock:
            self._values[(label_a, label_b)] = value

    def snapshot(self) -> dict[tuple[str, str], float]:
        """Return a copy of the current label → value map."""
        with self._lock:
            return dict(self._values)

    def reset(self) -> None:
        """Clear every entry."""
        with self._lock:
            self._values.clear()


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


def _render_double_labeled_counter(
    name: str,
    help_text: str,
    label_a: str,
    label_b: str,
    values: dict[tuple[str, str], int],
) -> list[str]:
    lines = [
        f"# HELP {name}_total {_escape_help_text(help_text)}",
        f"# TYPE {name}_total counter",
    ]
    for (val_a, val_b), count in sorted(values.items()):
        ea = _escape_label_value(val_a)
        eb = _escape_label_value(val_b)
        lines.append(f'{name}_total{{{label_a}="{ea}",{label_b}="{eb}"}} {count}')
    return lines


def _render_triple_labeled_counter(
    name: str,
    help_text: str,
    label_a: str,
    label_b: str,
    label_c: str,
    values: dict[tuple[str, str, str], int],
) -> list[str]:
    lines = [
        f"# HELP {name}_total {_escape_help_text(help_text)}",
        f"# TYPE {name}_total counter",
    ]
    for (val_a, val_b, val_c), count in sorted(values.items()):
        ea = _escape_label_value(val_a)
        eb = _escape_label_value(val_b)
        ec = _escape_label_value(val_c)
        lines.append(f'{name}_total{{{label_a}="{ea}",{label_b}="{eb}",{label_c}="{ec}"}} {count}')
    return lines


def _render_gauge(name: str, help_text: str, value: float) -> list[str]:
    lines = [
        f"# HELP {name} {_escape_help_text(help_text)}",
        f"# TYPE {name} gauge",
        f"{name} {_fmt_float(value)}",
    ]
    return lines


def _render_double_labeled_gauge(
    name: str,
    help_text: str,
    label_a: str,
    label_b: str,
    values: dict[tuple[str, str], float],
) -> list[str]:
    """Render a two-label gauge family in Prometheus text exposition format."""
    lines = [
        f"# HELP {name} {_escape_help_text(help_text)}",
        f"# TYPE {name} gauge",
    ]
    for (val_a, val_b), gauge_value in sorted(values.items()):
        ea = _escape_label_value(val_a)
        eb = _escape_label_value(val_b)
        lines.append(f'{name}{{{label_a}="{ea}",{label_b}="{eb}"}} {_fmt_float(gauge_value)}')
    return lines


def _render_labeled_gauge(
    name: str,
    help_text: str,
    label_name: str,
    values: dict[str, float],
) -> list[str]:
    lines = [
        f"# HELP {name} {_escape_help_text(help_text)}",
        f"# TYPE {name} gauge",
    ]
    # Sort numerically when labels are numeric strings so scrape output is
    # stable for sector="0", "1", ..., "10" ordering.
    try:
        ordered = sorted(values.items(), key=lambda kv: (int(kv[0]), kv[0]))
    except ValueError:
        ordered = sorted(values.items())
    for label_val, gauge_value in ordered:
        escaped = _escape_label_value(label_val)
        lines.append(f'{name}{{{label_name}="{escaped}"}} {_fmt_float(gauge_value)}')
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
        self._llm_translation_results = _LabeledCounter()
        self._llm_requests = _Counter()
        self._sensor_recoveries = _Counter()
        self._sensor_recovery_failures = _Counter()
        # Cloud Digital Twin — per-sink (telemetry/experience) result counters
        self._cloud_telemetry_publish = _LabeledCounter()
        self._cloud_experience_publish = _LabeledCounter()
        self._cloud_experience_export_records = _LabeledCounter()

        # Gauges
        self._loop_time_ms = _Gauge()
        self._battery_v = _Gauge()
        self._ws_clients = _Gauge()
        self._gpu_temp_c = _Gauge()
        self._publish_hz = _Gauge()
        self._lidar_sector_distance_m = _LabeledGauge()
        self._lidar_min_distance_m = _Gauge()
        self._lidar_scan_points = _Gauge()
        self._episodic_size = _Gauge()
        self._semantic_size = _Gauge()
        self._working_size = _Gauge()
        self._llm_latency_ms = _Gauge()
        self._curiosity_reward = _Gauge()

        # Cloud Digital Twin gauges — breaker state per breaker, export backlog
        self._cloud_circuit_state = _LabeledGauge()
        self._cloud_experience_hwm_lag = _Gauge()
        self._cloud_experience_queue_depth = _Gauge()

        # Labeled counters (Phase 7)
        self._voice_events = _LabeledCounter()

        # Cross-cutting subsystem failure counter (FailureRecorder)
        self._subsystem_failures = _TripleLabeledCounter()

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

        # Histogram (loop latency) — sort and guarantee +Inf sentinel
        raw_buckets = sorted(cfg.loop_latency_buckets_ms)
        if not raw_buckets or raw_buckets[-1] != float("inf"):
            raw_buckets.append(float("inf"))
        self._loop_histogram = _Histogram(tuple(raw_buckets))

        # LLM latency histogram — sort and guarantee +Inf sentinel
        llm_buckets = sorted(cfg.llm_latency_buckets_ms)
        if not llm_buckets or llm_buckets[-1] != float("inf"):
            llm_buckets.append(float("inf"))
        self._llm_translation_latency_ms = _Histogram(tuple(llm_buckets))

        # Cloud publish latency histograms - reuse LLM bucket layout by
        # default; both telemetry and experience publishes fall in the
        # 25 ms - 2 s envelope.
        cloud_buckets = list(llm_buckets)
        self._cloud_telemetry_publish_latency_ms = _Histogram(tuple(cloud_buckets))
        self._cloud_experience_publish_latency_ms = _Histogram(tuple(cloud_buckets))

        # MCP server metrics
        self._mcp_requests = _Counter()
        self._mcp_tool_calls = _DoubleLabeledCounter()
        mcp_raw_buckets = sorted(cfg.mcp_latency_buckets_ms)
        if not mcp_raw_buckets or mcp_raw_buckets[-1] != float("inf"):
            mcp_raw_buckets.append(float("inf"))
        self._mcp_request_latency_ms = _Histogram(tuple(mcp_raw_buckets))

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
        self._name_llm_translation = f"{ns}_llm_translation"
        self._name_llm_translation_latency = f"{ns}_llm_translation_latency_ms"
        self._name_lidar_sector_distance = f"{ns}_lidar_sector_distance_m"
        self._name_lidar_min_distance = f"{ns}_lidar_min_distance_m"
        self._name_lidar_scan_points = f"{ns}_lidar_scan_points"
        self._name_episodic_size = f"{ns}_memory_episodic_size"
        self._name_semantic_size = f"{ns}_memory_semantic_size"
        self._name_working_size = f"{ns}_memory_working_size"
        self._name_voice_events = f"{ns}_voice_events"
        self._name_subsystem_failures = f"{ns}_subsystem_failures"
        # PR #4 metric names
        self._name_sensor_liveness = f"{ns}_telemetry_sensor_liveness"
        self._name_mdns_registered = f"{ns}_telemetry_mdns_registered"
        self._name_bound_port = f"{ns}_telemetry_bound_port"
        self._name_lidar_raw_published = f"{ns}_telemetry_lidar_raw_published_total"
        self._name_lidar_raw_dropped = f"{ns}_telemetry_lidar_raw_dropped_total"
        self._name_llm_latency_ms = f"{ns}_llm_latency_ms"
        self._name_llm_requests = f"{ns}_llm_requests"
        self._name_curiosity_reward = f"{ns}_curiosity_intrinsic_reward"
        self._name_sensor_recoveries = f"{ns}_sensor_recoveries"
        self._name_sensor_recovery_failures = f"{ns}_sensor_recovery_failures"

        # MCP metric names — all derived from namespace
        self._name_mcp_requests = f"{ns}_mcp_requests"
        self._name_mcp_tool_calls = f"{ns}_mcp_tool_calls"
        self._name_mcp_request_latency = f"{ns}_mcp_request_latency_ms"

        # Cloud Digital Twin metric names — all derived from namespace
        self._name_cloud_telemetry_publish = f"{ns}_cloud_telemetry_publish"
        self._name_cloud_experience_publish = f"{ns}_cloud_experience_publish"
        self._name_cloud_telemetry_publish_latency = f"{ns}_cloud_telemetry_publish_latency_ms"
        self._name_cloud_experience_publish_latency = f"{ns}_cloud_experience_publish_latency_ms"
        self._name_cloud_circuit_state = f"{ns}_cloud_circuit_state"
        self._name_cloud_experience_export_records = f"{ns}_cloud_experience_export_records"
        self._name_cloud_experience_hwm_lag = f"{ns}_cloud_experience_hwm_lag"
        self._name_cloud_experience_queue_depth = f"{ns}_cloud_experience_queue_depth"

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

    def inc_llm_translation(self, result: str) -> None:
        """Increment the LLM translation counter for the given result label."""
        if self._cfg.track_llm_translations:
            self._llm_translation_results.inc(result)

    def observe_llm_translation_latency_ms(self, value: float) -> None:
        """Record LLM translation latency in milliseconds."""
        if self._cfg.track_llm_translations:
            self._llm_translation_latency_ms.observe(value)

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

    def set_episodic_size(self, size: int) -> None:
        """Set the current episodic replay buffer size."""
        if self._cfg.track_memory_tier:
            self._episodic_size.set(float(size))

    def set_semantic_size(self, size: int) -> None:
        """Set the current semantic index size."""
        if self._cfg.track_memory_tier:
            self._semantic_size.set(float(size))

    def set_working_size(self, size: int) -> None:
        """Set the current working memory buffer size."""
        if self._cfg.track_memory_tier:
            self._working_size.set(float(size))

    def inc_voice_event(self, event_type: str) -> None:
        """Increment the voice event counter for a given event type.

        Args:
            event_type: Event label (e.g. ``"startup"``, ``"emergency_stop"``).
        """
        if self._cfg.track_voice_events:
            self._voice_events.inc(event_type)

    def inc_subsystem_failure(
        self,
        subsystem: str,
        reason: str,
        level: str = "warning",
        amount: int = 1,
    ) -> None:
        """Increment the cross-cutting subsystem failure counter.

        Always recorded regardless of config toggles — failure observability
        should never be silenced by configuration.

        Args:
            subsystem: Logical subsystem name (e.g. ``"voice"``, ``"telemetry"``).
            reason: Machine-readable failure reason (e.g. ``"device_disconnected"``).
            level: Severity level string (``"warning"``, ``"error"``, ``"critical"``).
            amount: Increment amount (default 1).
        """
        self._subsystem_failures.inc(subsystem, reason, level, amount)

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

    def set_llm_latency_ms(self, value: float) -> None:
        """Set the last LLM mission parse latency in milliseconds."""
        if self._cfg.track_llm_latency:
            self._llm_latency_ms.set(value)
            self._llm_requests.inc()

    def set_curiosity_reward(self, value: float) -> None:
        """Set the latest intrinsic curiosity reward."""
        if self._cfg.track_curiosity:
            self._curiosity_reward.set(value)

    def inc_sensor_recoveries(self, amount: int = 1) -> None:
        """Increment successful sensor recovery counter."""
        if self._cfg.track_sensor_recovery:
            self._sensor_recoveries.inc(amount)

    def inc_sensor_recovery_failures(self, amount: int = 1) -> None:
        """Increment failed sensor recovery counter."""
        if self._cfg.track_sensor_recovery:
            self._sensor_recovery_failures.inc(amount)

    # ------------------------------------------------------------------
    # Cloud Digital Twin helpers
    # ------------------------------------------------------------------

    def inc_cloud_telemetry_publish(self, result: str, amount: int = 1) -> None:
        """Increment cloud telemetry publish counter for a result label.

        Args:
            result: Outcome label (e.g. ``"success"``, ``"error"``,
                ``"circuit_open"``, ``"retry_exhausted"``).
            amount: Increment amount (default 1).
        """
        if self._cfg.track_cloud:
            self._cloud_telemetry_publish.inc(result, amount)

    def inc_cloud_experience_publish(self, result: str, amount: int = 1) -> None:
        """Increment cloud experience publish counter for a result label."""
        if self._cfg.track_cloud:
            self._cloud_experience_publish.inc(result, amount)

    def observe_cloud_telemetry_publish_latency_ms(self, value: float) -> None:
        """Record cloud telemetry publish latency in milliseconds."""
        if self._cfg.track_cloud:
            self._cloud_telemetry_publish_latency_ms.observe(value)

    def observe_cloud_experience_publish_latency_ms(self, value: float) -> None:
        """Record cloud experience publish latency in milliseconds."""
        if self._cfg.track_cloud:
            self._cloud_experience_publish_latency_ms.observe(value)

    def set_cloud_circuit_state(self, breaker: str, state: str) -> None:
        """Record current circuit breaker state as a numeric gauge.

        Gauge encoding: ``0`` = CLOSED, ``1`` = HALF_OPEN, ``2`` = OPEN.
        Unknown states default to ``-1``. The mapping is intentionally
        not config-driven because Grafana dashboards rely on these
        numeric values.

        Args:
            breaker: Circuit breaker name (e.g. ``"cloud_telemetry"``).
            state: Lowercased state string from :class:`CircuitState`.
        """
        if not self._cfg.track_cloud:
            return
        encoded: dict[str, float] = {
            "closed": 0.0,
            "half_open": 1.0,
            "open": 2.0,
        }
        self._cloud_circuit_state.set(breaker, encoded.get(state, -1.0))

    def inc_cloud_experience_export_records(self, result: str, amount: int) -> None:
        """Increment experience-records-exported counter.

        Args:
            result: Outcome label (``"success"``, ``"error"``).
            amount: Number of records successfully/failed in this batch.
        """
        if self._cfg.track_cloud and amount > 0:
            self._cloud_experience_export_records.inc(result, amount)

    def set_cloud_experience_hwm_lag(self, lag_records: int) -> None:
        """Set how many records remain between current HWM and DB tip."""
        if self._cfg.track_cloud:
            self._cloud_experience_hwm_lag.set(float(lag_records))

    def set_cloud_experience_queue_depth(self, depth: int) -> None:
        """Set current in-memory experience queue depth."""
        if self._cfg.track_cloud:
            self._cloud_experience_queue_depth.set(float(depth))

    # ------------------------------------------------------------------
    # MCP server helpers
    # ------------------------------------------------------------------

    def inc_mcp_request(self, amount: int = 1) -> None:
        """Increment the total MCP request counter (any kind of request)."""
        if self._cfg.track_mcp:
            self._mcp_requests.inc(amount)

    def inc_mcp_tool_call(self, tool: str, result: str, amount: int = 1) -> None:
        """Increment the per-tool MCP call counter.

        Args:
            tool: Tool name (e.g. ``"health_check"``).
            result: Outcome label (e.g. ``"ok"``, ``"refused_emergency"``,
                ``"denied"``, ``"rate_limited"``, ``"timeout"``,
                ``"error"``, ``"client_disconnected"``).
            amount: Increment amount (default 1).
        """
        if self._cfg.track_mcp:
            self._mcp_tool_calls.inc(tool, result, amount)

    def observe_mcp_request_latency_ms(self, value: float) -> None:
        """Record total MCP request latency in milliseconds."""
        if self._cfg.track_mcp:
            self._mcp_request_latency_ms.observe(value)

    @staticmethod
    def _decode_cloud_circuit_state(value: float) -> str:
        """Map numeric breaker gauge values back to symbolic states."""
        if value == 0.0:
            return "closed"
        if value == 1.0:
            return "half_open"
        if value == 2.0:
            return "open"
        return "unknown"

    def get_cloud_health_snapshot(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot of cloud health metrics.

        The telemetry server uses this to expose ``/api/v1/health/cloud``
        without coupling to concrete cloud sink/exporter implementations.
        """
        if not self._cfg.track_cloud:
            return {"enabled": False, "status": "disabled"}

        breaker_states = {
            breaker: self._decode_cloud_circuit_state(encoded)
            for breaker, encoded in self._cloud_circuit_state.snapshot().items()
        }
        queue_depth = int(self._cloud_experience_queue_depth.value)
        hwm_lag = int(self._cloud_experience_hwm_lag.value)
        status = "ok"
        if any(state == "open" for state in breaker_states.values()):
            status = "degraded"
        elif queue_depth > 0 or hwm_lag > 0:
            status = "backlogged"

        return {
            "enabled": True,
            "status": status,
            "breaker_states": breaker_states,
            "queue_depth": queue_depth,
            "hwm_lag": hwm_lag,
            "telemetry_publish": self._cloud_telemetry_publish.snapshot(),
            "experience_publish": self._cloud_experience_publish.snapshot(),
            "experience_export_records": self._cloud_experience_export_records.snapshot(),
        }

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

        if cfg.track_llm_translations:
            llm_results = self._llm_translation_results.snapshot()
            if llm_results:
                sections.append(
                    _render_labeled_counter(
                        self._name_llm_translation,
                        "LLM translation results (label: result)",
                        "result",
                        llm_results,
                    )
                )
            llm_buckets, llm_sum, llm_count = self._llm_translation_latency_ms.snapshot()
            sections.append(
                _render_histogram(
                    self._name_llm_translation_latency,
                    "LLM translation latency histogram (milliseconds)",
                    llm_buckets,
                    llm_sum,
                    llm_count,
                )
            )

        if cfg.track_lidar:
            sector_snapshot = self._lidar_sector_distance_m.snapshot()
            if sector_snapshot:
                sections.append(
                    _render_labeled_gauge(
                        self._name_lidar_sector_distance,
                        "Per-sector LiDAR distance in metres (label: sector)",
                        "sector",
                        sector_snapshot,
                    )
                )
            sections.append(
                _render_gauge(
                    self._name_lidar_min_distance,
                    "Minimum LiDAR distance across all sectors (metres)",
                    self._lidar_min_distance_m.value,
                )
            )
            sections.append(
                _render_gauge(
                    self._name_lidar_scan_points,
                    "Number of raw points in the last LiDAR scan",
                    self._lidar_scan_points.value,
                )
            )

        # Phase 7 metrics — memory, voice, LLM, curiosity, recovery
        if cfg.track_memory_tier:
            sections.append(
                _render_gauge(
                    self._name_episodic_size,
                    "Episodic replay buffer size",
                    self._episodic_size.value,
                )
            )
            sections.append(
                _render_gauge(
                    self._name_semantic_size,
                    "Semantic index size",
                    self._semantic_size.value,
                )
            )
            sections.append(
                _render_gauge(
                    self._name_working_size,
                    "Working memory buffer size",
                    self._working_size.value,
                )
            )

        if cfg.track_voice_events:
            voice_snapshot = self._voice_events.snapshot()
            if voice_snapshot:
                sections.append(
                    _render_labeled_counter(
                        self._name_voice_events,
                        "Voice events triggered (label: event_type)",
                        "event_type",
                        voice_snapshot,
                    )
                )

        if cfg.track_llm_latency:
            sections.append(
                _render_gauge(
                    self._name_llm_latency_ms,
                    "Last LLM mission parse latency (milliseconds)",
                    self._llm_latency_ms.value,
                )
            )
            sections.append(
                _render_counter(
                    self._name_llm_requests,
                    "Total LLM mission parse requests",
                    self._llm_requests.value,
                )
            )

        if cfg.track_curiosity:
            sections.append(
                _render_gauge(
                    self._name_curiosity_reward,
                    "Latest intrinsic curiosity reward",
                    self._curiosity_reward.value,
                )
            )

        if cfg.track_sensor_recovery:
            sections.append(
                _render_counter(
                    self._name_sensor_recoveries,
                    "Total successful sensor recoveries",
                    self._sensor_recoveries.value,
                )
            )
            sections.append(
                _render_counter(
                    self._name_sensor_recovery_failures,
                    "Total failed sensor recovery attempts",
                    self._sensor_recovery_failures.value,
                )
            )

        if cfg.track_cloud:
            telemetry_counts = self._cloud_telemetry_publish.snapshot()
            if telemetry_counts:
                sections.append(
                    _render_labeled_counter(
                        self._name_cloud_telemetry_publish,
                        "Cloud telemetry publish outcomes (label: result)",
                        "result",
                        telemetry_counts,
                    )
                )
            experience_counts = self._cloud_experience_publish.snapshot()
            if experience_counts:
                sections.append(
                    _render_labeled_counter(
                        self._name_cloud_experience_publish,
                        "Cloud experience publish outcomes (label: result)",
                        "result",
                        experience_counts,
                    )
                )
            tel_buckets, tel_sum, tel_count = self._cloud_telemetry_publish_latency_ms.snapshot()
            if tel_count > 0:
                sections.append(
                    _render_histogram(
                        self._name_cloud_telemetry_publish_latency,
                        "Cloud telemetry publish latency (milliseconds)",
                        tel_buckets,
                        tel_sum,
                        tel_count,
                    )
                )
            exp_buckets, exp_sum, exp_count = self._cloud_experience_publish_latency_ms.snapshot()
            if exp_count > 0:
                sections.append(
                    _render_histogram(
                        self._name_cloud_experience_publish_latency,
                        "Cloud experience publish latency (milliseconds)",
                        exp_buckets,
                        exp_sum,
                        exp_count,
                    )
                )
            circuit_snapshot = self._cloud_circuit_state.snapshot()
            if circuit_snapshot:
                sections.append(
                    _render_labeled_gauge(
                        self._name_cloud_circuit_state,
                        ("Circuit breaker state (0=closed, 1=half_open, 2=open; label: breaker)"),
                        "breaker",
                        circuit_snapshot,
                    )
                )
            export_counts = self._cloud_experience_export_records.snapshot()
            if export_counts:
                sections.append(
                    _render_labeled_counter(
                        self._name_cloud_experience_export_records,
                        "Cloud experience records exported (label: result)",
                        "result",
                        export_counts,
                    )
                )
            sections.append(
                _render_gauge(
                    self._name_cloud_experience_hwm_lag,
                    "Experience records between LMDB HWM and tip",
                    self._cloud_experience_hwm_lag.value,
                )
            )
            sections.append(
                _render_gauge(
                    self._name_cloud_experience_queue_depth,
                    "Pending experience records awaiting cloud publish",
                    self._cloud_experience_queue_depth.value,
                )
            )

        if cfg.track_mcp:
            sections.append(
                _render_counter(
                    self._name_mcp_requests,
                    "Total MCP requests received",
                    self._mcp_requests.value,
                )
            )
            tool_call_snapshot = self._mcp_tool_calls.snapshot()
            if tool_call_snapshot:
                sections.append(
                    _render_double_labeled_counter(
                        self._name_mcp_tool_calls,
                        "MCP tool call outcomes (labels: tool, result)",
                        "tool",
                        "result",
                        tool_call_snapshot,
                    )
                )
            mcp_buckets, mcp_sum, mcp_count = self._mcp_request_latency_ms.snapshot()
            if mcp_count > 0:
                sections.append(
                    _render_histogram(
                        self._name_mcp_request_latency,
                        "MCP request latency histogram (milliseconds)",
                        mcp_buckets,
                        mcp_sum,
                        mcp_count,
                    )
                )

        # Subsystem failures — always emitted regardless of config toggles
        failure_snapshot = self._subsystem_failures.snapshot()
        if failure_snapshot:
            sections.append(
                _render_triple_labeled_counter(
                    self._name_subsystem_failures,
                    "Subsystem failure events (labels: subsystem, reason, level)",
                    "subsystem",
                    "reason",
                    "level",
                    failure_snapshot,
                )
            )

        sections.append(
            _render_gauge(
                self._name_publish_hz,
                "Telemetry publisher rate (Hz)",
                self._publish_hz.value,
            )
        )

        # PR #4 — telemetry streaming + dashboard liveness metrics.
        # Emit conditionally so deployments that never touch the new
        # APIs don't produce noisy zero-valued series.
        liveness_snapshot = self._sensor_liveness.snapshot()
        if liveness_snapshot:
            sections.append(
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
            sections.append(
                _render_labeled_gauge(
                    self._name_mdns_registered,
                    "mDNS service registration indicator (1=registered, 0=failed)",
                    "service",
                    mdns_snapshot,
                )
            )
        if self._bound_port.value > 0:
            sections.append(
                _render_gauge(
                    self._name_bound_port,
                    "Actual TCP port the telemetry server bound to",
                    self._bound_port.value,
                )
            )
        if self._lidar_raw_published.value > 0 or self._lidar_raw_dropped.value > 0:
            sections.append(
                _render_counter(
                    self._name_lidar_raw_published.removesuffix("_total"),
                    "Raw LiDAR scans published to the streaming queue",
                    self._lidar_raw_published.value,
                )
            )
            sections.append(
                _render_counter(
                    self._name_lidar_raw_dropped.removesuffix("_total"),
                    "Raw LiDAR scans dropped (streaming queue full)",
                    self._lidar_raw_dropped.value,
                )
            )

        return "\n\n".join("\n".join(section) for section in sections) + "\n"


def generate_metrics_sample() -> str:
    """Generate a representative Prometheus metrics sample for CI validation.

    Creates a :class:`MetricsRegistry` with default config, populates every
    metric family with representative data, and returns the rendered
    Prometheus text exposition output.  Used by the CI ``promtool check
    metrics`` step to validate format compliance.

    Returns:
        Prometheus text exposition format string with all metric families.
    """
    from mousedroid.config.schema import MetricsConfig

    cfg = MetricsConfig.model_validate({})
    registry = MetricsRegistry(cfg)

    # Populate every metric family so all appear in the output.
    registry.set_loop_time_ms(15.0)
    registry.set_battery_voltage(11.8)
    registry.set_ws_client_count(2)
    registry.set_gpu_temp_celsius(52.0)
    registry.set_publish_hz(10.0)
    registry.inc_frame_drops(3)
    registry.inc_safety_violation("law1")
    registry.inc_llm_translation("translated")
    registry.observe_llm_translation_latency_ms(42.0)
    registry.set_lidar_sectors([0.9, 0.4, 1.0, 1.0, 0.7, 1.0, 1.0, 0.2], max_range_m=12.0)
    registry.set_lidar_min_distance_m(2.4)
    registry.set_lidar_scan_points(456)
    registry.inc_subsystem_failure("voice", "device_disconnected", "error")
    registry.inc_subsystem_failure("telemetry", "bind_exhausted", "warning")

    return registry.render_prometheus()
