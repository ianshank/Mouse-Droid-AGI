"""Core control-loop, safety, power, LLM-translation, and subsystem-failure metrics.

Holds the shared/generic + orchestrator-core slice of
:class:`~mousedroid.telemetry.metrics.registry.MetricsRegistry`: frame drops,
safety-law violations, control-loop latency, battery/GPU/websocket gauges,
the legacy rule-based LLM mission-translation counter/latency (predates the
Anthropic gateway tier — see ``_registry_llm_gateway.py`` for that), and the
cross-cutting subsystem-failure counter used by ``FailureRecorder``.

``_init_core_metrics`` is called first by
:meth:`MetricsRegistry.__init__ <mousedroid.telemetry.metrics.registry.MetricsRegistry.__init__>`
and establishes ``self._cfg`` — every other mixin declares ``_cfg`` as a
bare class-level annotation and relies on this method having already run.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from mousedroid.telemetry.metrics.primitives import (
    _Counter,
    _Gauge,
    _Histogram,
    _LabeledCounter,
    _prepare_bucket_boundaries,
    _render_counter,
    _render_gauge,
    _render_histogram,
    _render_labeled_counter,
    _render_triple_labeled_counter,
    _TripleLabeledCounter,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import MetricsConfig


class _CoreMetricsMixin:
    """Control-loop, safety, power, LLM-translation, and subsystem-failure metrics.

    Also hosts ``_prepare_bucket_boundaries`` as a class-level re-export of
    the shared pure function in ``primitives.py``, preserving the pre-split
    ``MetricsRegistry._prepare_bucket_boundaries(...)`` call surface pinned
    by ``tests/unit/telemetry/test_telemetry_metrics.py``.
    """

    # Re-exposed so ``MetricsRegistry._prepare_bucket_boundaries`` (and
    # ``registry._prepare_bucket_boundaries``) keep working unchanged. Every
    # mixin's own ``_init_*_metrics`` calls the module-level function
    # directly (not via ``self.``) to avoid a cross-mixin attribute lookup.
    _prepare_bucket_boundaries = staticmethod(_prepare_bucket_boundaries)

    def _init_core_metrics(self, cfg: MetricsConfig) -> None:
        """Initialise the config reference, start time, and core metric families.

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

        # Gauges
        self._loop_time_ms = _Gauge()
        self._battery_v = _Gauge()
        self._ws_clients = _Gauge()
        self._gpu_temp_c = _Gauge()
        self._publish_hz = _Gauge()

        # Cross-cutting subsystem failure counter (FailureRecorder)
        self._subsystem_failures = _TripleLabeledCounter()

        # Histogram families — bucket boundaries normalised via the shared
        # ``_prepare_bucket_boundaries`` helper (sort ascending + guarantee
        # the trailing ``+Inf`` sentinel Prometheus semantics require).
        self._loop_histogram = _Histogram(_prepare_bucket_boundaries(cfg.loop_latency_buckets_ms))
        llm_buckets = _prepare_bucket_boundaries(cfg.llm_latency_buckets_ms)
        self._llm_translation_latency_ms = _Histogram(llm_buckets)

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
        self._name_subsystem_failures = f"{ns}_subsystem_failures"

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
    # Prometheus text exposition — family renderers
    # ------------------------------------------------------------------

    def _families_core_loop(self) -> list[list[str]]:
        """Core control-loop, safety, and power metric families."""
        cfg = self._cfg
        out: list[list[str]] = []
        if cfg.track_frame_drops:
            out.append(
                _render_counter(
                    self._name_frame_drops,
                    "Telemetry frames dropped due to backpressure",
                    self._frame_drops.value,
                )
            )

        if cfg.track_safety_violations:
            violations = self._safety_violations.snapshot()
            if violations:
                out.append(
                    _render_labeled_counter(
                        self._name_safety_violations,
                        "Safety law violations (label: law)",
                        "law",
                        violations,
                    )
                )

        if cfg.track_loop_time:
            out.append(
                _render_gauge(
                    self._name_loop_time_ms,
                    "Last control-loop iteration time (milliseconds)",
                    self._loop_time_ms.value,
                )
            )
            buckets, hsum, hcount = self._loop_histogram.snapshot()
            out.append(
                _render_histogram(
                    self._name_loop_latency,
                    "Control-loop iteration latency histogram (milliseconds)",
                    buckets,
                    hsum,
                    hcount,
                )
            )

        if cfg.track_battery:
            out.append(
                _render_gauge(
                    self._name_battery_v,
                    "Battery voltage (volts)",
                    self._battery_v.value,
                )
            )

        if cfg.track_ws_clients:
            out.append(
                _render_gauge(
                    self._name_ws_clients,
                    "Number of currently connected WebSocket clients",
                    self._ws_clients.value,
                )
            )

        if cfg.track_gpu_temp:
            out.append(
                _render_gauge(
                    self._name_gpu_temp_c,
                    "GPU temperature (degrees Celsius)",
                    self._gpu_temp_c.value,
                )
            )
        return out

    def _families_llm_translation(self) -> list[list[str]]:
        """LLM mission-translation result + latency families."""
        cfg = self._cfg
        out: list[list[str]] = []
        if cfg.track_llm_translations:
            llm_results = self._llm_translation_results.snapshot()
            if llm_results:
                out.append(
                    _render_labeled_counter(
                        self._name_llm_translation,
                        "LLM translation results (label: result)",
                        "result",
                        llm_results,
                    )
                )
            llm_buckets, llm_sum, llm_count = self._llm_translation_latency_ms.snapshot()
            out.append(
                _render_histogram(
                    self._name_llm_translation_latency,
                    "LLM translation latency histogram (milliseconds)",
                    llm_buckets,
                    llm_sum,
                    llm_count,
                )
            )
        return out

    def _families_subsystem_failures(self) -> list[list[str]]:
        """Subsystem-failure counter (always emitted when non-empty)."""
        out: list[list[str]] = []
        # Subsystem failures — always emitted regardless of config toggles
        failure_snapshot = self._subsystem_failures.snapshot()
        if failure_snapshot:
            out.append(
                _render_triple_labeled_counter(
                    self._name_subsystem_failures,
                    "Subsystem failure events (labels: subsystem, reason, level)",
                    "subsystem",
                    "reason",
                    "level",
                    failure_snapshot,
                )
            )
        return out
