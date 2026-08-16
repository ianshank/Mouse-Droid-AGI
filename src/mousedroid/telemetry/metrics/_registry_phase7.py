"""Phase-7 memory-tier, voice-event, LLM-latency, curiosity, and sensor-recovery metrics.

Mirrors the original code's own "Phase 7" grouping (memory / voice / LLM /
curiosity / recovery, shipped together in one PR) — kept as a single mixin
so the family's rendering order in ``render_prometheus`` stays untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.telemetry.metrics.primitives import (
    _Counter,
    _Gauge,
    _LabeledCounter,
    _render_counter,
    _render_gauge,
    _render_labeled_counter,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import MetricsConfig


class _Phase7MetricsMixin:
    """Memory-tier gauges, voice events, LLM latency, curiosity, sensor recovery."""

    # Populated by ``_CoreMetricsMixin._init_core_metrics``, which always runs
    # first from ``MetricsRegistry.__init__``.
    _cfg: MetricsConfig

    def _init_phase7_metrics(self, cfg: MetricsConfig) -> None:
        """Initialise Phase-7 memory / voice / LLM-latency / curiosity / recovery metrics.

        Args:
            cfg: Metrics configuration with namespace and toggle flags.
        """
        ns = cfg.namespace

        self._episodic_size = _Gauge()
        self._semantic_size = _Gauge()
        self._working_size = _Gauge()
        self._llm_latency_ms = _Gauge()
        self._curiosity_reward = _Gauge()

        # Labeled counters (Phase 7)
        self._voice_events = _LabeledCounter()

        self._llm_requests = _Counter()
        self._sensor_recoveries = _Counter()
        self._sensor_recovery_failures = _Counter()

        self._name_episodic_size = f"{ns}_memory_episodic_size"
        self._name_semantic_size = f"{ns}_memory_semantic_size"
        self._name_working_size = f"{ns}_memory_working_size"
        self._name_voice_events = f"{ns}_voice_events"
        self._name_llm_latency_ms = f"{ns}_llm_latency_ms"
        self._name_llm_requests = f"{ns}_llm_requests"
        self._name_curiosity_reward = f"{ns}_curiosity_intrinsic_reward"
        self._name_sensor_recoveries = f"{ns}_sensor_recoveries"
        self._name_sensor_recovery_failures = f"{ns}_sensor_recovery_failures"

    # ------------------------------------------------------------------
    # Write helpers — called by broadcast loop / orchestrator
    # ------------------------------------------------------------------

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
    # Prometheus text exposition — family renderer
    # ------------------------------------------------------------------

    def _families_phase7(self) -> list[list[str]]:
        """Phase-7 memory / voice / LLM-latency / curiosity / recovery families."""
        cfg = self._cfg
        out: list[list[str]] = []
        # Phase 7 metrics — memory, voice, LLM, curiosity, recovery
        if cfg.track_memory_tier:
            out.append(
                _render_gauge(
                    self._name_episodic_size,
                    "Episodic replay buffer size",
                    self._episodic_size.value,
                )
            )
            out.append(
                _render_gauge(
                    self._name_semantic_size,
                    "Semantic index size",
                    self._semantic_size.value,
                )
            )
            out.append(
                _render_gauge(
                    self._name_working_size,
                    "Working memory buffer size",
                    self._working_size.value,
                )
            )

        if cfg.track_voice_events:
            voice_snapshot = self._voice_events.snapshot()
            if voice_snapshot:
                out.append(
                    _render_labeled_counter(
                        self._name_voice_events,
                        "Voice events triggered (label: event_type)",
                        "event_type",
                        voice_snapshot,
                    )
                )

        if cfg.track_llm_latency:
            out.append(
                _render_gauge(
                    self._name_llm_latency_ms,
                    "Last LLM mission parse latency (milliseconds)",
                    self._llm_latency_ms.value,
                )
            )
            out.append(
                _render_counter(
                    self._name_llm_requests,
                    "Total LLM mission parse requests",
                    self._llm_requests.value,
                )
            )

        if cfg.track_curiosity:
            out.append(
                _render_gauge(
                    self._name_curiosity_reward,
                    "Latest intrinsic curiosity reward",
                    self._curiosity_reward.value,
                )
            )

        if cfg.track_sensor_recovery:
            out.append(
                _render_counter(
                    self._name_sensor_recoveries,
                    "Total successful sensor recoveries",
                    self._sensor_recoveries.value,
                )
            )
            out.append(
                _render_counter(
                    self._name_sensor_recovery_failures,
                    "Total failed sensor recovery attempts",
                    self._sensor_recovery_failures.value,
                )
            )
        return out
