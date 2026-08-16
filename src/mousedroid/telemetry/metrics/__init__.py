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

Package layout
--------------
This package replaces the former single ``metrics.py`` module. The public
call surface (every ``registry.inc_x/set_x/observe_x`` method, plus every
name previously importable from ``mousedroid.telemetry.metrics``) is
unchanged — only the internal file layout moved:

* ``primitives.py`` — the 8 Prometheus-primitive classes, the ``_render_*``
  rendering functions, and small formatting/constant helpers shared by every
  metric family.
* ``_registry_*.py`` — one mixin per metric family (core, phase7, lidar,
  streaming, cloud, mcp, llm_gateway, on_device_learning, growth, voice,
  replay_vla, mission_lifecycle). Each holds only its family's methods plus
  the ``_init_*_metrics`` helper that sets up its own attributes.
* ``registry.py`` — the concrete ``MetricsRegistry(*mixins)`` class (whose
  own body is just ``__init__`` orchestration + ``render_prometheus``
  orchestration, both genuinely cross-cutting) and ``generate_metrics_sample``.

Everything below is re-exported here so existing imports of
``mousedroid.telemetry.metrics`` — including the ``_``-prefixed primitives
and constants exercised directly by the test suite — keep working unchanged.
"""

from __future__ import annotations

from mousedroid.telemetry.metrics.primitives import (
    _GROWTH_DISTILL_OUTCOMES,
    _LLM_SERVED_OUTCOMES,
    _LLM_SERVED_TIERS,
    _LLM_TOKEN_TYPES,
    _MIN_OBSERVABLE_SECONDS,
    _ON_DEVICE_REVERT_REASONS,
    _VOICE_SPEAKER_DEGRADED_SUBSYSTEMS,
    _VOICE_TTS_APIS,
    _classify_dropped_observation,
    _Counter,
    _DoubleLabeledCounter,
    _DoubleLabeledGauge,
    _escape_help_text,
    _escape_label_value,
    _fmt_float,
    _Gauge,
    _Histogram,
    _LabeledCounter,
    _LabeledGauge,
    _log,
    _prepare_bucket_boundaries,
    _render_counter,
    _render_double_labeled_counter,
    _render_double_labeled_gauge,
    _render_gauge,
    _render_histogram,
    _render_labeled_counter,
    _render_labeled_gauge,
    _render_triple_labeled_counter,
    _TripleLabeledCounter,
)
from mousedroid.telemetry.metrics.registry import MetricsRegistry, generate_metrics_sample

__all__ = [
    "_GROWTH_DISTILL_OUTCOMES",
    "_LLM_SERVED_OUTCOMES",
    "_LLM_SERVED_TIERS",
    "_LLM_TOKEN_TYPES",
    "_MIN_OBSERVABLE_SECONDS",
    "_ON_DEVICE_REVERT_REASONS",
    "_VOICE_SPEAKER_DEGRADED_SUBSYSTEMS",
    "_VOICE_TTS_APIS",
    "MetricsRegistry",
    "_Counter",
    "_DoubleLabeledCounter",
    "_DoubleLabeledGauge",
    "_Gauge",
    "_Histogram",
    "_LabeledCounter",
    "_LabeledGauge",
    "_TripleLabeledCounter",
    "_classify_dropped_observation",
    "_escape_help_text",
    "_escape_label_value",
    "_fmt_float",
    "_log",
    "_prepare_bucket_boundaries",
    "_render_counter",
    "_render_double_labeled_counter",
    "_render_double_labeled_gauge",
    "_render_gauge",
    "_render_histogram",
    "_render_labeled_counter",
    "_render_labeled_gauge",
    "_render_triple_labeled_counter",
    "generate_metrics_sample",
]
