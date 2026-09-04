"""Prometheus-primitive types, rendering helpers, and shared constants.

Implements the Prometheus text exposition format (version 0.0.4) building
blocks without any external dependency on ``prometheus_client``: thread-safe
counter/gauge/histogram primitives, the ``_render_*`` functions that turn a
primitive's snapshot into exposition-format text lines, and the small
formatting/validation helpers shared across metric families.

Every :class:`~mousedroid.telemetry.metrics.registry.MetricsRegistry` mixin
imports from this module — it is the dependency-free foundation of the
``mousedroid.telemetry.metrics`` package, so it must never import a sibling
mixin module (that would create a cycle).

Thread / async safety
---------------------
Mutating operations are protected by per-metric ``threading.Lock`` instances
(``_Counter``, ``_Gauge``, ``_Histogram``, etc.) so updates from background
threads are safe. Scrapes read each metric independently; snapshots are
best-effort and may reflect in-flight changes across different metric families.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

# Defensive lower-bound for any latency-style histogram observation.
# Module-level so future histogram helpers can reference the same value
# (e.g. an upcoming planner-inference histogram) without each helper
# re-declaring its own threshold.
_MIN_OBSERVABLE_SECONDS: float = 0.0

# Fixed low-cardinality label value sets for the LLM-gateway families.
# Single source of truth so the writer helpers (runtime drop-guard) and the
# AQA label-hygiene tests reference the same sets — a forwarded SDK value or a
# future typo can never silently open a new Prometheus time series.
_LLM_TOKEN_TYPES: frozenset[str] = frozenset({"input", "output"})
_LLM_SERVED_TIERS: frozenset[str] = frozenset({"primary", "secondary"})
_LLM_SERVED_OUTCOMES: frozenset[str] = frozenset({"ok", "degraded"})

# Phase-6 on-device-learning revert reasons. Single source of truth for the
# writer drop-guard and the AQA label-hygiene test, so a free-text mission
# string can never leak into the counter's cardinality.
_ON_DEVICE_REVERT_REASONS: frozenset[str] = frozenset(
    {"regression_bound", "integrity_mismatch", "exception"}
)

# Growth-pillar distillation outcome label values. Same drop-guard + AQA-pinning
# role as the on-device set above, so a free-text string can never open a new
# time series on the growth counter.
_GROWTH_DISTILL_OUTCOMES: frozenset[str] = frozenset({"completed", "skipped_no_batch"})

# Voice-degradation label value sets. Same drop-guard + AQA-pinning role as the
# LLM/on-device sets above, so a driver-forwarded string can never open a new
# time series.
#   * subsystem — which voice component degraded: the USB speaker exhausted its
#     reconnect retries (``usb_speaker``) or the engine downgraded to a
#     MockSpeaker (``rocky_fallback``).
#   * api — the resolved Piper synthesis API whose call raised.
_VOICE_SPEAKER_DEGRADED_SUBSYSTEMS: frozenset[str] = frozenset({"usb_speaker", "rocky_fallback"})
_VOICE_TTS_APIS: frozenset[str] = frozenset({"synthesize", "synthesize_wav", "synthesize_wav_file"})
#: Runtime drop-guard for ``mousedroid_tick_phase_ms{phase}``. Mirrors
#: :data:`mousedroid.config.schema._primitives.TickPhaseLiteral`, which is the
#: compile-time half of the same guard — mypy rejects a mistyped phase at the
#: call site, and this rejects anything that reaches the registry dynamically.
#: A regression test asserts the two sets never diverge. Together they cap the
#: family at 8 label values, so cardinality cannot grow by accident.
_TICK_PHASES: frozenset[str] = frozenset(
    {"sense", "safety", "world_model", "plan", "act", "learn", "telemetry", "post"}
)


def _classify_dropped_observation(value: float) -> str | None:
    """Classify a histogram-observation candidate; return drop reason or ``None``.

    Centralises the defensive guards that every latency-histogram helper
    (``observe_vla_inference_seconds``, ``observe_world_model_observe_step_seconds``,
    and future siblings) must apply uniformly. Returning a reason string instead
    of a bool lets the caller emit the exact label that Grafana / log-aggregation
    queries depend on without each helper rewriting the same chain of conditions.

    Drop reasons (ordered by priority — NaN dominates because comparisons
    against NaN return ``False``, so the inf and negative checks would silently
    pass through it otherwise):

    * ``"nan"`` — ``value != value`` (canonical NaN check, no ``math`` import).
    * ``"inf"`` — ``value == float("inf")``. ``_Histogram`` routes ``+Inf``
      into the ``le=+Inf`` bucket without complaint, but the ``_sum`` accumulator
      would then go to ``+Inf`` forever, breaking every rate / quantile
      computation downstream. Negative infinity is caught by the next branch.
    * ``"negative"`` — ``value < _MIN_OBSERVABLE_SECONDS`` (clock skew, wrap,
      or division-by-zero producing ``-inf``).

    Args:
        value: Wall-clock seconds candidate from a ``time.perf_counter()``
            bracket. Untrusted — may be NaN / inf / negative.

    Returns:
        Drop-reason string for structured logging when the sample MUST be
        discarded, or ``None`` when the sample is safe to feed to
        :meth:`_Histogram.observe`.
    """
    if value != value:
        return "nan"
    if value == float("inf"):
        return "inf"
    if value < _MIN_OBSERVABLE_SECONDS:
        return "negative"
    return None


def _prepare_bucket_boundaries(raw_buckets: Sequence[float]) -> tuple[float, ...]:
    """Sort ``raw_buckets`` ascending and guarantee a trailing ``+Inf`` sentinel.

    Prometheus histogram semantics require the final bucket to be ``+Inf``
    — without it, samples above the largest finite bucket are silently
    dropped from the cumulative bucket counts (though they still update
    ``_sum`` / ``_count``, producing inconsistent rendered exposition).

    This helper centralises the boilerplate that previously appeared once
    per histogram family (loop / LLM / MCP / VLA inference / world-model
    observe_step). Adding a new histogram family is now: drop the bucket
    field on :class:`MetricsConfig`, register it in the validator, and
    call ``_prepare_bucket_boundaries(cfg.<field>)`` here.

    A pure module-level function (not a method) so every registry mixin can
    call it directly without a cross-mixin ``self.`` attribute lookup;
    :class:`~mousedroid.telemetry.metrics._registry_core._CoreMetricsMixin`
    re-exposes it as ``MetricsRegistry._prepare_bucket_boundaries`` for the
    pre-split call surface.

    Args:
        raw_buckets: Operator-configured bucket boundaries from
            :class:`MetricsConfig` (any order, ``+Inf`` optional).

    Returns:
        Sorted-ascending tuple of bucket boundaries with ``+Inf`` as
        the final element. Safe to feed directly to :class:`_Histogram`.
    """
    sorted_buckets = sorted(raw_buckets)
    if not sorted_buckets or sorted_buckets[-1] != float("inf"):
        sorted_buckets.append(float("inf"))
    return tuple(sorted_buckets)


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


class _LabeledHistogram:
    """Thread-safe Prometheus Histogram partitioned by a single label value.

    One independent bucket set per label value, materialised on first
    observation so a label that never fires contributes no series. Used for
    ``mousedroid_tick_phase_ms{phase}``, where the label domain is the fixed
    eight-phase set in :data:`_TICK_PHASES` — callers must drop unknown values
    *before* calling :meth:`observe`, which is what keeps cardinality bounded
    at ``len(domain) x len(buckets)`` rather than unbounded.

    Deliberately not a generalisation of :class:`_Histogram`: that class is on
    the hot path for every unlabelled latency metric and gains nothing from a
    per-observation dict lookup.
    """

    __slots__ = ("_buckets", "_counts", "_lock", "_sums", "_thresholds")

    def __init__(self, buckets: tuple[float, ...]) -> None:
        self._thresholds: tuple[float, ...] = buckets
        self._buckets: dict[str, list[int]] = {}
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def observe(self, label: str, value: float) -> None:
        """Record *value* against *label*, creating its bucket set on demand."""
        with self._lock:
            counters = self._buckets.get(label)
            if counters is None:
                counters = [0] * len(self._thresholds)
                self._buckets[label] = counters
                self._sums[label] = 0.0
                self._counts[label] = 0
            self._sums[label] += value
            self._counts[label] += 1
            for i, threshold in enumerate(self._thresholds):
                if value <= threshold:
                    counters[i] += 1
                    break

    def snapshot(self) -> dict[str, tuple[list[tuple[float, int]], float, int]]:
        """Return ``{label: ([(le, cumulative)…], sum, count)}`` under the lock."""
        with self._lock:
            out: dict[str, tuple[list[tuple[float, int]], float, int]] = {}
            for label, counters in self._buckets.items():
                cumulative = 0
                rows: list[tuple[float, int]] = []
                for threshold, bucket_count in zip(self._thresholds, counters, strict=True):
                    cumulative += bucket_count
                    rows.append((threshold, cumulative))
                out[label] = (rows, self._sums[label], self._counts[label])
            return out


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


def _render_labeled_histogram(
    name: str,
    help_text: str,
    label_name: str,
    snapshots: dict[str, tuple[list[tuple[float, int]], float, int]],
) -> list[str]:
    """Render a single-label histogram family.

    Label ordering matches ``_render_double_labeled_gauge``: the partitioning
    label first, then ``le``. Label values are sorted so the exposition output
    is deterministic across scrapes, which is what lets the golden-render
    regression fixture be a byte comparison.
    """
    lines = [
        f"# HELP {name} {_escape_help_text(help_text)}",
        f"# TYPE {name} histogram",
    ]
    for label_value in sorted(snapshots):
        buckets, total_sum, total_count = snapshots[label_value]
        escaped = _escape_label_value(label_value)
        for le, count in buckets:
            lines.append(f'{name}_bucket{{{label_name}="{escaped}",le="{_fmt_float(le)}"}} {count}')
        lines.append(f'{name}_sum{{{label_name}="{escaped}"}} {_fmt_float(total_sum)}')
        lines.append(f'{name}_count{{{label_name}="{escaped}"}} {total_count}')
    return lines
