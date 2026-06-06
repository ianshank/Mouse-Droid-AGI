"""Latency-sample summarisation for multi-iteration on-device probes.

Turns a list of per-iteration latency measurements (milliseconds) into a
typed :class:`LatencySummary` (min / mean / p50 / p95 / p99 / max). Shared by
the operator probes under ``tools/`` (``llm_latency_probe.py`` measures
``translate_mission`` round-trips; ``lidar_telemetry_probe.py`` measures
WebSocket frame inter-arrival jitter) so a single-shot presence check becomes
a latency-regression gate without duplicating the percentile maths.

Architecture invariants (per CLAUDE.md):

* Pure + deterministic — no I/O, no clock reads, no hidden state. Safe to call
  from any context. Identical input always yields identical output.
* No hardcoded thresholds — :func:`summarize` only *describes* a sample; the
  caller owns the pass/fail decision against its config-supplied target.
* Type-safe — ``mypy --strict`` clean; Pydantic model for JSON export to
  operator dashboards.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Definitional constants (NOT runtime-tunable): the percentile domain, the
# percentile points named in LatencySummary, and the seconds→ms unit factor.
# Named so the maths reads intentionally rather than via bare literals.
_PCT_MIN = 0.0
_PCT_MAX = 100.0  # hardcoded-ok: percentile domain upper bound
_P50, _P95, _P99 = 50.0, 95.0, 99.0  # hardcoded-ok: reported percentile points
_MS_PER_S = 1000.0  # hardcoded-ok: seconds -> milliseconds


class LatencySummary(BaseModel):
    """Percentile summary of a latency sample, in milliseconds.

    All fields are derived purely from the input sample; the model carries no
    pass/fail verdict (the probe gates on a config-supplied target).
    """

    n: int = Field(ge=1, description="Number of samples summarised.")
    min_ms: float = Field(ge=0.0)
    mean_ms: float = Field(ge=0.0)
    p50_ms: float = Field(ge=0.0, description="Median latency.")
    p95_ms: float = Field(ge=0.0)
    p99_ms: float = Field(ge=0.0)
    max_ms: float = Field(ge=0.0)

    def render_text(self) -> str:
        """Render a single-line operator summary."""
        return (
            f"n={self.n} min={self.min_ms:.1f} mean={self.mean_ms:.1f} "
            f"p50={self.p50_ms:.1f} p95={self.p95_ms:.1f} "
            f"p99={self.p99_ms:.1f} max={self.max_ms:.1f} (ms)"
        )


def percentile(sorted_samples: list[float], q: float) -> float:
    """Return the ``q``-th percentile of an already-sorted sample.

    Uses linear interpolation between closest ranks (the method NumPy applies
    by default), so the result is stable and matches operator expectations.

    Args:
        sorted_samples: Ascending-sorted, non-empty list of values.
        q: Percentile in ``[0, 100]``.

    Returns:
        The interpolated percentile value.

    Raises:
        ValueError: If ``sorted_samples`` is empty or ``q`` is out of range.
    """
    if not sorted_samples:
        raise ValueError("percentile() requires a non-empty sample")
    if not _PCT_MIN <= q <= _PCT_MAX:
        raise ValueError(f"percentile q must be in [{_PCT_MIN}, {_PCT_MAX}], got {q}")
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    rank = (q / _PCT_MAX) * (len(sorted_samples) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = rank - lo
    return sorted_samples[lo] + frac * (sorted_samples[hi] - sorted_samples[lo])


def summarize(samples_ms: list[float]) -> LatencySummary:
    """Summarise a latency sample into a :class:`LatencySummary`.

    Args:
        samples_ms: Non-empty list of per-iteration latencies in milliseconds.

    Returns:
        A :class:`LatencySummary` over the sample.

    Raises:
        ValueError: If ``samples_ms`` is empty.
    """
    if not samples_ms:
        raise ValueError("summarize() requires at least one sample")
    ordered = sorted(samples_ms)
    return LatencySummary(
        n=len(ordered),
        min_ms=ordered[0],
        mean_ms=sum(ordered) / len(ordered),
        p50_ms=percentile(ordered, _P50),
        p95_ms=percentile(ordered, _P95),
        p99_ms=percentile(ordered, _P99),
        max_ms=ordered[-1],
    )


def intervals_ms(timestamps_s: list[float]) -> list[float]:
    """Convert a sequence of arrival timestamps into inter-arrival gaps (ms).

    Pure helper shared by streaming probes (e.g. the LiDAR→WebSocket frame
    monitor): given monotonic arrival times in seconds, return the gaps between
    consecutive arrivals in milliseconds. A sequence with fewer than two
    timestamps yields an empty list (no interval can be formed) — never raises,
    so callers can summarise unconditionally.

    Args:
        timestamps_s: Arrival times in seconds (monotonic clock recommended).

    Returns:
        ``len(timestamps_s) - 1`` inter-arrival gaps in milliseconds, or an
        empty list when fewer than two timestamps are supplied.
    """
    return [
        (timestamps_s[i] - timestamps_s[i - 1]) * _MS_PER_S for i in range(1, len(timestamps_s))
    ]


__all__ = ["LatencySummary", "intervals_ms", "percentile", "summarize"]
