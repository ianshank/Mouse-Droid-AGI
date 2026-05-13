"""Property tests for FailureRecorder: arbitrary inputs produce valid counter state."""

from __future__ import annotations

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.failure_recorder import PrometheusFailureRecorder
from mousedroid.telemetry.metrics import MetricsRegistry

_SNAKE_CASE = st.text(
    alphabet=string.ascii_lowercase + string.digits + "_",
    min_size=1,
    max_size=32,
).filter(lambda s: s[0].isalpha())

_LEVEL = st.sampled_from(["warning", "error", "critical"])


@given(
    subsystem=_SNAKE_CASE,
    reason=_SNAKE_CASE,
    level=_LEVEL,
    count=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=200)
def test_counter_accumulates_without_collision(
    subsystem: str,
    reason: str,
    level: str,
    count: int,
) -> None:
    """Recording N failures for any (subsystem, reason, level) yields exactly N in counter."""
    registry = MetricsRegistry(MetricsConfig())
    rec = PrometheusFailureRecorder(registry)

    for _ in range(count):
        rec.record(subsystem, reason, level=level)  # type: ignore[arg-type]

    snapshot = registry._subsystem_failures.snapshot()
    assert snapshot[(subsystem, reason, level)] == count


@given(
    pairs=st.lists(
        st.tuples(_SNAKE_CASE, _SNAKE_CASE, _LEVEL),
        min_size=1,
        max_size=10,
        unique=True,
    ),
)
@settings(max_examples=100)
def test_distinct_labels_never_collide(
    pairs: list[tuple[str, str, str]],
) -> None:
    """Different (subsystem, reason, level) triples never share counter buckets."""
    registry = MetricsRegistry(MetricsConfig())
    rec = PrometheusFailureRecorder(registry)

    for subsystem, reason, level in pairs:
        rec.record(subsystem, reason, level=level)  # type: ignore[arg-type]

    snapshot = registry._subsystem_failures.snapshot()
    for key in snapshot:
        assert snapshot[key] == 1, f"Unexpected collision at {key}"


@given(
    subsystem=_SNAKE_CASE,
    reason=_SNAKE_CASE,
    level=_LEVEL,
)
@settings(max_examples=100)
def test_prometheus_output_contains_all_labels(
    subsystem: str,
    reason: str,
    level: str,
) -> None:
    """render_prometheus() output contains the recorded labels and value."""
    registry = MetricsRegistry(MetricsConfig())
    rec = PrometheusFailureRecorder(registry)

    rec.record(subsystem, reason, level=level)  # type: ignore[arg-type]

    output = registry.render_prometheus()
    assert "mousedroid_subsystem_failures_total" in output
    assert f'subsystem="{subsystem}"' in output
    assert f'reason="{reason}"' in output
    assert f'level="{level}"' in output
