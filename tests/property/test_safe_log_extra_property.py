"""Property tests for the reserved-log-key guard.

Invariant under test: for *any* caller-supplied ``extra`` mapping, splatting
the guarded result into a structlog bound logger neither raises nor loses a
value, and never overwrites a key the call site owns.
"""

from __future__ import annotations

import string

import structlog.testing
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import MetricsConfig
from mousedroid.logging.setup import RESERVED_LOG_KEYS, safe_log_extra
from mousedroid.telemetry.failure_recorder import PrometheusFailureRecorder
from mousedroid.telemetry.metrics import MetricsRegistry

_IDENTIFIER = st.text(
    alphabet=string.ascii_lowercase + string.digits + "_",
    min_size=1,
    max_size=24,
).filter(lambda s: s[0].isalpha())

# Bias generation hard toward the dangerous names so collisions actually occur.
_KEY = st.one_of(st.sampled_from(sorted(RESERVED_LOG_KEYS)), _IDENTIFIER)

_VALUE = st.one_of(
    st.text(max_size=32),
    st.integers(min_value=-10_000, max_value=10_000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
)

_EXTRA = st.dictionaries(keys=_KEY, values=_VALUE, max_size=12)


@given(extra=_EXTRA)
@settings(max_examples=300)
def test_output_is_always_splat_safe(extra: dict[str, object]) -> None:
    """No reserved key ever survives into the guarded mapping."""
    result = safe_log_extra(extra)

    assert RESERVED_LOG_KEYS.isdisjoint(result)


@given(extra=_EXTRA)
@settings(max_examples=300)
def test_no_value_is_ever_lost(extra: dict[str, object]) -> None:
    """Renaming preserves cardinality — guarding never drops a field."""
    result = safe_log_extra(extra)

    assert len(result) == len(extra)


@given(extra=_EXTRA, occupied=st.sets(_KEY, max_size=6))
@settings(max_examples=300)
def test_occupied_keys_are_never_overwritten(
    extra: dict[str, object],
    occupied: set[str],
) -> None:
    """A caller can never claim a key the call site already owns."""
    result = safe_log_extra(extra, occupied=occupied)

    assert occupied.isdisjoint(result)


@given(extra=_EXTRA)
@settings(max_examples=200, deadline=None)
def test_recorder_never_raises_and_keeps_its_own_fields(extra: dict[str, object]) -> None:
    """PrometheusFailureRecorder.record survives any extra mapping."""
    rec = PrometheusFailureRecorder(MetricsRegistry(MetricsConfig()))

    with structlog.testing.capture_logs() as logs:
        rec.record("voice", "prop_reason", level="error", extra=extra)  # type: ignore[arg-type]

    log = logs[0]
    assert log["event"] == "subsystem_failure_recorded"
    assert log["subsystem"] == "voice"
    assert log["reason"] == "prop_reason"
    assert log["log_level"] == "error"
