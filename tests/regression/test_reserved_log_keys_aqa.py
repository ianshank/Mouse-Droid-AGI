"""AQA pin — RESERVED_LOG_KEYS must stay in sync with the processor chain.

:data:`mousedroid.logging.setup.RESERVED_LOG_KEYS` is a hand-maintained
mirror of the keys claimed by the ``processors`` list in
:func:`~mousedroid.logging.setup.configure_logging` (plus ``event``, which the
bound-logger call signature itself owns).  Nothing structurally couples the two,
so adding a processor that writes a new key — ``CallsiteParameterAdder`` writes
``filename``/``lineno``/``func_name``, for instance — would silently reintroduce
the swallowed-field class of bug that ``safe_log_extra`` exists to prevent.

This test drives the *real* configured logger rather than
``structlog.testing.capture_logs`` (which installs its own short processor
chain and therefore cannot observe the production behaviour at all), and
asserts the set is not too small: every probe key that fails to round-trip
must already be declared reserved.
"""

from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Iterator

import pytest
import structlog
from structlog.contextvars import bind_contextvars

from mousedroid.config.schema import LoggingConfig, MetricsConfig
from mousedroid.logging.setup import RESERVED_LOG_KEYS, configure_logging, get_logger
from mousedroid.telemetry import failure_recorder
from mousedroid.telemetry.failure_recorder import PrometheusFailureRecorder
from mousedroid.telemetry.metrics import MetricsRegistry

_SENTINEL = "CALLER_SUPPLIED_VALUE"

# Keys worth probing: everything currently declared reserved, plus the names
# structlog processors that are *not* in the chain today would introduce.  The
# latter are the canaries — if someone enables one of those processors without
# updating RESERVED_LOG_KEYS, this test goes red.
_CANARY_KEYS = (
    "filename",
    "lineno",
    "func_name",
    "module",
    "pathname",
    "process",
    "process_name",
    "thread",
    "thread_name",
)
_PROBE_KEYS = tuple(sorted(RESERVED_LOG_KEYS)) + _CANARY_KEYS


@pytest.fixture
def _restore_structlog() -> Iterator[None]:
    """Snapshot and restore global structlog state around a test.

    ``configure_logging`` sets ``cache_logger_on_first_use=True``, so the first
    real log call through a module-level ``_log`` permanently replaces that lazy
    proxy with a bound logger that ignores every later ``structlog.configure``.
    Any test here that drives a production logger therefore silently breaks every
    subsequent ``structlog.testing.capture_logs`` assertion against that same
    module -- it captures nothing and ``logs[0]`` raises IndexError. Handing the
    module a fresh proxy on the way out undoes that.
    """
    snapshot = structlog.get_config().copy()
    try:
        yield
    finally:
        structlog.configure(**snapshot)
        structlog.contextvars.clear_contextvars()
        failure_recorder._log = structlog.get_logger(failure_recorder.__name__)


def _round_trips(key: str) -> bool:
    """Return True if a caller-supplied ``key`` survives to the rendered JSON."""
    log = get_logger(f"aqa_probe_{key}")
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            log.warning("aqa_probe_event", **{key: _SENTINEL})
    except TypeError:
        # ``event`` collides with the positional parameter — the hard failure.
        return False
    for line in buffer.getvalue().strip().splitlines():
        with contextlib.suppress(ValueError):
            if json.loads(line).get(key) == _SENTINEL:
                return True
    return False


@pytest.mark.usefixtures("_restore_structlog")
@pytest.mark.parametrize("key", _PROBE_KEYS)
def test_unsafe_keys_are_declared_reserved(key: str) -> None:
    """Any key the live processor chain swallows or rejects must be reserved."""
    configure_logging(LoggingConfig(level="DEBUG", format="json"))

    if _round_trips(key):
        pytest.skip(f"{key!r} round-trips under the current processor chain")

    assert key in RESERVED_LOG_KEYS, (
        f"{key!r} does not survive the configured structlog processor chain but "
        "is not listed in RESERVED_LOG_KEYS — add it there so safe_log_extra "
        "namespaces it instead of letting callers lose the field."
    )


@pytest.mark.usefixtures("_restore_structlog")
def test_event_key_is_the_hard_failure() -> None:
    """``event`` must raise TypeError when splatted — the crash this guards."""
    configure_logging(LoggingConfig(level="DEBUG", format="json"))
    log = get_logger("aqa_probe_event_collision")

    with pytest.raises(TypeError, match="multiple values for argument 'event'"):
        log.warning("aqa_probe_event", **{"event": _SENTINEL})


def test_reserved_set_is_a_frozenset() -> None:
    """The constant must not be mutable module state."""
    assert isinstance(RESERVED_LOG_KEYS, frozenset)


def test_event_is_always_reserved() -> None:
    """``event`` is reserved by the call signature, independent of processors."""
    assert "event" in RESERVED_LOG_KEYS


@pytest.mark.usefixtures("_restore_structlog")
def test_every_chain_written_key_is_declared_reserved() -> None:
    """Whatever the live chain writes unprompted must be declared reserved.

    Unlike the canary probe above, this needs no candidate list: it reads back
    the keys the configured processors and bound contextvars actually put in a
    rendered event, so a newly added processor or a newly bound contextvar is
    caught even when nobody thought to add it as a canary. This is the check
    that makes the constant's "adding a processor means adding that key here"
    comment enforceable rather than aspirational.
    """
    configure_logging(
        LoggingConfig(level="DEBUG", format="json"),
        robot_id="probe-robot",
    )
    # Mirror what orchestrator/mission_dispatcher.py binds per dispatch.
    bind_contextvars(trace_id="probe-trace", channel="probe-channel")
    log = get_logger("aqa_chain_written_probe")

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        log.warning("aqa_bare_probe")

    rendered: dict[str, object] = {}
    for line in buffer.getvalue().strip().splitlines():
        with contextlib.suppress(ValueError):
            candidate = json.loads(line)
            if candidate.get("event") == "aqa_bare_probe":
                rendered = candidate
                break

    assert rendered, "bare probe emitted no parseable JSON event"
    undeclared = set(rendered) - RESERVED_LOG_KEYS
    assert not undeclared, (
        f"the configured logging chain writes {sorted(undeclared)} unprompted, but "
        "those keys are absent from RESERVED_LOG_KEYS -- a caller passing one via "
        "`extra` would overwrite it. Add them to the constant in logging/setup.py."
    )


@pytest.mark.usefixtures("_restore_structlog")
def test_recorder_cannot_corrupt_bound_correlation_ids() -> None:
    """A caller ``extra`` must not replace robot_id / trace_id / channel.

    ``merge_contextvars`` is ``ctx.update(event_dict)``, so before these keys were
    reserved a caller-supplied value won outright: a subsystem failure logged
    during a mission dispatch could silently rewrite that dispatch's ``trace_id``,
    breaking the correlation the dispatcher binds it for. Drives the real
    recorder through the real chain rather than ``capture_logs``.
    """
    configure_logging(LoggingConfig(level="DEBUG", format="json"), robot_id="real-robot")
    bind_contextvars(trace_id="real-trace", channel="real-channel")
    recorder = PrometheusFailureRecorder(MetricsRegistry(MetricsConfig()))

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        recorder.record(
            "telemetry",
            "spoof_attempt",
            level="warning",
            extra={"robot_id": "spoof", "trace_id": "spoof", "channel": "spoof"},
        )

    rendered: dict[str, object] = {}
    for line in buffer.getvalue().strip().splitlines():
        with contextlib.suppress(ValueError):
            candidate = json.loads(line)
            if candidate.get("event") == "subsystem_failure_recorded":
                rendered = candidate
                break

    assert rendered, "recorder emitted no parseable JSON event"
    assert rendered["robot_id"] == "real-robot"
    assert rendered["trace_id"] == "real-trace"
    assert rendered["channel"] == "real-channel"
    # Lossless: the caller's values are still present, just namespaced.
    assert rendered["extra_robot_id"] == "spoof"
    assert rendered["extra_trace_id"] == "spoof"
    assert rendered["extra_channel"] == "spoof"
