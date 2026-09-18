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

from mousedroid.config.schema import LoggingConfig
from mousedroid.logging.setup import RESERVED_LOG_KEYS, configure_logging, get_logger

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
    """Snapshot and restore global structlog config around a test."""
    snapshot = structlog.get_config().copy()
    try:
        yield
    finally:
        structlog.configure(**snapshot)


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
