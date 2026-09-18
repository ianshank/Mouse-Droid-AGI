"""Hypothesis properties for the graceful-shutdown surface (S-1).

Two input spaces worth sweeping rather than sampling:

* ``request_shutdown`` is called from a signal handler, so it must hold for
  *any* arrival pattern — SIGTERM alone, SIGINT behind SIGTERM, a burst of
  repeats from an impatient operator hammering ``docker stop``. The
  example-based tests cover a couple of those; the property covers the
  shape.
* ``shutdown_grace_s`` accepts any positive float, so its validator
  boundary is a range, not the two values a unit test picks.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from mousedroid.config.schema import LoopConfig
from mousedroid.orchestrator._lifecycle_mixin import _LifecycleMixin

_SIGNAL_NAMES = st.sampled_from(["SIGTERM", "SIGINT"])


class _ShutdownFlag(_LifecycleMixin):
    """Minimal subclass carrying only what ``request_shutdown`` touches.

    ``request_shutdown`` touches only ``self._running`` and
    ``self._shutdown_requested``, so inheriting it exercises the real method — not a
    reimplementation — without building a whole orchestrator per Hypothesis
    example. Inherited rather than rebound onto a plain class, so nothing
    is mutated between examples.
    """

    def __init__(self, *, running: bool) -> None:
        self._running = running
        # The S-1 latch ``request_shutdown`` reads to decide whether this is
        # a first request or a repeat.
        self._shutdown_requested = False


@given(reasons=st.lists(_SIGNAL_NAMES, min_size=1, max_size=8))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_any_signal_sequence_ends_stopped(reasons: list[str]) -> None:
    """However many signals arrive, in any order, the loop ends up stopped."""
    flag = _ShutdownFlag(running=True)

    for reason in reasons:
        flag.request_shutdown(reason)

    assert flag._running is False


@given(reasons=st.lists(_SIGNAL_NAMES, min_size=1, max_size=8))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_signals_before_startup_never_raise(reasons: list[str]) -> None:
    """A signal racing bring-up must be inert, not an exception in a handler.

    An exception raised inside an asyncio signal callback goes to the loop
    exception handler and the shutdown is simply lost.
    """
    flag = _ShutdownFlag(running=False)

    for reason in reasons:
        flag.request_shutdown(reason)

    assert flag._running is False


@given(
    grace=st.floats(
        min_value=1e-6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    )
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_any_positive_grace_is_accepted_and_round_trips(grace: float) -> None:
    """The ``gt=0`` bound admits the whole positive range, not just the default."""
    assert LoopConfig(shutdown_grace_s=grace).shutdown_grace_s == grace


@given(
    grace=st.floats(
        max_value=0.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_no_non_positive_grace_is_accepted(grace: float) -> None:
    """Zero or negative would mean "cancel immediately" — always rejected.

    Swept rather than sampled because ``-0.0`` compares equal to ``0.0``
    and is exactly the kind of value a hand-written test forgets.
    """
    try:
        LoopConfig(shutdown_grace_s=grace)
    except ValidationError:
        return
    raise AssertionError(f"non-positive shutdown_grace_s {grace!r} was accepted")
