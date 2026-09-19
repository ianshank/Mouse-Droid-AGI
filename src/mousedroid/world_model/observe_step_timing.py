"""Engine-agnostic ``observe_step`` latency reporting.

Every world-model engine —
:class:`~mousedroid.world_model.rssm.RSSM`,
:class:`~mousedroid.world_model.dual_stream_rssm.DualStreamRSSM` and
:class:`~mousedroid.world_model.dual_stream_rssm_onnx.DualStreamRSSMOnnx` —
needs the same guarded "time the call, report it if a sink is wired" behaviour.
Keeping it here means the pattern exists once instead of three times, and the
engines stay typed against a :class:`~typing.Protocol` rather than importing
:class:`~mousedroid.telemetry.metrics.registry.MetricsRegistry` (CLAUDE.md
invariant 1: concrete types are imported inside ``factory/`` only).

Two deliberate choices, both load-bearing:

**Success-path recording only.** The context manager reports nothing when the
wrapped body raises, per ``telemetry/CLAUDE.md`` invariant 5 ("Record metrics on
success; never on cancellation or error paths"). A failed ``observe_step`` has no
meaningful latency to contribute, and recording one would corrupt the percentile
the deadline alert reads. The orchestrator's own ``_finish_tick_timing`` makes the
same choice for the tick histogram, so the two families stay consistent.

**Sink-optional, zero-overhead when absent.** ``sink=None`` short-circuits before
the clock is read, so an engine built without telemetry pays nothing on the 30 Hz
path. That matters because the engines are constructed with no registry in every
non-orchestrator caller (validation pillars, on-device learning, growth).

Bucket boundaries are *not* defined here — they come from
``MetricsConfig.world_model_observe_step_seconds_buckets`` so operators can retune
the histogram without a code change (CLAUDE.md invariant 2).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing-only base

    class _StateHookBase:
        """Typed stand-in for the copy/pickle hooks the mixin delegates through.

        Mixin-base aliasing, the standard way to type a mixin against the classes
        it is mixed INTO. At runtime :class:`ObserveStepTimingMixin` adds no base
        of its own — it is composed as
        ``class RSSM(ObserveStepTimingMixin, nn.Module)``, so ``super()`` reaches
        ``torch.nn.Module`` through the MRO and these bodies never execute.

        Declaring the base here rather than importing ``nn.Module`` does two
        things: it keeps this module free of a runtime torch import, and it gives
        the two hooks annotations that ``mypy --strict`` can check. Torch ships
        them unannotated, so delegating to ``nn.Module`` directly would demand a
        suppression — and the repository's suppression budgets are at their
        ceiling (``tests/regression/test_suppression_budget.py``), so this module
        must not spend from them.
        """

        def __getstate__(self) -> dict[str, Any]:
            """Return the state a copy or pickle should carry."""
            raise NotImplementedError

        def __setstate__(self, state: dict[str, Any]) -> None:
            """Restore previously returned state onto a fresh instance."""
            raise NotImplementedError

else:
    _StateHookBase = object

_log = get_logger(__name__)


@runtime_checkable
class ObserveStepLatencySink(Protocol):
    """Minimal surface an engine needs to report observe-step latency.

    :class:`~mousedroid.telemetry.metrics.registry.MetricsRegistry` satisfies
    this structurally. Declaring the narrow Protocol instead of the concrete
    registry keeps the world-model package free of a telemetry import and lets
    tests pass a three-line fake.
    """

    def observe_world_model_observe_step_seconds(self, value: float) -> None:
        """Record one observe-step latency sample, in seconds."""
        ...  # pragma: no cover - structural protocol declaration


class ObserveStepTimingMixin(_StateHookBase):
    """Gives an engine an optional, copy-safe observe-step latency sink.

    Subclasses assign ``self._metrics`` in ``__init__``. The mixin exists for
    one reason: to drop the sink when the engine is copied.

    ``src/mousedroid/learning/on_device/rssm_refiner.py`` does
    ``copy.deepcopy(self._base_rssm)`` to build a refinement *candidate*.
    Without this override that deepcopy raises
    ``TypeError: cannot pickle '_thread.lock' object``, because
    :class:`~mousedroid.telemetry.metrics.registry.MetricsRegistry` is live
    process state holding a lock, not model state.

    Dropping the sink is also the semantically correct answer rather than a
    workaround: a candidate model being trained off-loop must not emit into the
    production ``mousedroid_world_model_observe_step_seconds`` histogram, or the
    percentile the deadline alert reads would mix real ticks with refinement
    passes. A copied engine is therefore untimed until a caller wires a sink to
    it explicitly.

    Place the mixin before ``nn.Module`` in the bases so ``super()`` reaches
    ``nn.Module.__getstate__`` — the state dict is copied, then the sink nulled.
    """

    _metrics: ObserveStepLatencySink | None

    def __getstate__(self) -> dict[str, Any]:
        """Return copy/pickle state with the latency sink removed."""
        state = dict(super().__getstate__())
        if "_metrics" in state:
            state["_metrics"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state, defaulting the sink to ``None`` when absent."""
        state.setdefault("_metrics", None)
        super().__setstate__(state)


@contextmanager
def observe_step_latency(
    sink: ObserveStepLatencySink | None,
    *,
    engine: str,
) -> Iterator[None]:
    """Time the wrapped region and report it when ``sink`` is wired.

    Reports only when the body completes without raising, per
    ``telemetry/CLAUDE.md`` invariant 5. The sink itself drops NaN, ``+Inf`` and
    negative samples (see
    :meth:`MetricsRegistry.observe_world_model_observe_step_seconds`), so this
    helper deliberately does not re-validate the value — one guard, one place.

    Args:
        sink: Latency sink, or ``None`` to disable timing entirely. ``None``
            short-circuits before the clock is read.
        engine: Engine identifier for the debug event, so a operator reading
            logs can tell which engine produced a sample. Callers pass their own
            ``name``/engine literal; it is never used as a metric label, because
            that would add unbounded cardinality to the histogram.

    Yields:
        ``None`` — the caller runs its inference inside the ``with`` block.
    """
    if sink is None:
        yield
        return

    start = time.perf_counter()
    yield
    # Reached only when the body did not raise — invariant 5.
    elapsed = time.perf_counter() - start
    sink.observe_world_model_observe_step_seconds(elapsed)
    _log.debug(
        "world_model_observe_step_timed",
        engine=engine,
        seconds=elapsed,
    )
