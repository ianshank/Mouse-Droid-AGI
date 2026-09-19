"""Unit tests for the shared observe-step latency helper.

Deliberately torch-free: the helper is pure stdlib so these run in the blocking
``test`` job (which installs ``.[dev,telemetry,mcp]`` — no onnxruntime) and give
the timing contract always-on coverage. Engine wiring is covered separately in
``tests/integration/test_world_model_metrics_seam.py``.
"""

from __future__ import annotations

import copy
import math
import threading
from typing import Any

import pytest

from mousedroid.world_model.observe_step_timing import (
    ObserveStepLatencySink,
    ObserveStepTimingMixin,
    observe_step_latency,
)


def _raise(exc: BaseException) -> None:
    """Raise ``exc``.

    A one-statement raise keeps ``pytest.raises`` blocks PT012-clean while the
    exception instance is still built outside the block, so EM101 stays happy
    too.
    """
    raise exc


class _RecordingSink:
    """Minimal :class:`ObserveStepLatencySink` implementation for tests."""

    def __init__(self) -> None:
        self.samples: list[float] = []

    def observe_world_model_observe_step_seconds(self, value: float) -> None:
        self.samples.append(value)


class _LockingSink:
    """Sink holding a lock, i.e. unpicklable exactly like the real registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def observe_world_model_observe_step_seconds(self, value: float) -> None:
        with self._lock:
            pass


class _ExplodingSink:
    """Sink that raises, to prove the helper does not swallow sink failures."""

    def observe_world_model_observe_step_seconds(self, value: float) -> None:
        msg = f"sink rejected {value}"
        raise RuntimeError(msg)


class TestProtocolConformance:
    """The Protocol is structural, so a plain object satisfies it."""

    def test_recording_sink_satisfies_protocol(self) -> None:
        assert isinstance(_RecordingSink(), ObserveStepLatencySink)

    def test_object_without_the_method_does_not_satisfy_protocol(self) -> None:
        assert not isinstance(object(), ObserveStepLatencySink)

    def test_metrics_registry_satisfies_protocol(self) -> None:
        """The real registry must stay structurally compatible.

        This is the pin that catches a rename of
        ``observe_world_model_observe_step_seconds`` on the registry, which would
        otherwise silently stop every engine from reporting.
        """
        from mousedroid.config.schema import Settings
        from mousedroid.telemetry.metrics.registry import MetricsRegistry

        registry = MetricsRegistry(Settings(mock_hardware=True).metrics)
        assert isinstance(registry, ObserveStepLatencySink)


class TestReporting:
    """A completed body reports exactly one non-negative sample."""

    def test_records_one_sample_per_completed_block(self) -> None:
        sink = _RecordingSink()
        for _ in range(3):
            with observe_step_latency(sink, engine="unit"):
                pass
        assert len(sink.samples) == 3

    def test_sample_is_finite_and_non_negative(self) -> None:
        sink = _RecordingSink()
        with observe_step_latency(sink, engine="unit"):
            pass
        (sample,) = sink.samples
        assert sample >= 0.0
        assert math.isfinite(sample)

    def test_return_value_passes_through(self) -> None:
        """A ``return`` inside the block still triggers the report.

        ``contextlib.contextmanager`` resumes the generator on normal exit, so
        the engines' ``with ...: return self._observe_step_impl(...)`` shape
        reports correctly. Without this the seam would look wired and emit
        nothing.
        """
        sink = _RecordingSink()

        def _run() -> str:
            with observe_step_latency(sink, engine="unit"):
                return "payload"

        assert _run() == "payload"
        assert len(sink.samples) == 1


class TestSuccessPathOnly:
    """telemetry/CLAUDE.md invariant 5 — never record on an error path."""

    def test_raising_body_records_nothing(self) -> None:
        sink = _RecordingSink()
        error = ValueError("inference failed")
        with (
            pytest.raises(ValueError, match="inference failed"),
            observe_step_latency(sink, engine="unit"),
        ):
            _raise(error)
        assert sink.samples == []

    def test_raising_body_propagates_the_original_exception(self) -> None:
        sink = _RecordingSink()
        with (
            pytest.raises(KeyError),
            observe_step_latency(sink, engine="unit"),
        ):
            raise KeyError("missing modality")
        assert sink.samples == []

    def test_cancellation_records_nothing(self) -> None:
        """``BaseException`` (e.g. task cancellation) is also an error path."""
        sink = _RecordingSink()
        with (
            pytest.raises(KeyboardInterrupt),
            observe_step_latency(sink, engine="unit"),
        ):
            raise KeyboardInterrupt
        assert sink.samples == []


class TestSinkOptional:
    """``sink=None`` must be a no-op, not a crash."""

    def test_none_sink_runs_the_body(self) -> None:
        ran = False
        with observe_step_latency(None, engine="unit"):
            ran = True
        assert ran

    def test_none_sink_passes_return_value_through(self) -> None:
        def _run() -> int:
            with observe_step_latency(None, engine="unit"):
                return 42

        assert _run() == 42

    def test_none_sink_propagates_exceptions_unchanged(self) -> None:
        error = ValueError("boom")
        with (
            pytest.raises(ValueError, match="boom"),
            observe_step_latency(None, engine="unit"),
        ):
            _raise(error)


class TestSinkFailures:
    """A broken sink must surface, not be silently swallowed."""

    def test_sink_exception_propagates(self) -> None:
        with (
            pytest.raises(RuntimeError, match="sink rejected"),
            observe_step_latency(_ExplodingSink(), engine="unit"),
        ):
            pass


class _StatefulBase:
    """Stands in for ``nn.Module``: a base that defines the state hooks.

    The mixin delegates through ``super()``, so it must sit *before* such a base
    in the MRO. Reproducing that shape here (rather than subclassing
    ``nn.Module``) keeps this module torch-free.
    """

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class _Host(ObserveStepTimingMixin, _StatefulBase):
    """Mixin host carrying a genuinely unpicklable sink."""

    def __init__(self) -> None:
        self._metrics: ObserveStepLatencySink | None = _LockingSink()
        self.payload = [1, 2, 3]


class TestCopySafety:
    """``ObserveStepTimingMixin`` must drop the sink on copy.

    ``learning/on_device/rssm_refiner.py`` deepcopies the live engine to build a
    refinement candidate. A :class:`MetricsRegistry` holds a ``threading.Lock``,
    which is unpicklable, so without the mixin that copy raises
    ``TypeError: cannot pickle '_thread.lock' object``. Dropping the sink is also
    semantically right: an off-loop candidate must not emit into the production
    histogram.

    The real engines are covered in
    ``tests/regression/test_f050_backwards_compat.py``; this tier pins the
    mixin's own contract.
    """

    def test_the_mixin_precedes_the_state_holding_base(self) -> None:
        """MRO order is the whole mechanism — a shadowed mixin is a silent no-op."""
        mro = _Host.__mro__
        assert mro.index(ObserveStepTimingMixin) < mro.index(_StatefulBase)

    def test_deepcopy_of_a_wired_host_succeeds(self) -> None:
        assert copy.deepcopy(_Host()) is not None

    def test_shallow_copy_is_also_unwired(self) -> None:
        """Both copy protocols route through the state hooks, so both are safe.

        ``copy.copy`` reaches ``__getstate__`` via ``__reduce_ex__``, the same
        path ``pickle`` takes — checked here instead of round-tripping a pickle,
        which ruff blocks in the suite (``S301``).
        """
        assert copy.copy(_Host())._metrics is None

    def test_the_copy_is_unwired(self) -> None:
        assert copy.deepcopy(_Host())._metrics is None

    def test_the_original_keeps_its_sink(self) -> None:
        host = _Host()
        copy.deepcopy(host)
        assert host._metrics is not None, "copying must not disarm the live engine"

    def test_other_state_still_copies(self) -> None:
        assert copy.deepcopy(_Host()).payload == [1, 2, 3]

    def test_other_state_is_deep_copied_not_shared(self) -> None:
        host = _Host()
        copy.deepcopy(host).payload.append(4)
        assert host.payload == [1, 2, 3]

    def test_setstate_defaults_the_sink_when_absent(self) -> None:
        """An old pickle predating the field must still load."""
        host = _Host()
        host.__setstate__({"payload": []})
        assert host._metrics is None

    def test_getstate_leaves_an_unwired_host_alone(self) -> None:
        host = _Host()
        host._metrics = None
        assert host.__getstate__()["_metrics"] is None

    def test_a_copy_can_be_rewired_explicitly(self) -> None:
        """Dropping the sink is not one-way: a caller may opt the copy back in."""
        candidate = copy.deepcopy(_Host())
        sink = _RecordingSink()
        candidate._metrics = sink
        with observe_step_latency(candidate._metrics, engine="unit"):
            pass
        assert len(sink.samples) == 1
