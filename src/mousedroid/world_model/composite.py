"""Composite world model — split observe / imagine across engines.

Resolves the gap between ``cfg.world_model.engine = "onnx_trt"`` and
:class:`mousedroid.world_model.mcts.MCTSPlanner`. The ONNX export
(B2 Story 1) only ships ``observe_step``; the planner calls
``imagine_step`` during MCTS rollouts. Without composition, flipping
the engine would silently crash the planner at runtime.

This module owns the composition exactly once. The factory builds:

* a fast ONNX-backed ``observe_engine`` (:class:`DualStreamRSSMOnnx`)
* a PyTorch ``imagine_engine`` (:class:`DualStreamRSSM`) that retains
  the trained weights for ``imagine_step`` / ``get_safety_trace`` /
  any future MCTS-side helpers

The composite is the value the factory hands to the orchestrator. Both
child engines are owned and lifecycle-managed by the composite — no
caller has to construct them in parallel.

The class conforms to :class:`WorldModelProtocol` and (when the
imagine engine implements it) :class:`SafetyTraceProtocol` so the
orchestrator can keep its existing typed seam. ``SafetyTraceProtocol``
conformance is structural only -- ``get_safety_trace`` has no
production caller (:mod:`mousedroid.safety.monitor` never calls it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor

from mousedroid.logging.setup import get_logger
from mousedroid.sensing.protocol import ObservationProtocol
from mousedroid.world_model.protocol import (
    SafetyTraceProtocol,
    WarmableProtocol,
    WorldModelProtocol,
)

if TYPE_CHECKING:
    pass  # No type-checking-only imports needed yet.

_log = get_logger(__name__)


class CompositeWorldModel:
    """Split ``observe_step`` and ``imagine_step`` across two engines.

    Used by the factory when ``cfg.world_model.engine == "onnx_trt"`` so
    the runtime path benefits from the ONNX-accelerated ``observe_step``
    while MCTS rollouts continue to use the PyTorch model's
    ``imagine_step`` (the ONNX graph only contains the observe pipeline
    — :func:`scripts.export_dual_stream_rssm_onnx.run_export` is
    intentionally scoped to that critical-path method).

    The class is intentionally protocol-based — it doesn't hardcode
    ``DualStreamRSSMOnnx`` or ``DualStreamRSSM`` as types. Any pair
    conforming to :class:`WorldModelProtocol` works (e.g. future
    ``DualStreamRSSMTensorRT`` for direct-TRT inference, or a mock
    engine for tests).

    Args:
        observe_engine: World model serving ``observe_step`` calls (the
            hot path on the 30Hz orchestrator tick).
        imagine_engine: World model serving ``imagine_step`` calls (MCTS
            rollouts; typically a PyTorch ``DualStreamRSSM`` that retains
            the full prior network + reward head).
        name: Telemetry label, surfaced in structured logs.

    Safety-trace delegation:
        When ``imagine_engine`` implements :class:`SafetyTraceProtocol`,
        the composite's :meth:`get_safety_trace` forwards. This keeps
        the protocol satisfied under ``engine="onnx_trt"`` without
        special-casing the composite type. It wires no live consumer:
        ``get_safety_trace`` has no production caller today (see
        :meth:`get_safety_trace`).
    """

    def __init__(
        self,
        *,
        observe_engine: WorldModelProtocol,
        imagine_engine: WorldModelProtocol,
        name: str = "composite_world_model",
    ) -> None:
        self._observe_engine = observe_engine
        self._imagine_engine = imagine_engine
        self._name = name
        _log.info(
            "composite_world_model_initialized",
            observe_engine=type(observe_engine).__name__,
            imagine_engine=type(imagine_engine).__name__,
            name=name,
        )

    @property
    def name(self) -> str:
        """Telemetry name."""
        return self._name

    @property
    def observe_engine(self) -> WorldModelProtocol:
        """The world model serving ``observe_step`` (read-only)."""
        return self._observe_engine

    @property
    def imagine_engine(self) -> WorldModelProtocol:
        """The world model serving ``imagine_step`` (read-only)."""
        return self._imagine_engine

    def warmup(self) -> None:
        """Warm whichever delegate has a runtime session to build.

        Without this the composite silently defeats the orchestrator's
        ``start()``-time warmup. ``build_world_model`` returns a
        ``CompositeWorldModel`` for ``engine: onnx_trt``, and the
        ``Warmable`` object is the *observe engine* inside it, not the
        composite — so ``isinstance(model, WarmableProtocol)`` in
        ``orchestrator/_lifecycle_mixin.py`` was ``False`` for the one
        deployment that needs warming, the lazy TensorRT build happened on
        the first tick anyway, and that blows ``tick_timeout_s`` into
        ``emergency_stop()``.

        Both delegates are offered the call because which one holds a session
        is not this class's business to assume: today the ONNX engine serves
        ``observe_step`` and PyTorch serves ``imagine_step``, but the
        composite is a general two-engine seam. Each engine's ``warmup`` is
        required to be idempotent (see :class:`WarmableProtocol`), so warming
        a shared engine twice is harmless.
        """
        for role, engine in (
            ("observe", self._observe_engine),
            ("imagine", self._imagine_engine),
        ):
            if isinstance(engine, WarmableProtocol):
                _log.info(
                    "composite_world_model_warming_delegate",
                    role=role,
                    engine=type(engine).__name__,
                )
                engine.warmup()

    def observe_step(
        self,
        observation: ObservationProtocol,
        prev_action: Tensor,
        h: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, float]:
        """Delegate to the observe engine — typically the fast ONNX path."""
        return self._observe_engine.observe_step(observation, prev_action, h, z)

    def imagine_step(
        self,
        action: Tensor,
        h: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Delegate to the imagine engine — typically the PyTorch model.

        MCTS rollouts call this many times per planning tick. The
        delegate must conform to :class:`WorldModelProtocol`; the
        composite does no shape adaptation.
        """
        return self._imagine_engine.imagine_step(action, h, z)

    def get_safety_trace(self, h: Tensor) -> Tensor:
        """Forward to the imagine engine when it supports safety traces.

        ``DualStreamRSSM`` (the typical imagine engine) implements
        :class:`SafetyTraceProtocol`; ``DualStreamRSSMOnnx`` does not
        (the ONNX export doesn't carry the get_safety_trace head).
        Delegating to the PyTorch engine keeps the protocol satisfied
        whichever engine combination the operator picks. Note this
        method currently has **no production caller**:
        :mod:`mousedroid.safety.monitor` never invokes it, and the only
        callers in the tree are tests. It is kept for
        :class:`SafetyTraceProtocol` conformance and future CfC-trace
        consumers, not because anything is wired through it today.

        Raises:
            AttributeError: If the imagine engine doesn't implement
                :class:`SafetyTraceProtocol`. Surfaces a clear failure
                mode rather than silently returning zeros.
        """
        if not isinstance(self._imagine_engine, SafetyTraceProtocol):
            msg = (
                f"imagine_engine ({type(self._imagine_engine).__name__}) does "
                "not implement SafetyTraceProtocol; cannot serve "
                "get_safety_trace. Use a PyTorch DualStreamRSSM as the "
                "imagine engine when the safety monitor is active."
            )
            raise AttributeError(msg)
        return cast(SafetyTraceProtocol, self._imagine_engine).get_safety_trace(h)
