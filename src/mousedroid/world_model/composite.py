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
safety monitor + orchestrator can keep their existing typed seams.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor

from mousedroid.logging.setup import get_logger
from mousedroid.sensing.protocol import ObservationProtocol
from mousedroid.world_model.protocol import SafetyTraceProtocol, WorldModelProtocol

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
        the safety monitor working with ``engine="onnx_trt"`` without
        special-casing the composite type.
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
        Delegating to the PyTorch engine keeps the safety monitor wired
        whichever engine combination the operator picks.

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
