"""World model protocol — interface for RSSM variants."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from mousedroid.sensing.protocol import ObservationProtocol


@runtime_checkable
class WorldModelProtocol(Protocol):
    """Interface for world models (RSSM variants)."""

    def observe_step(
        self,
        observation: ObservationProtocol,
        prev_action: Tensor,
        h: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, float]:
        """Process one observation step.

        Returns:
            ``(new_h, new_z, reconstructed_obs, surprise)``
        """
        ...

    def imagine_step(
        self,
        action: Tensor,
        h: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Imagine one step forward in latent space.

        Returns:
            ``(new_h, new_z, predicted_reward)``
        """
        ...


@runtime_checkable
class LatentContextProtocol(Protocol):
    """Bounded-context latent memory operating on the carried ``(h, z)``.

    Implemented by :class:`~mousedroid.world_model.bounded_context.BoundedContextMemory`
    (F-023, ADR-015). The orchestrator calls ``observe`` with the RAW validated
    post-``observe_step`` state, then ``contextualize`` to blend the retrieved
    context into the state it carries forward. Both are pure deterministic
    ``no_grad`` tensor operations (hot-loop invariant #10: no training, no
    sampling). ``reset`` clears every store and re-arms sink capture (OTA
    weight-swap seam, ADR-010); ``rearm_sink`` re-arms only the sink (mission
    boundary seam).
    """

    def observe(self, h: Tensor, z: Tensor) -> None:
        """Store a validated latent state (detached). Drops non-finite inputs."""
        ...

    def contextualize(self, h: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """Blend retrieved context into ``(h, z)``; identity when empty/λ=0."""
        ...

    def rearm_sink(self) -> None:
        """Clear the sink anchor and restart warmup so a fresh sink is captured."""
        ...

    def reset(self) -> None:
        """Clear sink + ring + EMA summary and re-arm sink warmup."""
        ...

    def __len__(self) -> int:
        """Number of stored context vectors (ring + sink + EMA summary)."""
        ...


@runtime_checkable
class SafetyTraceProtocol(Protocol):
    """Optional interface for world models exposing CfC safety traces.

    Implemented by ``DualStreamRSSM`` to allow the safety monitor to
    independently inspect the CfC hidden state without coupling to
    the full RSSM internals.
    """

    def get_safety_trace(self, h: Tensor) -> Tensor:
        """Extract safety-relevant CfC hidden state from combined state.

        Args:
            h: Combined hidden state, shape ``(batch, combined_dim)``.

        Returns:
            CfC portion of hidden state, shape ``(batch, cfc_hidden_dim)``.
        """
        ...


@runtime_checkable
class WarmableProtocol(Protocol):
    """Optional capability: a synchronous, idempotent, one-shot warmup.

    ``WorldModelProtocol`` deliberately does not declare this. Only engines with
    a runtime session to build implement it —
    :class:`~mousedroid.world_model.dual_stream_rssm_onnx.DualStreamRSSMOnnx`
    creates its ONNX Runtime session and runs dummy inferences in
    ``warmup()``; the PyTorch engines need nothing and are simply not
    ``Warmable``, so the orchestrator's capability check skips them.

    Why this exists as a separate Protocol: ``observe_step`` warms *lazily* on
    first call (``dual_stream_rssm_onnx.py:223-224``), and the orchestrator's
    ``_update_world_model`` is synchronous
    (``orchestrator/_world_model_state_mixin.py:28``) inside a tick wrapped in
    ``asyncio.wait_for(..., tick_timeout_s)`` whose timeout path calls
    ``emergency_stop()``. ``wait_for`` cannot preempt a synchronous call, so a
    cold TensorRT engine build on the first tick would blow the tick budget and
    e-stop the rover. The orchestrator therefore warms at ``start()`` through
    :func:`asyncio.to_thread`, off the 30 Hz loop.

    Implementations must be idempotent — the lazy path may still call
    ``warmup()`` afterwards and must find the work already done.
    """

    def warmup(self) -> None:
        """Build any runtime session and run dummy inferences. Idempotent."""
        ...
