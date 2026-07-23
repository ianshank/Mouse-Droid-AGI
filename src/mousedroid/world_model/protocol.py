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
