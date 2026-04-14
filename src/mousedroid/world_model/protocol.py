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
