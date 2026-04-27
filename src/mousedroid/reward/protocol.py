"""Reward model protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class RewardModelProtocol(Protocol):
    """Interface for reward models.

    Phase 4 extension: ``compute_reward`` and ``forward`` accept optional
    keyword-only ``prev_obs`` / ``curr_obs`` / ``instruction`` arguments so a
    VLM-derived progress head (when wired) can score a transition. The
    arguments are keyword-only and default-``None`` so the legacy
    single-state call site (``model(state)``) remains byte-identical.
    """

    def compute_reward(
        self,
        state: Tensor,
        *,
        prev_obs: Tensor | None = ...,
        curr_obs: Tensor | None = ...,
        instruction: str | None = ...,
    ) -> dict[str, Tensor]:
        """Compute per-objective reward scores."""
        ...

    def aggregate(self, scores: dict[str, Tensor]) -> Tensor:
        """Aggregate per-objective scores into scalar reward."""
        ...

    def forward(
        self,
        state: Tensor,
        *,
        prev_obs: Tensor | None = ...,
        curr_obs: Tensor | None = ...,
        instruction: str | None = ...,
    ) -> Tensor:
        """Compute and aggregate reward in one call."""
        ...
