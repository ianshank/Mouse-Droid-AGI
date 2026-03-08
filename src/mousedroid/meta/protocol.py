"""Meta-learning protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MetaLearnerProtocol(Protocol):
    """Interface for meta-learning methods."""

    def adapt(self, support_data: list[object], n_steps: int) -> float:
        """Adapt to new task from support data. Returns loss."""
        ...
