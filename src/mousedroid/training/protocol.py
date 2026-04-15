"""Training phase protocol — defines the contract for pluggable training phases.

Each training phase (RSSM, warm-start, BDI, constitutional RL) implements
this protocol so the ``PipelineOrchestrator`` can dispatch to them uniformly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from mousedroid.config.schema import Settings


@runtime_checkable
class TrainingPhaseProtocol(Protocol):
    """Contract for a single training phase.

    Implementations wrap the actual training functions from
    ``training.run_pipeline`` and adapt them to a uniform async interface
    usable by ``PipelineOrchestrator``.
    """

    @property
    def name(self) -> str:
        """Human-readable phase name (e.g. ``"rssm"``)."""
        ...

    async def run(
        self,
        cfg: Settings,
        batch_size: int,
        checkpoint_dir: Path,
    ) -> Path:
        """Execute this training phase.

        Args:
            cfg: Root application settings.
            batch_size: Tuned batch size for this phase.
            checkpoint_dir: Directory for writing checkpoints / artifacts.

        Returns:
            Path to the primary artifact produced (checkpoint or weights dir).
        """
        ...
