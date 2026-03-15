"""Working memory — tensor attention buffer with configurable context."""

from __future__ import annotations

import torch
from torch import Tensor

from mousedroid.config.schema import MemoryConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class WorkingMemory:
    """Fixed-size FIFO tensor buffer for recent observations.

    Stores the last ``context_size`` embeddings and supports a simple
    dot-product attention retrieval over the buffer contents.

    Args:
        cfg: Memory configuration with ``working_context_size``.
        embed_dim: Dimension of stored embeddings.
    """

    def __init__(self, cfg: MemoryConfig, embed_dim: int) -> None:
        self._context_size = cfg.working_context_size
        self._embed_dim = embed_dim
        self._buffer: list[Tensor] = []

        _log.info(
            "working_memory_init",
            context_size=self._context_size,
            embed_dim=embed_dim,
        )

    def push(self, embedding: Tensor) -> None:
        """Append an embedding, evicting the oldest if at capacity.

        Args:
            embedding: Tensor of shape ``(embed_dim,)``.
        """
        self._buffer.append(embedding.detach())
        if len(self._buffer) > self._context_size:
            self._buffer.pop(0)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def attend(self, query: Tensor) -> Tensor:
        """Retrieve a weighted sum of buffer entries via dot-product attention.

        Args:
            query: Query tensor, shape ``(embed_dim,)``.

        Returns:
            Attended context tensor, shape ``(embed_dim,)``.
            Returns a zero tensor if the buffer is empty.
        """
        if not self._buffer:
            return torch.zeros(self._embed_dim, device=query.device)

        keys = torch.stack(self._buffer)  # (n, embed_dim)
        scores = keys @ query  # (n,)
        scale = float(self._embed_dim) ** 0.5
        weights = torch.softmax(scores / scale, dim=0)  # (n,)
        result: Tensor = weights @ keys  # (embed_dim,)
        return result

    def clear(self) -> None:
        """Reset the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Current number of stored embeddings."""
        return len(self._buffer)
