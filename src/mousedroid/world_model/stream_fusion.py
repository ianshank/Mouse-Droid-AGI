"""Stream fusion for dual-stream RSSM.

Concatenates GRU (slow dynamics) and CfC (fast dynamics) hidden states
into a combined representation.  The layout ``[h_gru | h_cfc]`` allows
independent extraction of either stream via simple tensor slicing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class StreamFusion(nn.Module):
    """Fuse GRU (slow) and CfC (fast) hidden states via concatenation.

    The combined hidden state has layout ``[h_gru | h_cfc]``.  Each
    stream's state is independently extractable by slicing, enabling
    the safety monitor to inspect the CfC state without accessing the
    full RSSM internals.

    Args:
        gru_dim: GRU hidden state dimension.
        cfc_dim: CfC hidden state dimension.
    """

    def __init__(self, gru_dim: int, cfc_dim: int) -> None:
        super().__init__()
        self._gru_dim = gru_dim
        self._cfc_dim = cfc_dim
        _log.info(
            "stream_fusion_init",
            gru_dim=gru_dim,
            cfc_dim=cfc_dim,
            combined_dim=gru_dim + cfc_dim,
        )

    @property
    def combined_dim(self) -> int:
        """Total dimension of the fused hidden state."""
        return self._gru_dim + self._cfc_dim

    @property
    def gru_dim(self) -> int:
        """GRU stream hidden dimension."""
        return self._gru_dim

    @property
    def cfc_dim(self) -> int:
        """CfC stream hidden dimension."""
        return self._cfc_dim

    def fuse(self, h_slow: Tensor, h_fast: Tensor) -> Tensor:
        """Concatenate slow (GRU) and fast (CfC) hidden states.

        Args:
            h_slow: GRU hidden state, shape ``(batch, gru_dim)``.
            h_fast: CfC hidden state, shape ``(batch, cfc_dim)``.

        Returns:
            Combined hidden state, shape ``(batch, gru_dim + cfc_dim)``.
        """
        return torch.cat([h_slow, h_fast], dim=-1)

    def extract_gru_state(self, h_combined: Tensor) -> Tensor:
        """Extract GRU stream hidden state from combined state.

        Args:
            h_combined: Combined hidden state, shape ``(batch, combined_dim)``.

        Returns:
            GRU portion, shape ``(batch, gru_dim)``.
        """
        return h_combined[..., : self._gru_dim]

    def extract_cfc_state(self, h_combined: Tensor) -> Tensor:
        """Extract CfC stream hidden state from combined state.

        Args:
            h_combined: Combined hidden state, shape ``(batch, combined_dim)``.

        Returns:
            CfC portion, shape ``(batch, cfc_dim)``.
        """
        return h_combined[..., self._gru_dim :]

    def forward(self, h_slow: Tensor, h_fast: Tensor) -> Tensor:
        """Forward pass — delegates to :meth:`fuse`."""
        return self.fuse(h_slow, h_fast)
