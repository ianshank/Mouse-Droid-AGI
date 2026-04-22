"""CfC (Closed-form Continuous-depth) cell wrapper.

Isolates all ``ncps`` library calls behind a single module so the
rest of the world-model code has no direct dependency on ``ncps``.
The wrapper provides a simple ``(x, h_prev) -> h_new`` interface
matching ``nn.GRUCell`` ergonomics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ModelConfig

_log = get_logger(__name__)


class CfCWrapper(nn.Module):
    """Wrapped CfC liquid neural network cell.

    Wraps ``ncps.torch.CfC`` with config-driven construction and
    structured logging.  Accepts the same ``(x, h_prev) -> h_new``
    interface as ``nn.GRUCell`` for composability.

    Args:
        input_dim: Size of the input feature vector
            (typically ``latent_dim + action_dim``).
        cfg: Model configuration containing CfC parameters.
    """

    def __init__(self, input_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.cfc_hidden_dim <= 0:
            raise ValueError(f"CfCWrapper requires cfc_hidden_dim > 0, got {cfg.cfc_hidden_dim}")
        self._hidden_dim = cfg.cfc_hidden_dim

        try:
            from ncps.torch import CfC
        except ImportError as exc:
            raise ImportError(
                "The 'ncps' package is required to use CfCWrapper. "
                "Install it with: pip install 'mousedroid[cfc]' or pip install ncps>=0.0.7"
            ) from exc

        self._cell = CfC(
            input_size=input_dim,
            units=cfg.cfc_hidden_dim,
            return_sequences=False,
            batch_first=True,
            mode=cfg.cfc_mode,
            backbone_units=cfg.cfc_backbone_units,
            backbone_layers=cfg.cfc_backbone_layers,
        )

        _log.info(
            "cfc_cell_init",
            input_dim=input_dim,
            hidden_dim=cfg.cfc_hidden_dim,
            backbone_units=cfg.cfc_backbone_units,
            backbone_layers=cfg.cfc_backbone_layers,
            mode=cfg.cfc_mode,
        )

    @property
    def hidden_size(self) -> int:
        """CfC hidden state dimension."""
        return self._hidden_dim

    def forward(
        self,
        x: Tensor,
        h: Tensor,
        *,
        dt: Tensor | None = None,
    ) -> Tensor:
        """Run one step of CfC dynamics.

        Args:
            x: Input features, shape ``(batch, input_dim)``.
            h: Previous hidden state, shape ``(batch, cfc_hidden_dim)``.
            dt: Optional time delta for ODE integration, shape
                ``(batch, 1)``.  When ``None``, uses default fixed
                step (equivalent to discrete GRU-style stepping).

        Returns:
            New hidden state, shape ``(batch, cfc_hidden_dim)``.
        """
        # ncps.CfC expects (batch, seq_len, features) with batch_first=True
        x_seq = x.unsqueeze(1)

        timespans: Tensor | None = None
        if dt is not None:
            # ncps expects timespans shape (batch, seq_len, units)
            timespans = dt.unsqueeze(1).expand(-1, 1, self._hidden_dim)

        # CfC returns (output, h_new) with return_sequences=False
        _output, h_new = self._cell(x_seq, h, timespans=timespans)
        return cast(Tensor, h_new)

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        """Create zero-initialized hidden state.

        Args:
            batch_size: Batch dimension.
            device: Target device for the tensor.

        Returns:
            Zero tensor of shape ``(batch_size, cfc_hidden_dim)``.
        """
        return torch.zeros(batch_size, self._hidden_dim, device=device)
