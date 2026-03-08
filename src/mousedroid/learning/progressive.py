"""Progressive neural network — column growth architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class _Column(nn.Module):
    """A single progressive network column.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.
        n_lateral: Number of lateral connections from previous columns.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_lateral: int,
    ) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

        # Lateral adapters from each previous column's hidden activations.
        self.laterals: nn.ModuleList = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_lateral)]
        )

    def forward(
        self,
        x: Tensor,
        lateral_inputs: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Forward pass with lateral connections.

        Args:
            x: Input tensor, shape ``(batch, input_dim)``.
            lateral_inputs: Hidden activations from previous columns.

        Returns:
            Tuple of ``(output, hidden)`` for downstream lateral use.
        """
        h = self.act(self.layer1(x))
        for i, lat in enumerate(lateral_inputs):
            h = h + self.laterals[i](lat)
        output: Tensor = self.layer2(h)
        return output, h


class ProgressiveNetwork(nn.Module):
    """Progressive network that grows new columns for each task.

    Existing columns are frozen; only the newest column trains.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self.columns: nn.ModuleList = nn.ModuleList()

        # Start with an initial column.
        self._add_column()
        _log.info(
            "progressive_init",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    def _add_column(self) -> None:
        """Append a new column with lateral connections to all prior columns."""
        n_lateral = len(self.columns)
        col = _Column(self._input_dim, self._hidden_dim, self._output_dim, n_lateral)
        self.columns.append(col)

    def grow(self) -> None:
        """Freeze existing columns and add a new trainable one."""
        # Freeze all existing columns.
        for col in self.columns:
            for param in col.parameters():
                param.requires_grad = False
        self._add_column()
        _log.info("progressive_grow", n_columns=len(self.columns))

    def forward(self, x: Tensor) -> Tensor:
        """Forward through the active (latest) column with lateral inputs.

        Args:
            x: Input tensor, shape ``(batch, input_dim)``.

        Returns:
            Output from the active column, shape ``(batch, output_dim)``.
        """
        hiddens: list[Tensor] = []
        output = torch.zeros(x.shape[0], self._output_dim, device=x.device)

        for i, col in enumerate(self.columns):
            lateral_inputs = hiddens[:i]
            output, h = col(x, lateral_inputs)
            hiddens.append(h)

        return output
