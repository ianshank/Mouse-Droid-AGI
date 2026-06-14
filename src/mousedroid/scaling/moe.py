"""Sparse Mixture-of-Experts layer with vectorized top-k routing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — canonical PyTorch alias for functional
from torch import Tensor

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class SparseMoELayer(nn.Module):
    """Sparse Mixture-of-Experts with vectorized dispatch (no Python loops).

    Each token is routed to the top-k experts via a learned gating network.
    Expert outputs are combined using the softmax-normalized gate scores.

    Args:
        input_dim: Input and output feature dimension.
        n_experts: Total number of expert sub-networks.
        expert_dim: Hidden dimension within each expert.
        top_k: Number of experts activated per token.

    Raises:
        ValueError: If ``top_k`` exceeds ``n_experts``.
    """

    def __init__(
        self,
        input_dim: int,
        n_experts: int,
        expert_dim: int,
        top_k: int,
    ) -> None:
        super().__init__()
        if top_k > n_experts:
            msg = f"top_k ({top_k}) must not exceed n_experts ({n_experts})"
            raise ValueError(msg)

        self._n_experts = n_experts
        self._top_k = top_k
        self._input_dim = input_dim

        # Gate: project input to per-expert logits.
        self.gate = nn.Linear(input_dim, n_experts, bias=False)

        # Expert parameters stored as 3-D tensors for vectorized matmul.
        self.expert_w1 = nn.Parameter(torch.empty(n_experts, input_dim, expert_dim))
        self.expert_b1 = nn.Parameter(torch.zeros(n_experts, expert_dim))
        self.expert_w2 = nn.Parameter(torch.empty(n_experts, expert_dim, input_dim))
        self.expert_b2 = nn.Parameter(torch.zeros(n_experts, input_dim))

        # Kaiming init.
        nn.init.kaiming_uniform_(self.expert_w1)
        nn.init.kaiming_uniform_(self.expert_w2)

        _log.info(
            "moe_init",
            n_experts=n_experts,
            expert_dim=expert_dim,
            top_k=top_k,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with sparse top-k expert routing.

        Args:
            x: Input tensor, shape ``(batch, input_dim)``.

        Returns:
            Output tensor, shape ``(batch, input_dim)``.
        """
        batch_size = x.shape[0]

        # Gate scores: (batch, n_experts)
        logits = self.gate(x)
        top_k_vals, top_k_idx = torch.topk(logits, self._top_k, dim=-1)
        gate_weights = F.softmax(top_k_vals, dim=-1)  # (batch, top_k)

        # Compute ALL expert outputs via batched matmul (no Python loop).
        # x: (batch, 1, input_dim) @ w1: (n_experts, input_dim, expert_dim)
        # -> (batch, n_experts, expert_dim)  via broadcast
        x_expanded = x.unsqueeze(1).expand(batch_size, self._n_experts, self._input_dim)
        hidden = torch.bmm(
            x_expanded.reshape(batch_size * self._n_experts, 1, self._input_dim),
            self.expert_w1.unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .reshape(batch_size * self._n_experts, self._input_dim, -1),
        ).reshape(batch_size, self._n_experts, -1)
        hidden = hidden + self.expert_b1.unsqueeze(0)
        hidden = F.relu(hidden)

        expert_out = torch.bmm(
            hidden.reshape(batch_size * self._n_experts, 1, -1),
            self.expert_w2.unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .reshape(batch_size * self._n_experts, -1, self._input_dim),
        ).reshape(batch_size, self._n_experts, self._input_dim)
        expert_out = expert_out + self.expert_b2.unsqueeze(0)

        # Gather only the top-k expert outputs: (batch, top_k, input_dim)
        idx_expanded = top_k_idx.unsqueeze(-1).expand(-1, -1, self._input_dim)
        selected = torch.gather(expert_out, 1, idx_expanded)

        # Weighted sum: (batch, top_k, 1) * (batch, top_k, input_dim) -> sum
        output: Tensor = (gate_weights.unsqueeze(-1) * selected).sum(dim=1)
        return output
