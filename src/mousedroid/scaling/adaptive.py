"""Adaptive compute — inference-time compute scaling."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class AdaptiveCompute(nn.Module):
    """Adaptive computation module that halts early when confident.

    Uses a learned halting probability at each step; once cumulative
    halt probability exceeds a threshold the computation stops.

    Args:
        input_dim: Feature dimension.
        max_steps: Maximum pondering steps.
        halt_threshold: Cumulative probability threshold to stop.
    """

    def __init__(
        self,
        input_dim: int,
        max_steps: int,
        halt_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self._max_steps = max_steps
        self._halt_threshold = halt_threshold

        self.transform = nn.Linear(input_dim, input_dim)
        self.halt_logit = nn.Linear(input_dim, 1)
        self.act = nn.ReLU()

        _log.info(
            "adaptive_compute_init",
            input_dim=input_dim,
            max_steps=max_steps,
            halt_threshold=halt_threshold,
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def forward(self, x: Tensor) -> tuple[Tensor, int]:
        """Run adaptive computation.

        Args:
            x: Input tensor, shape ``(batch, input_dim)``.

        Returns:
            Tuple of ``(output, n_steps)`` where ``n_steps`` is the actual
            number of pondering steps taken.
        """
        batch_size = x.shape[0]
        cum_halt = torch.zeros(batch_size, 1, device=x.device)
        output = torch.zeros_like(x)
        remainder = torch.ones(batch_size, 1, device=x.device)
        n_steps = 0

        for step in range(self._max_steps):
            x = self.act(self.transform(x))
            halt_prob = torch.sigmoid(self.halt_logit(x))

            still_running = (cum_halt < self._halt_threshold).float()
            # On the last step, use the remainder weight.
            if step == self._max_steps - 1:
                weight = remainder * still_running
            else:
                weight = halt_prob * still_running

            output = output + weight * x
            cum_halt = cum_halt + halt_prob * still_running
            remainder = remainder - halt_prob * still_running
            n_steps = step + 1

            if (cum_halt >= self._halt_threshold).all():
                break

        return output, n_steps
