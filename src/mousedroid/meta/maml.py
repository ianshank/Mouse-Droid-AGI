"""Model-Agnostic Meta-Learning (MAML) adapter."""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class MAMLAdapter:
    """MAML inner-outer loop meta-learner.

    Performs fast adaptation on support data (inner loop) and updates
    the meta-parameters on query data (outer loop).

    Args:
        model: Base model to meta-learn.
        inner_lr: Learning rate for the inner (adaptation) loop.
        outer_lr: Learning rate for the outer (meta) loop.
        inner_steps: Number of gradient steps in the inner loop.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float,
        outer_lr: float,
        inner_steps: int,
    ) -> None:
        self._model = model
        self._inner_lr = inner_lr
        self._inner_steps = inner_steps
        self._meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)

        _log.info(
            "maml_init",
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_steps=inner_steps,
        )

    def adapt(
        self,
        support_data: list[tuple[Tensor, Tensor]],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> nn.Module:
        """Inner-loop adaptation on support data.

        Args:
            support_data: List of ``(input, target)`` tensor pairs.
            loss_fn: Loss function ``(prediction, target) -> scalar``.

        Returns:
            Adapted model copy (original is unchanged).
        """
        adapted = copy.deepcopy(self._model)
        opt = torch.optim.SGD(adapted.parameters(), lr=self._inner_lr)

        for _ in range(self._inner_steps):
            total_loss = torch.tensor(0.0)
            for x, y in support_data:
                pred: Tensor = adapted(x)
                total_loss = total_loss + loss_fn(pred, y)
            opt.zero_grad()
            total_loss.backward()  # type: ignore[no-untyped-call]
            opt.step()

        return adapted

    def meta_step(
        self,
        tasks: list[tuple[list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]]]],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> float:
        """Outer-loop meta-update across a batch of tasks.

        Args:
            tasks: List of ``(support_data, query_data)`` per task.
            loss_fn: Loss function ``(prediction, target) -> scalar``.

        Returns:
            Mean meta-loss across tasks.
        """
        self._meta_optimizer.zero_grad()
        meta_loss = torch.tensor(0.0)

        for support, query in tasks:
            adapted = self.adapt(support, loss_fn)
            task_loss = torch.tensor(0.0)
            for x, y in query:
                pred: Tensor = adapted(x)
                task_loss = task_loss + loss_fn(pred, y)
            meta_loss = meta_loss + task_loss

        n_tasks = len(tasks)
        if n_tasks > 0:
            meta_loss = meta_loss / float(n_tasks)

        meta_loss.backward()  # type: ignore[no-untyped-call]
        self._meta_optimizer.step()

        return float(meta_loss.item())
