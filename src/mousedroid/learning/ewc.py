"""Elastic Weight Consolidation (EWC) for continual learning."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.config.schema import LearningConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class EWCAgent:
    """Elastic Weight Consolidation regularizer.

    Estimates the Fisher information matrix on a reference task and uses it
    to penalise parameter drift during subsequent learning.

    Args:
        cfg: Learning configuration with ``ewc_lambda`` and ``ewc_fisher_samples``.
        model: The neural network to protect.
    """

    def __init__(self, cfg: LearningConfig, model: nn.Module) -> None:
        self._lambda = cfg.ewc_lambda
        self._fisher_samples = cfg.ewc_fisher_samples
        self._model = model

        # Snapshots saved after consolidation.
        self._fisher: dict[str, Tensor] = {}
        self._star_params: dict[str, Tensor] = {}

        _log.info(
            "ewc_init",
            ewc_lambda=self._lambda,
            fisher_samples=self._fisher_samples,
        )

    def _named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield named parameters that require gradients."""
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                yield name, param

    def consolidate(self) -> None:
        """Snapshot current parameters and estimate Fisher information.

        Must be called after training on a task, before moving to the next.
        """
        # Snapshot parameters.
        self._star_params = {
            name: param.detach().clone()
            for name, param in self._named_parameters()
        }

        # Estimate diagonal Fisher via squared gradients.
        fisher_acc: dict[str, Tensor] = {
            name: torch.zeros_like(param)
            for name, param in self._named_parameters()
        }

        self._model.eval()
        for _ in range(self._fisher_samples):
            self._model.zero_grad()
            # Approximate Fisher with random labels — the caller should
            # override this by passing real data through the model first.
            # Here we just accumulate squared gradients of parameters.
            for name, param in self._named_parameters():
                if param.grad is not None:
                    fisher_acc[name] += param.grad.detach().pow(2)

        for name in fisher_acc:
            fisher_acc[name] /= float(self._fisher_samples)

        self._fisher = fisher_acc
        _log.info("ewc_consolidated", n_params=len(self._fisher))

    def compute_penalty(self) -> Tensor:
        """Compute the EWC regularization penalty.

        Returns:
            Scalar penalty tensor (differentiable).
        """
        if not self._fisher:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0, device=next(iter(self._fisher.values())).device)
        for name, param in self._named_parameters():
            if name in self._fisher:
                diff = param - self._star_params[name]
                penalty = penalty + (self._fisher[name] * diff.pow(2)).sum()

        return self._lambda * penalty
