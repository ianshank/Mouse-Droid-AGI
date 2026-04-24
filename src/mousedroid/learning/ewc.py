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
        self._fisher_batch_size = cfg.ewc_fisher_batch_size
        self._fallback_input_dim = cfg.ewc_fallback_input_dim
        self._model = model

        # Snapshots saved after consolidation.
        self._fisher: dict[str, Tensor] = {}
        self._star_params: dict[str, Tensor] = {}

        _log.info(
            "ewc_init",
            ewc_lambda=self._lambda,
            fisher_samples=self._fisher_samples,
            fisher_batch_size=self._fisher_batch_size,
        )

    def _named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield named parameters that require gradients."""
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                yield name, param

    def consolidate(self, data_loader: Iterator[Tensor] | None = None) -> None:
        """Snapshot current parameters and estimate Fisher information.

        Estimates the diagonal Fisher by computing squared gradients of
        the log-likelihood over *data_loader* samples.  When no loader is
        provided, random inputs matching the first layer's in-features are
        generated for ``ewc_fisher_samples`` iterations — this keeps the
        API backwards-compatible while producing meaningful (non-zero)
        Fisher estimates.

        Must be called after training on a task, before moving to the next.

        Args:
            data_loader: Optional iterator yielding input tensors for Fisher
                estimation.  When ``None``, random inputs are generated.
        """
        # Snapshot parameters.
        self._star_params = {
            name: param.detach().clone() for name, param in self._named_parameters()
        }

        # Estimate diagonal Fisher via squared gradients.
        fisher_acc: dict[str, Tensor] = {
            name: torch.zeros_like(param) for name, param in self._named_parameters()
        }

        self._model.eval()
        sample_iter = iter(data_loader) if data_loader is not None else None
        input_dim = self._infer_input_dim()
        _log.debug(
            "ewc_consolidation_start",
            fisher_samples=self._fisher_samples,
            input_dim=input_dim,
            using_data_loader=sample_iter is not None,
        )

        for _ in range(self._fisher_samples):
            self._model.zero_grad()

            if sample_iter is not None:
                try:
                    x = next(sample_iter)
                except StopIteration:
                    break
            else:
                x = torch.randn(self._fisher_batch_size, input_dim)

            output = self._model(x)
            log_prob = torch.log_softmax(output, dim=-1)
            target_idx = log_prob.argmax(dim=-1)
            loss = -log_prob.gather(1, target_idx.unsqueeze(-1)).mean()
            loss.backward()  # type: ignore[no-untyped-call]

            for name, param in self._named_parameters():
                if param.grad is not None:
                    fisher_acc[name] += param.grad.detach().pow(2)

        for name in fisher_acc:
            fisher_acc[name] /= float(self._fisher_samples)

        self._fisher = fisher_acc
        _log.info("ewc_consolidated", n_params=len(self._fisher))

    def _infer_input_dim(self) -> int:
        """Infer input dimension from the first Linear layer of the model.

        Returns:
            Number of input features for the first layer.
        """
        for module in self._model.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        _log.warning("ewc_input_dim_fallback", fallback=self._fallback_input_dim)
        return self._fallback_input_dim

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
