"""Bounded EWC-regularized on-device online update (Phase 6 WS2).

Refreshes a policy/world-model on fresh rover experience between cloud
retraining cycles, WITHOUT corrupting the cloud-pulled base weights:

* The base model is deep-copied into a *candidate* before any gradient flows;
  every optimizer step touches only the candidate. The base parameters are
  bitwise-unchanged on return (pinned by
  ``tests/property/test_on_device_no_inplace_corruption.py``).
* The regularization term reuses ``learning/ewc.py`` (``EWCAgent``) — the
  diagonal-Fisher penalty anchored on the consolidated base weights — weighted
  by ``OnDeviceLearningConfig.ewc_lambda``. No Fisher/penalty maths is
  re-implemented here.
* Steps / learning-rate / EWC weight are ALL config-driven
  (``OnDeviceLearningConfig``); nothing is hardcoded.
* The task criterion is injectable (``task_loss_fn``), defaulting to
  :func:`default_task_loss` (the squared-mean stand-in) so existing behaviour is
  byte-identical. WS4/WS5 can inject a real reward/supervised criterion without
  editing this class.

The update is SYNC and uses ``torch.autograd.grad`` + a manual SGD step (no
``loss.backward()``, no optimizer object) so the candidate's ``.grad`` buffers
and the base model are both left untouched.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.config.schema import LearningConfig, OnDeviceLearningConfig
from mousedroid.learning.ewc import EWCAgent
from mousedroid.learning.on_device.protocol import OnDeviceUpdateResult
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

#: Type of a task-loss criterion: maps the candidate's forward output to a
#: scalar loss tensor. Injected at construction so WS4/WS5 can swap in a real
#: reward/supervised criterion without editing this class.
TaskLossFn = Callable[[Tensor], Tensor]


def default_task_loss(output: Tensor) -> Tensor:
    """Default placeholder task loss: mean of squared activations.

    This is a stand-in that keeps the bounded update self-contained until a
    real reward/supervised criterion is injected (WS4/WS5). It is the documented
    backwards-compatible default — omitting ``task_loss_fn`` reproduces the
    legacy behaviour byte-for-byte.

    Args:
        output: The candidate model's forward output for the experience batch.

    Returns:
        A scalar loss tensor (mean of the squared outputs).
    """
    return output.pow(2).mean()


class EWCOnlineLearner:
    """Bounded EWC-regularized online learner producing candidate weights.

    Args:
        cfg: On-device learning configuration (steps, learning rate, EWC weight).
        base_model: The cloud-pulled / live model to refine. It is NEVER
            mutated — a deep copy is updated and returned as the candidate.
        fisher_loader: Optional iterator of input tensors used to estimate the
            diagonal Fisher information for the EWC anchor. When ``None``, the
            reused :class:`EWCAgent` falls back to random inputs sized to the
            model's first layer (its documented backwards-compatible path).
        task_loss_fn: Optional criterion mapping the candidate's forward output
            to a scalar task loss. Defaults to :func:`default_task_loss` (the
            squared-mean stand-in) so existing behaviour is byte-identical;
            WS4/WS5 can inject a real reward/supervised criterion here without
            editing this class.
    """

    def __init__(
        self,
        cfg: OnDeviceLearningConfig,
        base_model: nn.Module,
        fisher_loader: Iterator[Tensor] | None = None,
        task_loss_fn: TaskLossFn | None = None,
    ) -> None:
        self._cfg = cfg
        self._base_model = base_model
        self._fisher_loader = fisher_loader
        self._task_loss_fn: TaskLossFn = task_loss_fn or default_task_loss

    def update(self, batch: Tensor) -> OnDeviceUpdateResult:
        """Run ``cfg.update_steps`` bounded steps and return the candidate.

        Args:
            batch: Experience batch shaped ``(n, input_dim)`` matching the
                base model's first-layer in-features.

        Returns:
            An :class:`OnDeviceUpdateResult` with the candidate state-dict, the
            final train loss, and the number of steps executed.
        """
        steps = self._cfg.update_steps
        lr = self._cfg.learning_rate
        ewc_lambda = self._cfg.ewc_lambda

        _log.info(
            "on_device_update_start",
            steps=steps,
            learning_rate=lr,
            ewc_lambda=ewc_lambda,
        )

        # Candidate = deep copy of the base. All gradient flow is confined here;
        # the base model's parameters stay bitwise-identical.
        candidate = copy.deepcopy(self._base_model)

        # Build the EWC anchor BEFORE switching to train mode: the reused
        # ``EWCAgent.consolidate`` internally calls ``model.eval()`` and does not
        # restore it. Re-assert train mode afterwards so BN/dropout layers
        # optimise correctly (and are not silently corrupted) in the step loop.
        ewc_agent = self._build_ewc_anchor(ewc_lambda, candidate)
        candidate.train()

        candidate_params = [p for p in candidate.parameters() if p.requires_grad]

        final_loss = 0.0
        for _ in range(steps):
            output = candidate(batch)
            loss = self._task_loss_fn(output)
            if ewc_agent is not None:
                loss = loss + ewc_agent.compute_penalty()

            grads = torch.autograd.grad(loss, candidate_params)
            with torch.no_grad():
                for param, grad in zip(candidate_params, grads, strict=True):
                    param.add_(grad, alpha=-lr)

            final_loss = float(loss.detach())

        with torch.no_grad():
            candidate_state_dict = {
                name: param.detach().clone() for name, param in candidate.state_dict().items()
            }

        _log.info(
            "on_device_update_complete",
            steps=steps,
            final_loss=final_loss,
            ewc_lambda=ewc_lambda,
        )

        return OnDeviceUpdateResult(
            candidate_state_dict=candidate_state_dict,
            train_loss=final_loss,
            n_steps=steps,
            metadata={"learning_rate": lr, "ewc_lambda": ewc_lambda},
        )

    def _build_ewc_anchor(self, ewc_lambda: float, candidate: nn.Module) -> EWCAgent | None:
        """Build an EWC anchor on the candidate, weighted by ``ewc_lambda``.

        The penalty pulls the candidate back toward the consolidated base
        weights. ``ewc_lambda == 0`` disables regularization (the on-device
        config permits ``ge=0``, but ``LearningConfig.ewc_lambda`` is ``gt=0``),
        so we skip the agent entirely in that case rather than fabricate an
        out-of-range config.

        Args:
            ewc_lambda: On-device EWC penalty weight.
            candidate: The model whose drift is penalized.

        Returns:
            A consolidated :class:`EWCAgent`, or ``None`` when ``ewc_lambda`` is
            zero.
        """
        if ewc_lambda <= 0.0:
            return None

        # Reuse EWCAgent by threading the on-device weight through the shared
        # LearningConfig field it reads (``ewc_lambda``); all other Fisher
        # knobs keep their schema defaults.
        learning_cfg = LearningConfig(ewc_lambda=ewc_lambda)
        agent = EWCAgent(learning_cfg, candidate)
        agent.consolidate(self._fisher_loader)
        return agent


__all__ = ["EWCOnlineLearner", "TaskLossFn", "default_task_loss"]
