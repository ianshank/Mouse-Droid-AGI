"""Knowledge distillation — teacher to student transfer."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — canonical PyTorch alias for functional
from torch import Tensor

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

#: Distillation objective. ``"classification"`` (default) preserves the legacy
#: KL(soft) + CE(hard) loss over class logits byte-identically. ``"regression"``
#: swaps both terms for MSE so a continuous-output teacher (e.g. a VLA action
#: policy) can be distilled into a compact student — the growth-pillar wiring.
DistillObjective = Literal["classification", "regression"]


def _backward(loss: Tensor) -> None:
    """Isolate torch stub gaps around ``Tensor.backward``."""
    loss.backward()  # type: ignore[no-untyped-call]  # torch ships no stub for Tensor.backward


class KnowledgeDistiller:
    """Teacher-student knowledge distillation with configurable temperature.

    Two objectives are supported, selected at construction:

    - ``"classification"`` (default, unchanged): loss is a weighted combination
      of KL divergence between teacher and student soft targets (scaled by
      ``T**2``) and cross-entropy between student predictions and hard labels.
    - ``"regression"``: loss is a weighted combination of MSE against the
      teacher's continuous outputs (soft term) and MSE against ground-truth
      targets (hard term). ``hard_labels`` may be ``None`` for pure
      teacher-matching self-distillation (soft term only), which is how the
      growth pillar distills the VLA action policy.

    Args:
        teacher: Pre-trained teacher model (frozen on init). May be a paramless
            adapter around a non-``nn.Module`` policy — the freeze loop is then
            a harmless no-op and the student optimizer never sees teacher params.
        student: Student model to train.
        temperature: Softmax temperature for soft targets (classification only;
            ignored under the regression objective).
        alpha: Weight of the distillation (soft) loss vs the hard-target loss.
        lr: Learning rate for the student optimizer.
        objective: ``"classification"`` (default) or ``"regression"``.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float,
        alpha: float,
        lr: float,
        *,
        objective: DistillObjective = "classification",
    ) -> None:
        self._teacher = teacher
        self._student = student
        self._temperature = temperature
        self._alpha = alpha
        self._objective: DistillObjective = objective
        self._student_optimizer = __import__("torch").optim.Adam(
            student.parameters(),
            lr=lr,
        )

        # Freeze teacher.
        for param in self._teacher.parameters():
            param.requires_grad = False

        _log.info(
            "distiller_init",
            temperature=temperature,
            alpha=alpha,
            lr=lr,
            objective=objective,
        )

    def distill_step(self, x: Tensor, hard_labels: Tensor | None = None) -> Tensor:
        """One distillation training step.

        Args:
            x: Input batch tensor.
            hard_labels: Ground-truth targets for the hard-target component.
                Required under ``"classification"`` (integer class labels for
                cross-entropy); optional under ``"regression"`` (``None`` uses
                the soft/teacher-matching term alone).

        Returns:
            Combined scalar loss (differentiable).

        Raises:
            ValueError: If the objective is ``"classification"`` and
                ``hard_labels`` is ``None``.
        """
        self._teacher.eval()
        self._student.train()

        # Teacher forward (no gradients).
        teacher_out: Tensor = self._teacher(x).detach()

        # Student forward.
        student_out: Tensor = self._student(x)

        if self._objective == "regression":
            loss = self._regression_loss(student_out, teacher_out, hard_labels)
        else:
            if hard_labels is None:
                msg = "classification distillation requires integer hard_labels"
                raise ValueError(msg)
            loss = self._classification_loss(student_out, teacher_out, hard_labels)

        self._student_optimizer.zero_grad()
        _backward(loss)
        self._student_optimizer.step()

        return loss

    def _classification_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        hard_labels: Tensor,
    ) -> Tensor:
        """Legacy KL(soft) + CE(hard) loss over class logits (unchanged)."""
        t = self._temperature
        teacher_soft = F.log_softmax(teacher_logits / t, dim=-1)
        student_soft = F.log_softmax(student_logits / t, dim=-1)
        kl_loss = F.kl_div(student_soft, teacher_soft.exp(), reduction="batchmean") * (t * t)
        ce_loss = F.cross_entropy(student_logits, hard_labels)
        return self._alpha * kl_loss + (1.0 - self._alpha) * ce_loss

    def _regression_loss(
        self,
        student_out: Tensor,
        teacher_out: Tensor,
        hard_labels: Tensor | None,
    ) -> Tensor:
        """MSE(teacher soft) [+ MSE(hard)] loss for continuous-output distillation."""
        soft_loss = F.mse_loss(student_out, teacher_out)
        if hard_labels is None:
            return soft_loss
        hard_loss = F.mse_loss(student_out, hard_labels)
        return self._alpha * soft_loss + (1.0 - self._alpha) * hard_loss
