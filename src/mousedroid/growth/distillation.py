"""Knowledge distillation — teacher to student transfer."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


def _backward(loss: Tensor) -> None:
    """Isolate torch stub gaps around ``Tensor.backward``."""
    loss.backward()  # type: ignore[no-untyped-call]


class KnowledgeDistiller:
    """Teacher-student knowledge distillation with configurable temperature.

    Loss is a weighted combination of:
    - KL divergence between teacher and student soft targets (scaled by T^2).
    - Cross-entropy between student predictions and hard labels.

    Args:
        teacher: Pre-trained teacher model.
        student: Student model to train.
        temperature: Softmax temperature for soft targets.
        alpha: Weight of the distillation (KL) loss vs hard-label (CE) loss.
        lr: Learning rate for the student optimizer.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float,
        alpha: float,
        lr: float,
    ) -> None:
        self._teacher = teacher
        self._student = student
        self._temperature = temperature
        self._alpha = alpha
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
        )

    def distill_step(self, x: Tensor, hard_labels: Tensor) -> Tensor:
        """One distillation training step.

        Args:
            x: Input batch tensor.
            hard_labels: Ground-truth labels for the CE component.

        Returns:
            Combined scalar loss (differentiable).
        """
        self._teacher.eval()
        self._student.train()

        # Teacher forward (no gradients).
        teacher_logits: Tensor = self._teacher(x).detach()

        # Student forward.
        student_logits: Tensor = self._student(x)

        # Soft-target KL divergence loss.
        t = self._temperature
        teacher_soft = F.log_softmax(teacher_logits / t, dim=-1)
        student_soft = F.log_softmax(student_logits / t, dim=-1)
        kl_loss = F.kl_div(student_soft, teacher_soft.exp(), reduction="batchmean") * (t * t)

        # Hard-label cross-entropy loss.
        ce_loss = F.cross_entropy(student_logits, hard_labels)

        # Combined loss.
        loss: Tensor = self._alpha * kl_loss + (1.0 - self._alpha) * ce_loss

        self._student_optimizer.zero_grad()
        _backward(loss)
        self._student_optimizer.step()

        return loss
