"""Unit tests for teacher-student knowledge distillation.

Uses tiny linear teacher/student networks. Verifies the teacher is frozen on
init, and that a distillation step returns a finite scalar loss and updates
the student's parameters while leaving the teacher untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mousedroid.growth.distillation import KnowledgeDistiller


def _distiller() -> KnowledgeDistiller:
    torch.manual_seed(0)
    teacher = nn.Linear(4, 3)
    student = nn.Linear(4, 3)
    return KnowledgeDistiller(teacher, student, temperature=2.0, alpha=0.5, lr=0.05)


def test_init_freezes_teacher() -> None:
    """All teacher parameters have gradients disabled after construction."""
    distiller = _distiller()
    teacher = distiller._teacher
    assert all(not p.requires_grad for p in teacher.parameters())


def test_distill_step_updates_student_only() -> None:
    """A distillation step returns a finite loss and moves student params only."""
    distiller = _distiller()
    student = distiller._student
    teacher = distiller._teacher
    student_before = student.weight.detach().clone()
    teacher_before = teacher.weight.detach().clone()

    x = torch.randn(8, 4)
    labels = torch.randint(0, 3, (8,))
    loss = distiller.distill_step(x, labels)

    assert loss.item() == loss.item()  # not NaN
    assert not torch.equal(student.weight.detach(), student_before)
    assert torch.equal(teacher.weight.detach(), teacher_before)


def test_distill_step_loss_is_scalar() -> None:
    """The combined loss is a 0-dim differentiable tensor."""
    distiller = _distiller()
    loss = distiller.distill_step(torch.randn(4, 4), torch.randint(0, 3, (4,)))
    assert loss.ndim == 0
    assert loss.requires_grad
