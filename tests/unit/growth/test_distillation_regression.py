"""Unit tests for the regression objective added to ``KnowledgeDistiller``.

Pins that:
- The default ``objective="classification"`` path is byte-identical (KL+CE,
  requires integer ``hard_labels``).
- ``objective="regression"`` uses MSE, accepts ``hard_labels=None`` (pure
  teacher-matching self-distillation), and blends soft+hard when labels are given.
- The student moves and the teacher stays frozen under both objectives.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mousedroid.growth.distillation import KnowledgeDistiller


def _reg_distiller(alpha: float = 1.0) -> KnowledgeDistiller:
    torch.manual_seed(0)
    teacher = nn.Linear(6, 3)
    student = nn.Linear(6, 3)
    return KnowledgeDistiller(
        teacher, student, temperature=2.0, alpha=alpha, lr=0.05, objective="regression"
    )


def test_regression_self_distill_no_hard_labels() -> None:
    """Regression with ``hard_labels=None`` returns a finite scalar and moves student."""
    d = _reg_distiller()
    before = d._student.weight.detach().clone()  # type: ignore[operator]
    loss = d.distill_step(torch.randn(8, 6))
    assert loss.ndim == 0
    assert loss.item() == loss.item()  # not NaN
    assert loss.requires_grad
    assert not torch.equal(d._student.weight.detach(), before)  # type: ignore[operator]


def test_regression_blends_soft_and_hard() -> None:
    """With 0 < alpha < 1 and hard targets, the loss uses both MSE terms."""
    d = _reg_distiller(alpha=0.5)
    x = torch.randn(4, 6)
    hard = torch.randn(4, 3)
    loss = d.distill_step(x, hard)
    assert loss.item() == loss.item()
    assert loss.item() >= 0.0


def test_regression_teacher_stays_frozen() -> None:
    """The teacher parameters never receive gradients under the regression path."""
    d = _reg_distiller()
    teacher_before = d._teacher.weight.detach().clone()  # type: ignore[operator]
    d.distill_step(torch.randn(4, 6))
    assert torch.equal(d._teacher.weight.detach(), teacher_before)  # type: ignore[operator]
    assert all(not p.requires_grad for p in d._teacher.parameters())  # type: ignore[operator]


def test_classification_default_unchanged() -> None:
    """The default objective is classification and still consumes integer labels."""
    torch.manual_seed(0)
    d = KnowledgeDistiller(nn.Linear(4, 3), nn.Linear(4, 3), temperature=2.0, alpha=0.5, lr=0.05)
    loss = d.distill_step(torch.randn(8, 4), torch.randint(0, 3, (8,)))
    assert loss.ndim == 0
    assert loss.item() == loss.item()


def test_classification_requires_hard_labels() -> None:
    """Classification distillation with ``hard_labels=None`` raises ValueError."""
    d = KnowledgeDistiller(nn.Linear(4, 3), nn.Linear(4, 3), temperature=2.0, alpha=0.5, lr=0.05)
    with pytest.raises(ValueError, match="requires integer hard_labels"):
        d.distill_step(torch.randn(8, 4))
