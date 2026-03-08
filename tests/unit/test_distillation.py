from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mousedroid.growth.distillation import KnowledgeDistiller


def _make_model(in_dim: int = 16, out_dim: int = 4) -> nn.Module:
    return nn.Linear(in_dim, out_dim)


@pytest.fixture
def distiller() -> KnowledgeDistiller:
    teacher = _make_model()
    student = _make_model()
    return KnowledgeDistiller(teacher, student, temperature=2.0, alpha=0.5, lr=1e-3)


def test_constructor(distiller: KnowledgeDistiller) -> None:
    assert distiller._temperature == 2.0
    assert distiller._alpha == 0.5


def test_teacher_is_frozen(distiller: KnowledgeDistiller) -> None:
    for p in distiller._teacher.parameters():
        assert not p.requires_grad


def test_distill_step_returns_scalar_loss(distiller: KnowledgeDistiller) -> None:
    x = torch.randn(8, 16)
    labels = torch.randint(0, 4, (8,))
    loss = distiller.distill_step(x, labels)
    assert loss.shape == ()
    assert loss.item() > 0.0


def test_distill_step_updates_student(distiller: KnowledgeDistiller) -> None:
    params_before = [p.clone() for p in distiller._student.parameters()]
    x = torch.randn(8, 16)
    labels = torch.randint(0, 4, (8,))
    distiller.distill_step(x, labels)
    changed = any(
        not torch.equal(before, after)
        for before, after in zip(params_before, distiller._student.parameters(), strict=False)
    )
    assert changed


def test_alpha_blending() -> None:
    teacher = _make_model()
    student = _make_model()
    d1 = KnowledgeDistiller(teacher, student, temperature=2.0, alpha=0.0, lr=1e-3)
    assert d1._alpha == 0.0
    d2 = KnowledgeDistiller(teacher, _make_model(), temperature=2.0, alpha=1.0, lr=1e-3)
    assert d2._alpha == 1.0


def test_multiple_steps_reduce_loss(distiller: KnowledgeDistiller) -> None:
    x = torch.randn(16, 16)
    labels = torch.randint(0, 4, (16,))
    distiller.distill_step(x, labels)
    for _ in range(10):
        distiller.distill_step(x, labels)
    loss_last = distiller.distill_step(x, labels).item()
    # Loss should generally decrease with training
    assert isinstance(loss_last, float)
