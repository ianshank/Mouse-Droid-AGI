"""Unit tests for the growth student + VLA teacher adapter."""

from __future__ import annotations

import pytest
import torch

from mousedroid.growth.student import StudentVLAPolicy, VLATeacherModule
from mousedroid.vla.policy import MockVLA, VLAObservation


def test_student_forward_shape() -> None:
    """The student maps ``[B, h+z]`` to ``[B, action_dim]``."""
    student = StudentVLAPolicy(h_dim=5, z_dim=3, hidden_dim=16, action_dim=4)
    assert student.obs_dim == 8
    assert student.action_dim == 4
    out = student(torch.randn(7, 8))
    assert out.shape == (7, 4)


@pytest.mark.parametrize(
    ("h", "z", "hid", "act"),
    [(0, 3, 16, 3), (5, 0, 16, 3), (5, 3, 0, 3), (5, 3, 16, 0)],
)
def test_student_rejects_nonpositive_dims(h: int, z: int, hid: int, act: int) -> None:
    """Any non-positive dimension raises ValueError."""
    with pytest.raises(ValueError, match="must all be > 0"):
        StudentVLAPolicy(h_dim=h, z_dim=z, hidden_dim=hid, action_dim=act)


def test_teacher_module_has_no_parameters() -> None:
    """The VLA teacher wrapper registers no trainable parameters."""
    teacher = VLATeacherModule(MockVLA(action_dim=3), h_dim=5, z_dim=3)
    assert list(teacher.parameters()) == []


def test_teacher_module_stacks_actions() -> None:
    """``teacher(x)`` returns ``[B, action_dim]`` matching per-row VLA predictions."""
    canned = torch.tensor([0.1, -0.2, 0.3])
    teacher = VLATeacherModule(MockVLA(action_dim=3, canned_action=canned), h_dim=5, z_dim=3)
    x = torch.randn(4, 8)
    out = teacher(x)
    assert out.shape == (4, 3)
    # MockVLA returns the canned action for every row.
    for i in range(4):
        assert torch.allclose(out[i], canned)


def test_teacher_module_slices_h_and_z() -> None:
    """The wrapper splits ``x`` into the configured h/z widths before predicting."""
    seen: list[tuple[int, int]] = []

    class _SpyVLA:
        @property
        def name(self) -> str:
            return "spy"

        def predict(self, observation: VLAObservation):  # type: ignore[no-untyped-def]
            seen.append((observation.h.shape[0], observation.z.shape[0]))
            return MockVLA(action_dim=2).predict(observation)

    teacher = VLATeacherModule(_SpyVLA(), h_dim=5, z_dim=3)
    teacher(torch.randn(2, 8))
    assert seen == [(5, 3), (5, 3)]
