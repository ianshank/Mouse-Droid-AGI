"""Compact student policy + teacher adapter for VLA knowledge distillation.

The growth pillar distils the wired :class:`~mousedroid.vla.policy.VLAPolicyProtocol`
(a frozen, possibly ONNX-backed teacher) into a small, trainable torch student.
Two collaborators make that fit the classification-shaped
:class:`~mousedroid.growth.distillation.KnowledgeDistiller` (used under its
``objective="regression"`` mode):

- :class:`StudentVLAPolicy` — a compact MLP mapping the concatenated latent
  ``[h | z]`` to a continuous action vector. This is the model that is *grown*.
- :class:`VLATeacherModule` — a **paramless** ``nn.Module`` that wraps a
  ``VLAPolicyProtocol`` so the distiller can call ``teacher(x)`` uniformly. It
  holds no trainable parameters, so the distiller's teacher-freeze loop is a
  no-op and the student optimizer never touches teacher weights.

All dimensions come from :class:`mousedroid.config.schema.ModelConfig`
(``hidden_dim`` -> ``h``, ``latent_dim`` -> ``z``, ``action_dim``) — no values
are hardcoded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.vla.policy import VLAPolicyProtocol

_log = get_logger(__name__)


class StudentVLAPolicy(nn.Module):
    """Compact MLP student mapping concatenated latent ``[h | z]`` -> action.

    Args:
        h_dim: Deterministic (recurrent) latent dimension (``model.hidden_dim``).
        z_dim: Stochastic latent dimension (``model.latent_dim``).
        hidden_dim: Student hidden-layer width (``growth.student_hidden_dim``).
        action_dim: Output action dimension (``model.action_dim``).

    Raises:
        ValueError: If any dimension is not strictly positive.
    """

    def __init__(
        self,
        *,
        h_dim: int,
        z_dim: int,
        hidden_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        if min(h_dim, z_dim, hidden_dim, action_dim) <= 0:
            msg = (
                "StudentVLAPolicy dims must all be > 0 "
                f"(h={h_dim}, z={z_dim}, hidden={hidden_dim}, action={action_dim})"
            )
            raise ValueError(msg)
        self._h_dim = h_dim
        self._z_dim = z_dim
        self._action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(h_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    @property
    def obs_dim(self) -> int:
        """Concatenated observation dimension ``h_dim + z_dim``."""
        return self._h_dim + self._z_dim

    @property
    def action_dim(self) -> int:
        """Output action dimension."""
        return self._action_dim

    def forward(self, x: Tensor) -> Tensor:
        """Map ``x`` (shape ``[B, h_dim + z_dim]``) to actions ``[B, action_dim]``."""
        out: Tensor = self.net(x)
        return out


class VLATeacherModule(nn.Module):
    """Paramless ``nn.Module`` wrapping a :class:`VLAPolicyProtocol` teacher.

    Splits each concatenated ``[h | z]`` row back into a
    :class:`~mousedroid.vla.policy.VLAObservation`, runs the frozen policy under
    ``torch.no_grad``, and stacks the resulting action vectors so the distiller
    sees a uniform ``teacher(x) -> [B, action_dim]`` interface.

    The wrapped policy is stored as a plain attribute (not a registered
    submodule), so this module exposes **no** trainable parameters.

    Args:
        policy: The frozen teacher policy to distil from.
        h_dim: Deterministic latent dimension used to slice ``x``.
        z_dim: Stochastic latent dimension used to slice ``x``.
    """

    def __init__(
        self,
        policy: VLAPolicyProtocol,
        *,
        h_dim: int,
        z_dim: int,
    ) -> None:
        super().__init__()
        if min(h_dim, z_dim) <= 0:
            msg = f"VLATeacherModule dims must be > 0 (h={h_dim}, z={z_dim})"
            raise ValueError(msg)
        # Stored as a bare attribute — NOT via ``add_module`` — so no teacher
        # parameters are ever registered on this module.
        self._policy = policy
        self._h_dim = h_dim
        self._z_dim = z_dim

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Return teacher actions ``[B, action_dim]`` for latent rows ``x``."""
        from mousedroid.vla.policy import VLAObservation

        h = x[:, : self._h_dim]
        z = x[:, self._h_dim : self._h_dim + self._z_dim]
        actions = [
            self._policy.predict(VLAObservation(h=h[i], z=z[i])).action for i in range(x.shape[0])
        ]
        return torch.stack(actions, dim=0)


__all__ = ["StudentVLAPolicy", "VLATeacherModule"]
