"""VLA policy types, protocol, and ``MockVLA`` reference implementation.

This module is **import-graph-isolated**: it must not import any heavyweight
dependencies (``torch`` is the existing project-wide dependency; no
``onnxruntime``, no ``transformers`` here). The Phase 3b
``DistilledVLAOnnx`` implementation will live alongside ``MockVLA`` and
lazy-import its runtime in ``warmup``.

All thresholds, dimensions, and timeouts come from
:class:`mousedroid.config.schema.VLAConfig` /
:class:`mousedroid.config.schema.LoopConfig`. No values are hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import structlog
import torch

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class VLAObservation:
    """Inputs to a VLA policy.

    Wraps the current latent state plus an optional natural-language
    instruction. The orchestrator constructs this each tick.
    """

    h: torch.Tensor
    z: torch.Tensor
    instruction: str = ""


@dataclass(frozen=True)
class VLAAction:
    """Output of a VLA policy.

    Carries the action tensor (shape ``[action_dim]``) and an optional
    confidence in ``[0.0, 1.0]`` for downstream gating.
    """

    action: torch.Tensor
    confidence: float = 1.0


@runtime_checkable
class VLAPolicyProtocol(Protocol):
    """Protocol every VLA policy implementation must satisfy."""

    @property
    def name(self) -> str:
        """Human-readable policy name (used in telemetry)."""
        ...

    def predict(self, observation: VLAObservation) -> VLAAction:
        """Compute the next action for ``observation``.

        Implementations must run inference under ``torch.no_grad()`` and
        return a tensor with the configured ``model.action_dim``.
        """
        ...


class MockVLA:
    """Deterministic, zero-dependency VLA policy used for tests and dry-runs.

    The action returned is configurable; default is a zero action of the
    requested ``action_dim``. Useful as an upper-bound timing reference
    (effectively free) and as a structural test target for the
    orchestrator's VLA branch.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        canned_action: torch.Tensor | None = None,
        confidence: float = 1.0,
        name: str = "mock_vla",
    ) -> None:
        """Initialize a ``MockVLA``.

        Args:
            action_dim: Dimensionality of the action vector. Must match
                ``model.action_dim`` from the active settings.
            canned_action: Optional fixed action tensor. When ``None`` a
                zero action of length ``action_dim`` is returned.
            confidence: Confidence value emitted with each ``VLAAction``.
                Must be in ``[0.0, 1.0]``.
            name: Telemetry name; defaults to ``"mock_vla"``.

        Raises:
            ValueError: If ``action_dim`` is not positive, ``confidence``
                is outside ``[0.0, 1.0]``, or ``canned_action`` shape
                does not match ``action_dim``.
        """
        if action_dim <= 0:
            msg = f"action_dim must be > 0 (got {action_dim})"
            raise ValueError(msg)
        if not 0.0 <= confidence <= 1.0:
            msg = f"confidence must be in [0.0, 1.0] (got {confidence})"
            raise ValueError(msg)
        if canned_action is not None and canned_action.shape != (action_dim,):
            msg = (
                f"canned_action shape {tuple(canned_action.shape)} does not match "
                f"(action_dim,) = ({action_dim},)"
            )
            raise ValueError(msg)

        self._action_dim = action_dim
        self._canned_action = (
            canned_action.detach().clone()
            if canned_action is not None
            else torch.zeros(action_dim, dtype=torch.float32)
        )
        self._confidence = confidence
        self._name = name
        _log.debug(
            "mock_vla_initialized",
            action_dim=action_dim,
            confidence=confidence,
            name=name,
        )

    @property
    def name(self) -> str:
        """Telemetry name."""
        return self._name

    def predict(self, observation: VLAObservation) -> VLAAction:
        """Return the canned action regardless of ``observation``.

        Args:
            observation: Ignored. Accepted to satisfy the protocol.

        Returns:
            A :class:`VLAAction` with a fresh clone of the canned action.
        """
        del observation  # unused — MockVLA is stateless and deterministic
        with torch.no_grad():
            return VLAAction(
                action=self._canned_action.detach().clone(),
                confidence=self._confidence,
            )
