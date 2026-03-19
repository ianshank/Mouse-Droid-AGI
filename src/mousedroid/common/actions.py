"""Reusable action normalisation utilities.

Centralises action clamping, padding, and truncation logic used by
agents, the orchestrator, and planning modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


def normalize_action_tensor(action: torch.Tensor, expected_dim: int) -> torch.Tensor:
    """Normalise a torch action tensor to the expected dimensionality.

    Pads with zeros if too small, truncates if too large, and clamps to [-1, 1].

    Args:
        action: Raw action tensor (any shape, will be flattened).
        expected_dim: Required output dimensionality.

    Returns:
        1-D torch tensor of shape ``(expected_dim,)`` with values in ``[-1, 1]``.
    """
    flat = action.detach().flatten()
    if flat.numel() < expected_dim:
        _log.debug(
            "action_padded",
            received_dim=flat.numel(),
            expected_dim=expected_dim,
        )
        flat = torch.nn.functional.pad(flat, (0, expected_dim - flat.numel()))
    elif flat.numel() > expected_dim:
        _log.debug(
            "action_truncated",
            received_dim=flat.numel(),
            expected_dim=expected_dim,
        )
        flat = flat[:expected_dim]
    return torch.clamp(flat, -1.0, 1.0)


def normalize_action_numpy(action: NDArray[np.floating[Any]], expected_dim: int) -> torch.Tensor:
    """Normalise a numpy action array to a clamped torch tensor.

    Convenience wrapper that converts numpy -> torch then delegates to
    :func:`normalize_action_tensor`.

    Args:
        action: Raw action as numpy array.
        expected_dim: Required output dimensionality.

    Returns:
        1-D torch tensor of shape ``(expected_dim,)`` with values in ``[-1, 1]``.
    """
    arr = np.asarray(action, dtype=np.float32).flatten()
    return normalize_action_tensor(torch.from_numpy(arr), expected_dim)
