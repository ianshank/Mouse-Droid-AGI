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


def _resolve_action_bounds(
    expected_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    action_min: torch.Tensor | None,
    action_max: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-dimension clamp bounds aligned to the target tensor."""
    lower = action_min
    upper = action_max
    if lower is None:
        lower = torch.full((expected_dim,), -1.0, dtype=dtype, device=device)
    else:
        lower = lower.detach().flatten().to(device=device, dtype=dtype)
    if upper is None:
        upper = torch.full((expected_dim,), 1.0, dtype=dtype, device=device)
    else:
        upper = upper.detach().flatten().to(device=device, dtype=dtype)

    if lower.numel() != expected_dim or upper.numel() != expected_dim:
        msg = "action bounds must match expected_dim"
        raise ValueError(msg)
    return lower, upper


def normalize_action_tensor(
    action: torch.Tensor,
    expected_dim: int,
    action_min: torch.Tensor | None = None,
    action_max: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalise a torch action tensor to the expected dimensionality.

    Pads with zeros if too small, truncates if too large, and clamps to
    the configured per-dimension bounds.

    Args:
        action: Raw action tensor (any shape, will be flattened).
        expected_dim: Required output dimensionality.
        action_min: Optional per-dimension lower clamp bounds.
        action_max: Optional per-dimension upper clamp bounds.

    Returns:
        1-D torch tensor of shape ``(expected_dim,)`` within the requested bounds.
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
    lower, upper = _resolve_action_bounds(
        expected_dim,
        device=flat.device,
        dtype=flat.dtype,
        action_min=action_min,
        action_max=action_max,
    )
    return torch.max(torch.min(flat, upper), lower)


def normalize_action_numpy(
    action: NDArray[np.floating[Any]],
    expected_dim: int,
    action_min: NDArray[np.floating[Any]] | list[float] | None = None,
    action_max: NDArray[np.floating[Any]] | list[float] | None = None,
) -> torch.Tensor:
    """Normalise a numpy action array to a clamped torch tensor.

    Convenience wrapper that converts numpy -> torch then delegates to
    :func:`normalize_action_tensor`.

    Args:
        action: Raw action as numpy array.
        expected_dim: Required output dimensionality.
        action_min: Optional per-dimension lower clamp bounds.
        action_max: Optional per-dimension upper clamp bounds.

    Returns:
        1-D torch tensor of shape ``(expected_dim,)`` within the requested bounds.
    """
    arr = np.asarray(action, dtype=np.float32).flatten()
    lower = None if action_min is None else torch.as_tensor(action_min, dtype=torch.float32)
    upper = None if action_max is None else torch.as_tensor(action_max, dtype=torch.float32)
    return normalize_action_tensor(
        torch.from_numpy(arr),
        expected_dim,
        action_min=lower,
        action_max=upper,
    )
