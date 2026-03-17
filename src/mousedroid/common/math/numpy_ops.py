"""Shared numpy operations — single source of truth for common activations.

Avoids duplicate implementations across bdi_model, constitutional_rl, and
training scripts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_SOFTMAX_EPS: float = 1e-8
"""Epsilon for softmax numerical stability."""


def relu(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Element-wise ReLU activation.

    Args:
        x: Input array.

    Returns:
        Array with negative values zeroed.
    """
    result: NDArray[np.floating[Any]] = np.maximum(x, 0.0)
    return result


def softmax(x: NDArray[np.floating[Any]], *, axis: int = -1) -> NDArray[np.floating[Any]]:
    """Numerically stable softmax.

    Args:
        x: Input logits array.
        axis: Axis along which to compute softmax.

    Returns:
        Probability distribution along *axis*.
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    result: NDArray[np.floating[Any]] = e / (e.sum(axis=axis, keepdims=True) + _SOFTMAX_EPS)
    return result


def layer_norm(x: NDArray[np.floating[Any]], *, eps: float = 1e-6) -> NDArray[np.floating[Any]]:
    """Simple layer normalisation (zero mean, unit variance).

    Args:
        x: Input array.
        eps: Epsilon for numerical stability.

    Returns:
        Normalised array.
    """
    mean = np.mean(x)
    var = np.var(x)
    result: NDArray[np.floating[Any]] = (x - mean) / np.sqrt(var + eps)
    return result
