"""Shared numpy operations — single source of truth for common activations.

DEPRECATED: This module is deprecated and will be removed in a future release.
Please use `mousedroid.common.math.numpy_ops` instead.
"""

from __future__ import annotations

import warnings

from mousedroid.common.math.numpy_ops import _SOFTMAX_EPS, layer_norm, relu, softmax

warnings.warn(
    "mousedroid.utils.numpy_ops is deprecated. Use mousedroid.common.math.numpy_ops instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["_SOFTMAX_EPS", "layer_norm", "relu", "softmax"]
