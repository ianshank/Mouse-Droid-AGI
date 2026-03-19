"""Shared numpy operations — single source of truth for common activations.

DEPRECATED: This module is deprecated and will be removed in a future release.
Please use `mousedroid.common.math.numpy_ops` instead.
"""

from __future__ import annotations

import warnings

from mousedroid.common.math.numpy_ops import layer_norm, relu, softmax
from mousedroid.constants import SOFTMAX_EPSILON

warnings.warn(
    "mousedroid.utils.numpy_ops is deprecated. Use mousedroid.common.math.numpy_ops instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SOFTMAX_EPSILON", "layer_norm", "relu", "softmax"]
