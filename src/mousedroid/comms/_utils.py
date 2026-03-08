"""Shared utilities for communication drivers."""

from __future__ import annotations


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value between lo and hi.

    Args:
        value: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped value.
    """
    return max(lo, min(hi, value))
