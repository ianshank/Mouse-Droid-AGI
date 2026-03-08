"""Shared MCTS budget computation — DRY helper for surprise-adaptive planning."""

from __future__ import annotations


def compute_mcts_budget(surprise: float, base: int, maximum: int) -> int:
    """Compute surprise-adaptive MCTS simulation budget.

    Higher surprise -> more simulations for better planning.

    Args:
        surprise: Current world model surprise value (>= 0).
        base: Base number of MCTS simulations.
        maximum: Maximum number of MCTS simulations.

    Returns:
        Number of simulations to run, clamped to ``[base, maximum]``.
    """
    scale = min(surprise + 1.0, float(maximum) / max(float(base), 1.0))
    return min(max(int(base * scale), base), maximum)
