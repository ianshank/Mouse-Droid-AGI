"""Shared MCTS budget computation — DRY helper for surprise-adaptive planning."""

from __future__ import annotations

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


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
    budget = min(max(int(base * scale), base), maximum)
    _log.debug("mcts_budget_computed", surprise=surprise, budget=budget, base=base, maximum=maximum)
    return budget
