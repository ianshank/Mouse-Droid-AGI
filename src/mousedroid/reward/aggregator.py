"""Reward aggregation utilities — Pareto-weighted multi-objective blending."""

from __future__ import annotations

import torch
from torch import Tensor

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


def pareto_weighted_sum(
    objectives: dict[str, Tensor],
    weights: dict[str, float],
) -> Tensor:
    """Compute a Pareto-weighted sum of objective tensors.

    Weights are normalized to sum to 1 before aggregation so the caller
    does not need to guarantee normalization.

    Args:
        objectives: Mapping of objective name to score tensor.
        weights: Mapping of objective name to raw weight.

    Returns:
        Aggregated scalar reward tensor.
    """
    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        _log.warning("pareto_zero_weights", total_weight=total_weight)
        return torch.zeros_like(next(iter(objectives.values())))

    result = torch.zeros_like(next(iter(objectives.values())))
    for name, score in objectives.items():
        w = weights.get(name, 0.0) / total_weight
        result = result + w * score
    return result


def pareto_dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """Check whether solution *a* Pareto-dominates solution *b*.

    *a* dominates *b* iff *a* is at least as good on every objective and
    strictly better on at least one.

    Args:
        a: Objective scores for candidate A.
        b: Objective scores for candidate B.

    Returns:
        ``True`` if *a* dominates *b*.
    """
    at_least_as_good = all(a[k] >= b[k] for k in a)
    strictly_better = any(a[k] > b[k] for k in a)
    return at_least_as_good and strictly_better
