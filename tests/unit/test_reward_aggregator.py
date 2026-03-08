"""Tests for reward aggregation utilities."""

from __future__ import annotations

import torch

from mousedroid.reward.aggregator import pareto_dominates, pareto_weighted_sum


def test_pareto_weighted_sum_basic():
    objectives = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    weights = {"a": 1.0, "b": 1.0}
    result = pareto_weighted_sum(objectives, weights)
    # (1.0 * 0.5) + (2.0 * 0.5) = 1.5
    assert torch.isclose(result, torch.tensor(1.5))


def test_pareto_weighted_sum_normalizes_weights():
    objectives = {"x": torch.tensor(4.0)}
    weights = {"x": 2.0}
    result = pareto_weighted_sum(objectives, weights)
    # 4.0 * (2.0 / 2.0) = 4.0
    assert torch.isclose(result, torch.tensor(4.0))


def test_pareto_weighted_sum_zero_weights():
    objectives = {"a": torch.tensor(1.0)}
    weights = {"a": 0.0}
    result = pareto_weighted_sum(objectives, weights)
    assert result.item() == 0.0


def test_pareto_weighted_sum_missing_weight_uses_zero():
    objectives = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    weights = {"a": 1.0}  # "b" missing
    result = pareto_weighted_sum(objectives, weights)
    # a: 1.0 * (1.0/1.0) = 1.0, b: 2.0 * (0.0/1.0) = 0.0
    assert torch.isclose(result, torch.tensor(1.0))


def test_pareto_dominates_true():
    a = {"speed": 2.0, "safety": 3.0}
    b = {"speed": 1.0, "safety": 3.0}
    assert pareto_dominates(a, b) is True


def test_pareto_dominates_false_equal():
    a = {"speed": 1.0, "safety": 1.0}
    b = {"speed": 1.0, "safety": 1.0}
    assert pareto_dominates(a, b) is False


def test_pareto_dominates_false_mixed():
    a = {"speed": 2.0, "safety": 1.0}
    b = {"speed": 1.0, "safety": 2.0}
    assert pareto_dominates(a, b) is False
