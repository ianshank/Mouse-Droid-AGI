from __future__ import annotations

from mousedroid.agents._planning import compute_mcts_budget


def test_zero_surprise_returns_base() -> None:
    result = compute_mcts_budget(surprise=0.0, base=50, maximum=200)
    assert result == 50


def test_high_surprise_increases_budget() -> None:
    base_result = compute_mcts_budget(surprise=0.0, base=50, maximum=200)
    high_result = compute_mcts_budget(surprise=2.0, base=50, maximum=200)
    assert high_result > base_result


def test_maximum_clamping() -> None:
    result = compute_mcts_budget(surprise=1000.0, base=50, maximum=200)
    assert result <= 200


def test_base_is_minimum() -> None:
    result = compute_mcts_budget(surprise=0.0, base=50, maximum=200)
    assert result >= 50


def test_moderate_surprise() -> None:
    result = compute_mcts_budget(surprise=1.0, base=50, maximum=200)
    assert 50 <= result <= 200
