"""Unit tests for the in-context (prompt-based) learner.

Pure-Python module (no torch): covers the sliding context window, prompt
construction, bulk adaptation, and clearing.
"""

from __future__ import annotations

from mousedroid.meta.in_context import InContextLearner


def test_add_example_evicts_oldest_when_full() -> None:
    """The window is bounded; adding past capacity drops the oldest example."""
    learner = InContextLearner(max_context_examples=2)
    learner.add_example("a", 1)
    learner.add_example("b", 2)
    learner.add_example("c", 3)
    assert len(learner) == 2
    prompt = learner.build_prompt("q")
    inputs = [entry["input"] for entry in prompt]
    assert inputs == ["b", "c", "q"]  # "a" evicted, query appended last


def test_build_prompt_appends_query_with_null_output() -> None:
    """The query is appended as the final entry with a ``None`` output slot."""
    learner = InContextLearner(max_context_examples=5)
    learner.add_example("obs", "act")
    prompt = learner.build_prompt("current")
    assert prompt[-1] == {"input": "current", "output": None}
    assert prompt[0] == {"input": "obs", "output": "act"}


def test_adapt_replaces_previous_context() -> None:
    """``adapt`` clears prior examples before loading the new support set."""
    learner = InContextLearner(max_context_examples=10)
    learner.add_example("stale", 0)
    learner.adapt([("x", 1), ("y", 2)])
    assert len(learner) == 2
    inputs = [entry["input"] for entry in learner.build_prompt("q")[:-1]]
    assert inputs == ["x", "y"]


def test_adapt_respects_capacity() -> None:
    """Bulk adaptation still honours the max-context bound."""
    learner = InContextLearner(max_context_examples=2)
    learner.adapt([("a", 1), ("b", 2), ("c", 3)])
    assert len(learner) == 2


def test_clear_empties_context() -> None:
    """``clear`` removes all examples."""
    learner = InContextLearner(max_context_examples=3)
    learner.add_example("a", 1)
    learner.clear()
    assert len(learner) == 0
    assert learner.build_prompt("q") == [{"input": "q", "output": None}]
