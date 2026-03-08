"""Tests for InContextLearner."""

from __future__ import annotations

from mousedroid.meta.in_context import InContextLearner


def test_constructor():
    learner = InContextLearner(max_context_examples=5)
    assert learner._max_examples == 5
    assert len(learner) == 0


def test_add_example():
    learner = InContextLearner(max_context_examples=3)
    learner.add_example("input1", "output1")
    assert len(learner) == 1


def test_add_example_evicts_oldest():
    learner = InContextLearner(max_context_examples=2)
    learner.add_example("a", 1)
    learner.add_example("b", 2)
    learner.add_example("c", 3)
    assert len(learner) == 2
    # The oldest ("a") should have been evicted
    assert learner._context[0]["input"] == "b"


def test_build_prompt():
    learner = InContextLearner(max_context_examples=5)
    learner.add_example("x", "y")
    prompt = learner.build_prompt("query")
    assert len(prompt) == 2
    assert prompt[0] == {"input": "x", "output": "y"}
    assert prompt[1] == {"input": "query", "output": None}


def test_adapt_bulk_loads():
    learner = InContextLearner(max_context_examples=10)
    learner.add_example("old", "data")
    learner.adapt([("a", 1), ("b", 2), ("c", 3)])
    assert len(learner) == 3
    # Old data should be gone
    assert learner._context[0]["input"] == "a"


def test_clear():
    learner = InContextLearner(max_context_examples=5)
    learner.add_example("x", "y")
    learner.clear()
    assert len(learner) == 0


def test_build_prompt_empty_context():
    learner = InContextLearner(max_context_examples=5)
    prompt = learner.build_prompt("q")
    assert len(prompt) == 1
    assert prompt[0]["input"] == "q"
