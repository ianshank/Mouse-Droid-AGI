from __future__ import annotations

import pytest

from mousedroid.config.schema import MemoryConfig
from mousedroid.memory.episodic import EpisodicReplay


@pytest.fixture
def cfg() -> MemoryConfig:
    return MemoryConfig(episodic_capacity=10)


@pytest.fixture
def replay(cfg: MemoryConfig) -> EpisodicReplay:
    return EpisodicReplay(cfg)


def test_constructor(replay: EpisodicReplay) -> None:
    assert len(replay) == 0


def test_push_adds_to_buffer(replay: EpisodicReplay) -> None:
    replay.push("exp1")
    assert len(replay) == 1


def test_push_multiple(replay: EpisodicReplay) -> None:
    for i in range(5):
        replay.push(f"exp{i}")
    assert len(replay) == 5


def test_maxlen_is_respected(cfg: MemoryConfig) -> None:
    replay = EpisodicReplay(cfg)
    for i in range(20):
        replay.push(f"exp{i}")
    assert len(replay) == 10


def test_sample_returns_correct_batch_size(replay: EpisodicReplay) -> None:
    for i in range(10):
        replay.push(f"exp{i}")
    batch = replay.sample(3)
    assert len(batch) == 3


def test_sample_empty_buffer_returns_empty(replay: EpisodicReplay) -> None:
    result = replay.sample(5)
    assert result == []


def test_sample_more_than_available(replay: EpisodicReplay) -> None:
    replay.push("a")
    replay.push("b")
    result = replay.sample(10)
    assert len(result) == 2


def test_priority_weighted_sampling() -> None:
    cfg = MemoryConfig(episodic_capacity=100)
    replay = EpisodicReplay(cfg)
    replay.push("low", priority=0.0)
    replay.push("high", priority=100.0)
    # With extreme priority difference, high should be sampled most often
    counts = {"low": 0, "high": 0}
    for _ in range(50):
        result = replay.sample(1)
        counts[result[0]] += 1
    assert counts["high"] > counts["low"]


def test_safe_normalization_with_zero_priorities() -> None:
    cfg = MemoryConfig(episodic_capacity=10)
    replay = EpisodicReplay(cfg)
    replay.push("a", priority=0.0)
    replay.push("b", priority=0.0)
    replay.push("c", priority=0.0)
    # Should not crash; falls back to uniform sampling
    result = replay.sample(2)
    assert len(result) == 2


def test_push_with_inf_priority(replay: EpisodicReplay) -> None:
    replay.push("inf_exp", priority=float("inf"))
    assert len(replay) == 1
    result = replay.sample(1)
    assert result == ["inf_exp"]
