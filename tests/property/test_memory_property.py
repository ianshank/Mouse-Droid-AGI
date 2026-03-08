from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import MemoryConfig
from mousedroid.memory.episodic import EpisodicReplay


def _make_replay(capacity: int = 100) -> EpisodicReplay:
    cfg = MemoryConfig(episodic_capacity=capacity)
    return cfg, EpisodicReplay(cfg)


@given(
    n_items=st.integers(min_value=0, max_value=200),
)
@settings(max_examples=20)
def test_never_exceeds_capacity(n_items: int) -> None:
    capacity = 50
    _, replay = _make_replay(capacity)
    for i in range(n_items):
        replay.push(f"exp_{i}", priority=1.0)
    assert len(replay) <= capacity


@given(
    batch_size=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=20)
def test_sample_returns_at_most_batch_size(batch_size: int) -> None:
    _, replay = _make_replay(200)
    for i in range(50):
        replay.push(f"exp_{i}", priority=1.0)
    result = replay.sample(batch_size)
    assert len(result) <= batch_size


@given(
    priority=st.floats(allow_nan=True, allow_infinity=True),
)
@settings(max_examples=30)
def test_push_any_priority_no_crash(priority: float) -> None:
    _, replay = _make_replay(10)
    replay.push("data", priority=priority)
    assert len(replay) == 1


def test_sample_empty_buffer_returns_empty() -> None:
    _, replay = _make_replay(10)
    assert replay.sample(5) == []


def test_push_and_sample_roundtrip() -> None:
    _, replay = _make_replay(10)
    for i in range(5):
        replay.push(f"item_{i}", priority=1.0)
    sampled = replay.sample(3)
    assert len(sampled) == 3
    for item in sampled:
        assert item.startswith("item_")
