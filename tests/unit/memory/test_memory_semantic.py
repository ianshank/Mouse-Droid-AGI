from __future__ import annotations

import numpy as np
import pytest

from mousedroid.config.schema import MemoryConfig
from mousedroid.memory.semantic import SemanticIndex


@pytest.fixture
def cfg() -> MemoryConfig:
    return MemoryConfig(semantic_dim=8)


@pytest.fixture
def index(cfg: MemoryConfig) -> SemanticIndex:
    return SemanticIndex(cfg)


def test_constructor(index: SemanticIndex) -> None:
    assert index.size == 0


def test_store_increments_size(index: SemanticIndex) -> None:
    vec = np.random.randn(8).astype(np.float32)
    index.store("k1", vec)
    assert index.size == 1


def test_add_and_search(index: SemanticIndex) -> None:
    v1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    index.store("a", v1)
    index.store("b", v2)
    results = index.retrieve(v1, k=1)
    assert len(results) == 1
    assert results[0][0] == "a"


def test_retrieve_empty_index(index: SemanticIndex) -> None:
    query = np.random.randn(8).astype(np.float32)
    results = index.retrieve(query, k=5)
    assert results == []


def test_retrieve_k_larger_than_stored(index: SemanticIndex) -> None:
    index.store("only", np.random.randn(8).astype(np.float32))
    results = index.retrieve(np.random.randn(8).astype(np.float32), k=10)
    assert len(results) == 1


def test_multiple_stores_and_search(index: SemanticIndex) -> None:
    for i in range(5):
        vec = np.zeros(8, dtype=np.float32)
        vec[i] = 1.0
        index.store(f"vec_{i}", vec)
    assert index.size == 5
    query = np.zeros(8, dtype=np.float32)
    query[2] = 1.0
    results = index.retrieve(query, k=2)
    assert results[0][0] == "vec_2"
