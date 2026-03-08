"""Tests for MemoryConsolidation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from mousedroid.config.schema import MemoryConfig
from mousedroid.memory.consolidation import MemoryConsolidation


def _make_consolidation(batch_size: int = 4) -> MemoryConsolidation:
    cfg = MemoryConfig(consolidation_batch_size=batch_size)
    episodic = MagicMock()
    semantic = MagicMock()
    return MemoryConsolidation(cfg, episodic, semantic)


def test_constructor():
    c = _make_consolidation(batch_size=8)
    assert c._batch_size == 8
    assert c._consolidation_count == 0


def test_consolidate_with_embeddings():
    cfg = MemoryConfig(consolidation_batch_size=2)
    episodic = MagicMock()
    semantic = MagicMock()

    # Create experiences with embedding attributes
    exp1 = SimpleNamespace(embedding=np.ones(256, dtype=np.float32))
    exp2 = SimpleNamespace(embedding=np.zeros(256, dtype=np.float32))
    episodic.sample.return_value = [exp1, exp2]

    c = MemoryConsolidation(cfg, episodic, semantic)
    count = c.consolidate()

    assert count == 2
    assert c._consolidation_count == 2
    assert semantic.store.call_count == 2


def test_consolidate_skips_experiences_without_embedding():
    cfg = MemoryConfig(consolidation_batch_size=2)
    episodic = MagicMock()
    semantic = MagicMock()

    exp_good = SimpleNamespace(embedding=np.ones(4, dtype=np.float32))
    exp_bad = SimpleNamespace()  # no embedding attr
    episodic.sample.return_value = [exp_good, exp_bad]

    c = MemoryConsolidation(cfg, episodic, semantic)
    count = c.consolidate()

    assert count == 1
    assert semantic.store.call_count == 1


def test_consolidate_empty_batch():
    cfg = MemoryConfig(consolidation_batch_size=4)
    episodic = MagicMock()
    semantic = MagicMock()
    episodic.sample.return_value = []

    c = MemoryConsolidation(cfg, episodic, semantic)
    count = c.consolidate()
    assert count == 0


def test_extract_embedding_returns_none_for_no_attr():
    result = MemoryConsolidation._extract_embedding(object())
    assert result is None


def test_extract_embedding_returns_array():
    exp = SimpleNamespace(embedding=[1.0, 2.0, 3.0])
    result = MemoryConsolidation._extract_embedding(exp)
    assert result is not None
    assert result.dtype == np.float32
