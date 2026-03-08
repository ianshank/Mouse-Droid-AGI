from __future__ import annotations

import pytest
import torch

from mousedroid.config.schema import MemoryConfig
from mousedroid.memory.working import WorkingMemory


@pytest.fixture
def cfg() -> MemoryConfig:
    return MemoryConfig(working_context_size=4)


@pytest.fixture
def wm(cfg: MemoryConfig) -> WorkingMemory:
    return WorkingMemory(cfg, embed_dim=16)


def test_constructor(wm: WorkingMemory) -> None:
    assert len(wm) == 0


def test_push_and_len(wm: WorkingMemory) -> None:
    wm.push(torch.randn(16))
    assert len(wm) == 1
    wm.push(torch.randn(16))
    assert len(wm) == 2


def test_context_size_limit(wm: WorkingMemory) -> None:
    for _ in range(10):
        wm.push(torch.randn(16))
    assert len(wm) == 4


def test_attend_empty_returns_zeros(wm: WorkingMemory) -> None:
    query = torch.randn(16)
    result = wm.attend(query)
    assert result.shape == (16,)
    assert (result == 0.0).all()


def test_attend_returns_correct_shape(wm: WorkingMemory) -> None:
    for _ in range(3):
        wm.push(torch.randn(16))
    query = torch.randn(16)
    result = wm.attend(query)
    assert result.shape == (16,)
    assert torch.isfinite(result).all()


def test_clear(wm: WorkingMemory) -> None:
    wm.push(torch.randn(16))
    wm.push(torch.randn(16))
    wm.clear()
    assert len(wm) == 0
