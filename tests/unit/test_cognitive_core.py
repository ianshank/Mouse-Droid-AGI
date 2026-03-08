from __future__ import annotations

import numpy as np

from mousedroid.cognitive.bdi_model import NeuralBDI
from mousedroid.cognitive.cognitive_core import _SLOW_QUEUE_MAXSIZE, CognitiveCore
from mousedroid.cognitive.constitutional_rl import ConstitutionalChecker
from mousedroid.cognitive.metacognitive import MetacognitiveModel


def _make_core() -> CognitiveCore:
    bdi = NeuralBDI()
    metacog = MetacognitiveModel()
    checker = ConstitutionalChecker()
    return CognitiveCore(bdi, metacog, checker)


def test_constructor() -> None:
    core = _make_core()
    assert core is not None


def test_tick_fast_returns_tuple() -> None:
    core = _make_core()
    obs = {"state": np.zeros(128, dtype=np.float32)}
    result = core.tick_fast(obs)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_tick_fast_returns_ndarray_and_list() -> None:
    core = _make_core()
    obs = {"state": np.zeros(128, dtype=np.float32)}
    action, violations = core.tick_fast(obs)
    assert isinstance(action, np.ndarray)
    assert isinstance(violations, list)


def test_tick_fast_with_empty_obs() -> None:
    core = _make_core()
    action, _violations = core.tick_fast({})
    assert isinstance(action, np.ndarray)


async def test_start_stop_lifecycle() -> None:
    core = _make_core()
    await core.start()
    assert core._slow_task is not None
    assert not core._slow_task.done()
    await core.stop()
    assert core._slow_task.done()


async def test_stop_without_start() -> None:
    core = _make_core()
    await core.stop()  # Should not raise


def test_slow_queue_maxsize() -> None:
    assert _SLOW_QUEUE_MAXSIZE == 2


def test_tick_fast_with_curiosity_scores() -> None:
    core = _make_core()
    obs = {
        "state": np.zeros(128, dtype=np.float32),
        "curiosity": {"social": 0.5, "epistemic": 0.3, "perceptual": 0.2, "metacognitive": 0.1},
    }
    action, _violations = core.tick_fast(obs)
    assert isinstance(action, np.ndarray)
