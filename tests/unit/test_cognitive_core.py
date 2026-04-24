"""Tests for CognitiveCore — full coverage including slow loop."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from mousedroid.cognitive.bdi_model import NeuralBDI
from mousedroid.cognitive.cognitive_core import CognitiveCore
from mousedroid.cognitive.constitutional_rl import ConstitutionalChecker
from mousedroid.cognitive.metacognitive import MetacognitiveModel
from mousedroid.constants import SLOW_QUEUE_MAXSIZE


def _make_core() -> CognitiveCore:
    bdi = NeuralBDI()
    metacog = MetacognitiveModel()
    checker = ConstitutionalChecker()
    return CognitiveCore(bdi, metacog, checker)


def test_constructor() -> None:
    core = _make_core()
    assert core is not None


def test_get_latest_affect_returns_zero_before_first_inference() -> None:
    """Before the slow loop runs, the accessor returns neutral affect."""
    core = _make_core()
    assert core.latest_bdi == {}
    assert core.get_latest_affect() == (0.0, 0.0)


def test_get_latest_affect_returns_valence_arousal() -> None:
    core = _make_core()
    core._latest_bdi = {"affect": np.array([0.42, -0.18], dtype=np.float32)}
    valence, arousal = core.get_latest_affect()
    assert valence == pytest.approx(0.42, abs=1e-5)
    assert arousal == pytest.approx(-0.18, abs=1e-5)


def test_get_latest_affect_rejects_wrong_shape() -> None:
    """Malformed affect vectors fall back to neutral instead of crashing."""
    core = _make_core()
    core._latest_bdi = {"affect": np.array([0.5, 0.0, 0.0], dtype=np.float32)}
    assert core.get_latest_affect() == (0.0, 0.0)


def test_get_latest_affect_rejects_non_ndarray() -> None:
    core = _make_core()
    core._latest_bdi = {"affect": [0.5, 0.5]}
    assert core.get_latest_affect() == (0.0, 0.0)


def test_latest_bdi_property_exposes_dict() -> None:
    core = _make_core()
    sentinel = {"affect": np.zeros(2, dtype=np.float32), "intentions": np.zeros(4)}
    core._latest_bdi = sentinel
    assert core.latest_bdi is sentinel


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
    assert SLOW_QUEUE_MAXSIZE == 2


def test_tick_fast_with_curiosity_scores() -> None:
    core = _make_core()
    obs = {
        "state": np.zeros(128, dtype=np.float32),
        "curiosity": {"social": 0.5, "epistemic": 0.3, "perceptual": 0.2, "metacognitive": 0.1},
    }
    action, _violations = core.tick_fast(obs)
    assert isinstance(action, np.ndarray)


async def test_slow_loop_processes_observation() -> None:
    core = _make_core()
    await core.start()

    # Enqueue observation directly to avoid fast-path dimension mismatch.
    # The slow loop's BDI expects 256-d, but fast path uses 128-d policy.
    obs = {
        "state": np.random.default_rng(42).standard_normal(256).astype(np.float32),
        "battery_v": 12.0,
        "loop_time_ms": 25.0,
    }
    core._slow_queue.put_nowait(obs)

    # Give the slow loop time to process
    await asyncio.sleep(0.5)

    assert core._latest_bdi != {}
    assert "belief" in core._latest_bdi
    assert "intentions" in core._latest_bdi

    await core.stop()


async def test_tick_fast_queues_full_bdi_state_for_slow_loop() -> None:
    core = _make_core()
    core._bdi = MagicMock()
    core._bdi.infer.return_value = {
        "belief": np.zeros(128, dtype=np.float32),
        "intentions": np.zeros(10, dtype=np.float32),
    }

    await core.start()
    obs = {
        "state": np.zeros(128, dtype=np.float32),
        "bdi_state": np.ones(256, dtype=np.float32),
        "battery_v": 12.0,
        "loop_time_ms": 25.0,
    }
    core.tick_fast(obs)

    await asyncio.sleep(0.1)

    infer_arg = core._bdi.infer.call_args[0][0]
    assert infer_arg.shape == (256,)
    np.testing.assert_array_equal(infer_arg, obs["bdi_state"])

    await core.stop()


async def test_slow_loop_timeout_continues() -> None:
    core = _make_core()
    await core.start()
    # Don't put anything in the queue — the loop should timeout and continue
    await asyncio.sleep(0.1)
    assert not core._slow_task.done()
    await core.stop()


def test_tick_fast_with_context_keys() -> None:
    core = _make_core()
    obs = {
        "state": np.zeros(128, dtype=np.float32),
        "battery_v": 11.0,
        "obstacle_dist_m": 0.1,
        "mcts_sims": 4,
    }
    action, violations = core.tick_fast(obs)
    assert isinstance(action, np.ndarray)
    # Should have violations for low battery, close obstacle, low mcts_sims
    assert len(violations) > 0


async def test_start_idempotent() -> None:
    core = _make_core()
    await core.start()
    first_task = core._slow_task
    await core.start()  # Should not create a second task
    assert core._slow_task is first_task
    await core.stop()
