from __future__ import annotations

import numpy as np

from mousedroid.cognitive.metacognitive import (
    _CAPABILITY_NAMES,
    _N_CAPABILITIES,
    MetacognitiveModel,
    _apply_capped_penalty,
    _ema_blend,
)


def test_constructor_default() -> None:
    model = MetacognitiveModel()
    assert model is not None


def test_capability_vector_has_8_dims() -> None:
    model = MetacognitiveModel()
    summary = model.get_capability_summary()
    assert len(summary) == _N_CAPABILITIES
    assert _N_CAPABILITIES == 8


def test_initial_capabilities_all_ones() -> None:
    model = MetacognitiveModel()
    summary = model.get_capability_summary()
    for name in _CAPABILITY_NAMES:
        assert summary[name] == 1.0


def test_ema_blend_basic() -> None:
    old = np.array([1.0, 1.0], dtype=np.float32)
    new = np.array([0.0, 0.0], dtype=np.float32)
    result = _ema_blend(old, new, alpha=0.5)
    np.testing.assert_allclose(result, [0.5, 0.5])


def test_ema_blend_alpha_zero_keeps_old() -> None:
    old = np.array([1.0, 2.0], dtype=np.float32)
    new = np.array([5.0, 6.0], dtype=np.float32)
    result = _ema_blend(old, new, alpha=0.0)
    np.testing.assert_allclose(result, old)


def test_ema_blend_alpha_one_uses_new() -> None:
    old = np.array([1.0, 2.0], dtype=np.float32)
    new = np.array([5.0, 6.0], dtype=np.float32)
    result = _ema_blend(old, new, alpha=1.0)
    np.testing.assert_allclose(result, new)


def test_apply_capped_penalty_basic() -> None:
    value = np.array([0.5, 0.3], dtype=np.float32)
    penalty = np.array([0.2, 0.5], dtype=np.float32)
    result = _apply_capped_penalty(value, penalty, floor=0.0)
    np.testing.assert_allclose(result, [0.3, 0.0])


def test_apply_capped_penalty_with_floor() -> None:
    value = np.array([0.5], dtype=np.float32)
    penalty = np.array([1.0], dtype=np.float32)
    result = _apply_capped_penalty(value, penalty, floor=0.1)
    np.testing.assert_allclose(result, [0.1])


def test_update_with_battery() -> None:
    model = MetacognitiveModel(alpha=1.0, battery_nominal_v=12.6)
    model.update({"battery_v": 12.6})
    summary = model.get_capability_summary()
    assert 0.99 < summary["battery_management"] <= 1.0


def test_knowledge_graph_has_edges() -> None:
    model = MetacognitiveModel()
    downstream = model.propagate_degradation("vision")
    assert "navigation" in downstream
    assert "obstacle_avoidance" in downstream


def test_update_with_nav_score() -> None:
    model = MetacognitiveModel(alpha=1.0)
    model.update({"nav_score": 0.8})
    assert abs(model.get_capability_summary()["navigation"] - 0.8) < 1e-5


def test_update_with_obstacle_score() -> None:
    model = MetacognitiveModel(alpha=1.0)
    model.update({"obstacle_score": 0.7})
    assert abs(model.get_capability_summary()["obstacle_avoidance"] - 0.7) < 1e-5


def test_update_with_vision_score() -> None:
    model = MetacognitiveModel(alpha=1.0)
    model.update({"vision_score": 0.9})
    assert abs(model.get_capability_summary()["vision"] - 0.9) < 1e-5


def test_update_with_mcts_score() -> None:
    model = MetacognitiveModel(alpha=1.0)
    model.update({"mcts_score": 0.6})
    assert abs(model.get_capability_summary()["mcts"] - 0.6) < 1e-5


def test_update_with_bdi_score() -> None:
    model = MetacognitiveModel(alpha=1.0)
    model.update({"bdi_score": 0.5})
    assert abs(model.get_capability_summary()["bdi"] - 0.5) < 1e-5


def test_update_with_loop_time_ms() -> None:
    model = MetacognitiveModel(alpha=1.0)
    model.update({"loop_time_ms": 33.0})
    summary = model.get_capability_summary()
    assert 0.0 <= summary["loop_timing"] <= 1.0


def test_update_with_comm_score() -> None:
    model = MetacognitiveModel(alpha=1.0)
    model.update({"comm_score": 0.95})
    assert abs(model.get_capability_summary()["communication"] - 0.95) < 1e-5


def test_geometric_mean_all_ones() -> None:
    model = MetacognitiveModel()
    gm = model.geometric_mean()
    assert abs(gm - 1.0) < 1e-5
