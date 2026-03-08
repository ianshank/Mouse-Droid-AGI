from __future__ import annotations

import numpy as np

from mousedroid.cognitive.bdi_model import (
    _AFFECT_DIM,
    _BELIEF_DIM,
    _DESIRE_DIM,
    _INTENTION_CLASSES,
    BDIInput,
    BeliefEncoder,
    NeuralBDI,
    _safe_softmax,
)


def test_neural_bdi_constructor_default() -> None:
    bdi = NeuralBDI()
    assert bdi is not None


def test_neural_bdi_constructor_with_none_weights() -> None:
    bdi = NeuralBDI(weights_dir=None)
    assert bdi is not None


def test_infer_returns_expected_keys() -> None:
    bdi = NeuralBDI()
    obs = np.random.default_rng(0).standard_normal(256).astype(np.float32)
    result = bdi.infer(obs)
    assert set(result.keys()) == {"belief", "desire", "intentions", "affect", "approach_rate"}


def test_infer_belief_shape() -> None:
    bdi = NeuralBDI()
    obs = np.zeros(256, dtype=np.float32)
    result = bdi.infer(obs)
    assert result["belief"].shape == (_BELIEF_DIM,)


def test_infer_desire_shape() -> None:
    bdi = NeuralBDI()
    obs = np.zeros(256, dtype=np.float32)
    result = bdi.infer(obs)
    assert result["desire"].shape == (_DESIRE_DIM,)


def test_infer_intentions_shape() -> None:
    bdi = NeuralBDI()
    obs = np.zeros(256, dtype=np.float32)
    result = bdi.infer(obs)
    assert result["intentions"].shape == (_INTENTION_CLASSES,)


def test_infer_affect_shape() -> None:
    bdi = NeuralBDI()
    obs = np.zeros(256, dtype=np.float32)
    result = bdi.infer(obs)
    assert result["affect"].shape == (_AFFECT_DIM,)


def test_infer_approach_rate_positive() -> None:
    bdi = NeuralBDI()
    obs = np.zeros(256, dtype=np.float32)
    result = bdi.infer(obs)
    assert result["approach_rate"] > 0


def test_bdi_input_from_belief_state_basic() -> None:
    state = {"belief_state": np.ones(256, dtype=np.float32)}
    inp = BDIInput.from_belief_state(state)
    assert inp.belief_state.shape == (256,)
    assert inp.intentions.shape == (_INTENTION_CLASSES,)


def test_bdi_input_from_belief_state_with_intentions() -> None:
    intentions = np.ones(_INTENTION_CLASSES, dtype=np.float32) * 0.5
    state = {
        "belief_state": np.zeros(256, dtype=np.float32),
        "intentions": intentions,
    }
    inp = BDIInput.from_belief_state(state)
    np.testing.assert_allclose(inp.intentions, 0.5)


def test_safe_softmax_sums_to_one() -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    probs = _safe_softmax(logits)
    assert abs(float(np.sum(probs)) - 1.0) < 1e-5


def test_safe_softmax_large_logits_stable() -> None:
    logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    probs = _safe_softmax(logits)
    assert np.all(np.isfinite(probs))
    assert abs(float(np.sum(probs)) - 1.0) < 1e-5


def test_safe_softmax_all_zeros() -> None:
    logits = np.zeros(5, dtype=np.float32)
    probs = _safe_softmax(logits)
    assert np.all(np.isfinite(probs))


def test_dimension_constants() -> None:
    assert _BELIEF_DIM == 128
    assert _DESIRE_DIM == 64
    assert _INTENTION_CLASSES == 8
    assert _AFFECT_DIM == 2


def test_belief_encoder_default_init() -> None:
    enc = BeliefEncoder()
    out = enc.forward(np.zeros(256, dtype=np.float32))
    assert out.shape == (_BELIEF_DIM,)
