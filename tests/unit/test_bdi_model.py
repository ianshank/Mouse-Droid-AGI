"""Tests for BDI model — full coverage including weight loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mousedroid.cognitive.bdi_model import (
    _AFFECT_DIM,
    _BELIEF_DIM,
    _DESIRE_DIM,
    _INTENTION_CLASSES,
    AffectEstimator,
    BDIInput,
    BeliefEncoder,
    DesireEncoder,
    IntentionPredictor,
    NeuralBDI,
    _bayesian_normalise,
    _relu,
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
    assert _INTENTION_CLASSES == 10
    assert _AFFECT_DIM == 2


def test_belief_encoder_default_init() -> None:
    enc = BeliefEncoder()
    out = enc.forward(np.zeros(256, dtype=np.float32))
    assert out.shape == (_BELIEF_DIM,)


# -- Weight loading tests --


def test_bayesian_normalise_basic() -> None:
    values = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    result = _bayesian_normalise(values)
    assert abs(float(np.sum(result)) - 1.0) < 1e-5


def test_bayesian_normalise_all_zeros() -> None:
    values = np.zeros(4, dtype=np.float32)
    result = _bayesian_normalise(values)
    assert np.all(np.isfinite(result))
    assert float(np.sum(result)) < 1e-5


def test_relu_basic() -> None:
    x = np.array([-1.0, 0.0, 1.0, -0.5, 0.5], dtype=np.float32)
    result = _relu(x)
    expected = np.array([0.0, 0.0, 1.0, 0.0, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_belief_encoder_from_weights(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    npz_path = tmp_path / "belief.npz"
    np.savez(
        npz_path,
        w1=rng.standard_normal((256, 128)).astype(np.float32) * 0.01,
        b1=np.zeros(128, dtype=np.float32),
        w2=rng.standard_normal((128, 128)).astype(np.float32) * 0.01,
        b2=np.zeros(128, dtype=np.float32),
    )
    enc = BeliefEncoder(weights_path=npz_path)
    out = enc.forward(np.zeros(256, dtype=np.float32))
    assert out.shape == (_BELIEF_DIM,)


def test_desire_encoder_from_weights(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    npz_path = tmp_path / "desire.npz"
    np.savez(
        npz_path,
        w1=rng.standard_normal((128, 64)).astype(np.float32) * 0.01,
        b1=np.zeros(64, dtype=np.float32),
    )
    enc = DesireEncoder(weights_path=npz_path)
    out = enc.forward(np.zeros(128, dtype=np.float32))
    assert out.shape == (_DESIRE_DIM,)


def test_intention_predictor_from_weights(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    npz_path = tmp_path / "intention.npz"
    np.savez(
        npz_path,
        w1=rng.standard_normal((64, _INTENTION_CLASSES)).astype(np.float32) * 0.01,
        b1=np.zeros(_INTENTION_CLASSES, dtype=np.float32),
    )
    pred = IntentionPredictor(weights_path=npz_path)
    out = pred.forward(np.zeros(64, dtype=np.float32))
    assert out.shape == (_INTENTION_CLASSES,)
    assert abs(float(np.sum(out)) - 1.0) < 1e-5


def test_affect_estimator_from_weights(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    input_dim = _DESIRE_DIM + _INTENTION_CLASSES
    npz_path = tmp_path / "affect.npz"
    np.savez(
        npz_path,
        w1=rng.standard_normal((input_dim, 2)).astype(np.float32) * 0.01,
        b1=np.zeros(2, dtype=np.float32),
    )
    est = AffectEstimator(weights_path=npz_path)
    out = est.forward(
        np.zeros(_DESIRE_DIM, dtype=np.float32),
        np.zeros(_INTENTION_CLASSES, dtype=np.float32),
    )
    assert out.shape == (_AFFECT_DIM,)


def test_neural_bdi_with_weights_dir(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    np.savez(
        tmp_path / "belief.npz",
        w1=rng.standard_normal((256, 128)).astype(np.float32) * 0.01,
        b1=np.zeros(128, dtype=np.float32),
        w2=rng.standard_normal((128, 128)).astype(np.float32) * 0.01,
        b2=np.zeros(128, dtype=np.float32),
    )
    np.savez(
        tmp_path / "desire.npz",
        w1=rng.standard_normal((128, 64)).astype(np.float32) * 0.01,
        b1=np.zeros(64, dtype=np.float32),
    )
    np.savez(
        tmp_path / "intention.npz",
        w1=rng.standard_normal((64, _INTENTION_CLASSES)).astype(np.float32) * 0.01,
        b1=np.zeros(_INTENTION_CLASSES, dtype=np.float32),
    )
    input_dim = _DESIRE_DIM + _INTENTION_CLASSES
    np.savez(
        tmp_path / "affect.npz",
        w1=rng.standard_normal((input_dim, 2)).astype(np.float32) * 0.01,
        b1=np.zeros(2, dtype=np.float32),
    )

    bdi = NeuralBDI(weights_dir=tmp_path)
    result = bdi.infer(np.zeros(256, dtype=np.float32))
    assert "belief" in result
    assert result["approach_rate"] > 0


def test_neural_bdi_with_partial_weights_dir(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    np.savez(
        tmp_path / "belief.npz",
        w1=rng.standard_normal((256, 128)).astype(np.float32) * 0.01,
        b1=np.zeros(128, dtype=np.float32),
        w2=rng.standard_normal((128, 128)).astype(np.float32) * 0.01,
        b2=np.zeros(128, dtype=np.float32),
    )

    bdi = NeuralBDI(weights_dir=tmp_path)
    result = bdi.infer(np.zeros(256, dtype=np.float32))
    assert "belief" in result
