"""Phase 2.3b — Train BDI sub-networks on annotated navigation data.

Belief autoencoder uses Adam (F-041). Intention's 10-class claim is only
valid when ``intention_features`` in the npz match ``label_intention``
inputs — vision-only X cannot carry that signal. Do not publish weights
until :func:`passes_bdi_publish_bars` is true on the operator's dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray

from mousedroid.common.math.numpy_ops import relu as _relu
from mousedroid.common.math.numpy_ops import softmax as _softmax
from mousedroid.constants import (
    DEFAULT_AFFECT_DIM,
    DEFAULT_BELIEF_DIM,
    DEFAULT_DESIRE_DIM,
    DEFAULT_GRADIENT_SCALE,
    DEFAULT_INTENTION_CLASSES,
)
from training.training_utils import AdamOptimizer, iter_batches, log_epoch_loss

_log = structlog.get_logger(__name__)

_BELIEF_DIM = DEFAULT_BELIEF_DIM
_DESIRE_DIM = DEFAULT_DESIRE_DIM
_INTENTION_CLASSES = DEFAULT_INTENTION_CLASSES
_AFFECT_DIM = DEFAULT_AFFECT_DIM


def _init_weights(rng: np.random.Generator, fan_in: int, fan_out: int) -> NDArray[Any]:
    """He-scaled initialisation for a ReLU layer.

    Replaces ``WEIGHT_INIT_SCALE``, one fixed scalar (0.01) applied regardless
    of fan-in, which shrank the signal about 100x per layer through the belief
    autoencoder's 256 -> 128 -> 128 -> 256 stack until the reconstruction was
    numerically zero. Scaling by ``sqrt(2 / fan_in)`` holds activation variance
    roughly constant across depth instead.

    Kept local to training on purpose: ``WEIGHT_INIT_SCALE`` is also used at
    runtime by ``cognitive/bdi_model.py`` and ``cognitive/constitutional_rl.py``,
    and a training fix should not change a shared production constant as a side
    effect. Initialisation only matters while training -- loading trained
    weights replaces it outright -- so the two need not agree.

    Args:
        rng: Seeded generator, so runs stay reproducible.
        fan_in: Input dimension of the layer.
        fan_out: Output dimension of the layer.

    Returns:
        A ``(fan_in, fan_out)`` float32 weight matrix.
    """
    scale = np.sqrt(2.0 / fan_in)
    weights: NDArray[Any] = (rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float32)
    return weights


def _save_weight_file(path: Path, weights: dict[str, NDArray[Any]]) -> None:
    """Persist numpy weights without mypy ignore shims."""
    arrays: dict[str, Any] = {name: np.asarray(value) for name, value in weights.items()}
    np.savez(path, **arrays)


def _cross_entropy(logits: NDArray[Any], labels: NDArray[Any]) -> float:
    """Batch cross-entropy loss."""
    probs = _softmax(logits)
    n = len(labels)
    log_probs = np.log(probs[np.arange(n), labels] + 1e-8)
    return -float(np.mean(log_probs))


def pca_reconstruction_mse(observations: NDArray[Any], *, rank: int = 128) -> float:
    """PCA-``rank`` reconstruction MSE (mean-centered SVD).

    Args:
        observations: Array ``(n, dim)``.
        rank: Number of principal components to keep.

    Returns:
        Mean squared reconstruction error.
    """
    mean = observations.mean(axis=0, keepdims=True)
    centered = observations - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    keep = min(rank, int(vt.shape[0]))
    recon = (centered @ vt[:keep].T) @ vt[:keep] + mean
    return float(np.mean((recon - observations) ** 2))


def reconstruct_from_belief_encoder(
    observations: NDArray[Any],
    weights: dict[str, NDArray[Any]],
) -> NDArray[Any]:
    """Reconstruct observations with the trained encoder+decoder weights."""
    hidden = _relu(observations @ weights["w1"] + weights["b1"])
    latent = _relu(hidden @ weights["w2"] + weights["b2"])
    recon: NDArray[Any] = latent @ weights["w_dec"] + weights["b_dec"]
    return recon


def belief_reconstruction_mse(
    observations: NDArray[Any],
    weights: dict[str, NDArray[Any]],
) -> float:
    """MSE between observations and the belief autoencoder reconstruction."""
    recon = reconstruct_from_belief_encoder(observations, weights)
    return float(np.mean((recon - observations) ** 2))


def majority_class_accuracy(labels: NDArray[Any]) -> float:
    """Accuracy of always predicting the most frequent label."""
    if labels.size == 0:
        return 0.0
    _values, counts = np.unique(labels, return_counts=True)
    return float(counts.max()) / float(labels.size)


def linear_classifier_accuracy(
    features: NDArray[Any],
    labels: NDArray[Any],
    weights: dict[str, NDArray[Any]],
) -> float:
    """Argmax accuracy of the causal intention head (linear or one-hidden MLP)."""
    inputs = features.astype(np.float32, copy=False)
    if "x_mean" in weights and "x_std" in weights:
        inputs = (inputs - weights["x_mean"]) / weights["x_std"]
    if "w2" in weights:
        hidden = _relu(inputs @ weights["w1"] + weights["b1"])
        logits = hidden @ weights["w2"] + weights["b2"]
    else:
        logits = inputs @ weights["w1"] + weights["b1"]
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == labels))


def passes_bdi_publish_bars(
    *,
    belief_mse: float,
    pca_mse: float,
    intention_acc: float,
    majority_acc: float,
) -> bool:
    """True only when belief AE beats PCA-128 and intention beats majority.

    HuggingFace publish stays blocked until this returns True on the
    operator dataset. The OTA poller reads ``ianshank/mousedroid-policy-v2``,
    not ``mousedroid-weights``.
    """
    return belief_mse < pca_mse and intention_acc > majority_acc


def _adam_epochs(
    *,
    n: int,
    batch_size: int,
    epochs: int,
    rng: np.random.Generator,
    weights: dict[str, NDArray[Any]],
    lr: float,
    event: str,
    batch_fn: Callable[[NDArray[np.intp]], tuple[float, dict[str, NDArray[Any]]]],
) -> None:
    """Run Adam mini-batch epochs with the shared epoch logger."""
    optimizer = AdamOptimizer(weights, lr=lr)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for idx in iter_batches(n, batch_size, rng):
            loss, grads = batch_fn(idx)
            optimizer.step(grads)
            total_loss += loss
            n_batches += 1
        log_epoch_loss(_log, event, epoch, total_loss, n_batches)


def _belief_ae_loss_and_grads(
    x: NDArray[Any],
    weights: dict[str, NDArray[Any]],
    gradient_scale: float,
) -> tuple[float, dict[str, NDArray[Any]]]:
    """Forward+backward for one belief autoencoder batch."""
    hidden = _relu(x @ weights["w1"] + weights["b1"])
    latent = _relu(hidden @ weights["w2"] + weights["b2"])
    recon = latent @ weights["w_dec"] + weights["b_dec"]
    loss = float(np.mean((recon - x) ** 2))
    batch = float(x.shape[0])
    d_recon = (gradient_scale / batch) * (recon - x)
    d_latent = (d_recon @ weights["w_dec"].T) * (latent > 0).astype(np.float32)
    d_hidden = (d_latent @ weights["w2"].T) * (hidden > 0).astype(np.float32)
    grads = {
        "w_dec": latent.T @ d_recon,
        "b_dec": d_recon.sum(axis=0),
        "w2": hidden.T @ d_latent,
        "b2": d_latent.sum(axis=0),
        "w1": x.T @ d_hidden,
        "b1": d_hidden.sum(axis=0),
    }
    return loss, grads


def train_belief_encoder(
    observations: NDArray[Any],
    lr: float = 3e-4,
    epochs: int = 100,
    batch_size: int = 32,
    gradient_scale: float = DEFAULT_GRADIENT_SCALE,
) -> dict[str, NDArray[Any]]:
    """Train BeliefEncoder as an autoencoder (256 → 128 → 256) with Adam.

    Returns:
        Weight dict with encoder keys ``w1``, ``b1``, ``w2``, ``b2`` plus
        training-only decoder ``w_dec``, ``b_dec`` for reconstruction eval.
        :class:`~mousedroid.cognitive.bdi_model.NeuralBDI` ignores extra keys.
    """
    rng = np.random.default_rng(42)
    obs_dim = int(observations.shape[1])
    weights: dict[str, NDArray[Any]] = {
        "w1": _init_weights(rng, obs_dim, _BELIEF_DIM),
        "b1": np.zeros(_BELIEF_DIM, dtype=np.float32),
        "w2": _init_weights(rng, _BELIEF_DIM, _BELIEF_DIM),
        "b2": np.zeros(_BELIEF_DIM, dtype=np.float32),
        "w_dec": _init_weights(rng, _BELIEF_DIM, obs_dim),
        "b_dec": np.zeros(obs_dim, dtype=np.float32),
    }

    def _batch(idx: NDArray[np.intp]) -> tuple[float, dict[str, NDArray[Any]]]:
        return _belief_ae_loss_and_grads(observations[idx], weights, gradient_scale)

    _adam_epochs(
        n=len(observations),
        batch_size=batch_size,
        epochs=epochs,
        rng=rng,
        weights=weights,
        lr=lr,
        event="belief_epoch",
        batch_fn=_batch,
    )
    return weights


def _desire_ae_loss_and_grads(
    x: NDArray[Any],
    weights: dict[str, NDArray[Any]],
    gradient_scale: float,
) -> tuple[float, dict[str, NDArray[Any]]]:
    """Forward+backward for one desire autoencoder batch."""
    desire = _relu(x @ weights["w1"] + weights["b1"])
    recon = desire @ weights["w_dec"] + weights["b_dec"]
    loss = float(np.mean((recon - x) ** 2))
    batch = float(x.shape[0])
    d_recon = (gradient_scale / batch) * (recon - x)
    d_desire = (d_recon @ weights["w_dec"].T) * (desire > 0).astype(np.float32)
    grads = {
        "w_dec": desire.T @ d_recon,
        "b_dec": d_recon.sum(axis=0),
        "w1": x.T @ d_desire,
        "b1": d_desire.sum(axis=0),
    }
    return loss, grads


def train_desire_encoder(
    observations: NDArray[Any],
    belief_weights: dict[str, NDArray[Any]],
    lr: float = 3e-4,
    epochs: int = 100,
    batch_size: int = 32,
    gradient_scale: float = DEFAULT_GRADIENT_SCALE,
) -> dict[str, NDArray[Any]]:
    """Train DesireEncoder to map belief → desire (reward-relevant features).

    Returns:
        Weight dict with keys ``w1``, ``b1``.
    """
    rng = np.random.default_rng(43)
    hidden = _relu(observations @ belief_weights["w1"] + belief_weights["b1"])
    beliefs = _relu(hidden @ belief_weights["w2"] + belief_weights["b2"])
    weights: dict[str, NDArray[Any]] = {
        "w1": _init_weights(rng, _BELIEF_DIM, _DESIRE_DIM),
        "b1": np.zeros(_DESIRE_DIM, dtype=np.float32),
        "w_dec": _init_weights(rng, _DESIRE_DIM, _BELIEF_DIM),
        "b_dec": np.zeros(_BELIEF_DIM, dtype=np.float32),
    }

    def _batch(idx: NDArray[np.intp]) -> tuple[float, dict[str, NDArray[Any]]]:
        return _desire_ae_loss_and_grads(beliefs[idx], weights, gradient_scale)

    _adam_epochs(
        n=len(observations),
        batch_size=batch_size,
        epochs=epochs,
        rng=rng,
        weights=weights,
        lr=lr,
        event="desire_epoch",
        batch_fn=_batch,
    )
    return {"w1": weights["w1"], "b1": weights["b1"]}


def _mlp_ce_loss_and_grads(
    x: NDArray[Any],
    y: NDArray[Any],
    weights: dict[str, NDArray[Any]],
) -> tuple[float, dict[str, NDArray[Any]]]:
    """Softmax CE gradients for a one-hidden ReLU MLP."""
    hidden = _relu(x @ weights["w1"] + weights["b1"])
    logits = hidden @ weights["w2"] + weights["b2"]
    loss = _cross_entropy(logits, y)
    probs = _softmax(logits)
    d_logits = probs.copy()
    d_logits[np.arange(len(y)), y] -= 1.0
    d_logits /= float(len(y))
    d_hidden = (d_logits @ weights["w2"].T) * (hidden > 0).astype(np.float32)
    grads = {
        "w2": hidden.T @ d_logits,
        "b2": d_logits.sum(axis=0),
        "w1": x.T @ d_hidden,
        "b1": d_hidden.sum(axis=0),
    }
    return loss, grads


def _linear_ce_loss_and_grads(
    x: NDArray[Any],
    y: NDArray[Any],
    weights: dict[str, NDArray[Any]],
) -> tuple[float, dict[str, NDArray[Any]]]:
    """Softmax cross-entropy gradients for a linear head."""
    logits = x @ weights["w1"] + weights["b1"]
    loss = _cross_entropy(logits, y)
    probs = _softmax(logits)
    d_logits = probs.copy()
    d_logits[np.arange(len(y)), y] -= 1.0
    d_logits /= float(len(y))
    grads = {"w1": x.T @ d_logits, "b1": d_logits.sum(axis=0)}
    return loss, grads


def _encode_desires(
    observations: NDArray[Any],
    belief_weights: dict[str, NDArray[Any]],
    desire_weights: dict[str, NDArray[Any]],
) -> NDArray[Any]:
    """Map observations through the frozen belief→desire stack."""
    hidden = _relu(observations @ belief_weights["w1"] + belief_weights["b1"])
    beliefs = _relu(hidden @ belief_weights["w2"] + belief_weights["b2"])
    desires: NDArray[Any] = _relu(beliefs @ desire_weights["w1"] + desire_weights["b1"])
    return desires


def train_intention_predictor(
    observations: NDArray[Any],
    intentions: NDArray[Any],
    belief_weights: dict[str, NDArray[Any]] | None = None,
    desire_weights: dict[str, NDArray[Any]] | None = None,
    lr: float = 3e-4,
    epochs: int = 100,
    batch_size: int = 32,
    *,
    features: NDArray[Any] | None = None,
) -> dict[str, NDArray[Any]]:
    """Train IntentionPredictor with cross-entropy on labelled intentions.

    When *features* is set, train on that matrix (F-041 causal X). Otherwise
    route *observations* through the frozen belief→desire stack so
    ``intention.npz`` still loads in :class:`NeuralBDI` (64 → 10). That
    runtime path is not the 10-class claim — labels are not a function of
    vision.

    Returns:
        Weight dict with keys ``w1``, ``b1``.
    """
    rng = np.random.default_rng(44)
    x_mean: NDArray[Any] | None = None
    x_std: NDArray[Any] | None = None
    if features is not None:
        raw = features.astype(np.float32, copy=False)
        x_mean = raw.mean(axis=0).astype(np.float32)
        x_std = raw.std(axis=0)
        x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
        inputs = (raw - x_mean) / x_std
        hidden_dim = 32
        weights = {
            "w1": _init_weights(rng, int(inputs.shape[1]), hidden_dim),
            "b1": np.zeros(hidden_dim, dtype=np.float32),
            "w2": _init_weights(rng, hidden_dim, _INTENTION_CLASSES),
            "b2": np.zeros(_INTENTION_CLASSES, dtype=np.float32),
        }
        batch_loss = _mlp_ce_loss_and_grads
    else:
        if belief_weights is None or desire_weights is None:
            msg = "belief_weights and desire_weights are required when features is None"
            raise ValueError(msg)
        inputs = _encode_desires(observations, belief_weights, desire_weights)
        weights = {
            "w1": _init_weights(rng, _DESIRE_DIM, _INTENTION_CLASSES),
            "b1": np.zeros(_INTENTION_CLASSES, dtype=np.float32),
        }
        batch_loss = _linear_ce_loss_and_grads

    labels = intentions.astype(np.int64, copy=False)

    def _batch(idx: NDArray[np.intp]) -> tuple[float, dict[str, NDArray[Any]]]:
        return batch_loss(inputs[idx], labels[idx], weights)

    _adam_epochs(
        n=len(labels),
        batch_size=batch_size,
        epochs=epochs,
        rng=rng,
        weights=weights,
        lr=lr,
        event="intention_epoch",
        batch_fn=_batch,
    )
    result: dict[str, NDArray[Any]] = {
        "w1": weights["w1"],
        "b1": weights["b1"],
    }
    if "w2" in weights:
        result["w2"] = weights["w2"]
        result["b2"] = weights["b2"]
    if x_mean is not None and x_std is not None:
        result["x_mean"] = x_mean
        result["x_std"] = x_std
    return result


def train_affect_estimator(
    observations: NDArray[Any],
    belief_weights: dict[str, NDArray[Any]],
    desire_weights: dict[str, NDArray[Any]],
    intention_weights: dict[str, NDArray[Any]],
    lr: float = 3e-4,
    epochs: int = 100,
    batch_size: int = 32,
    gradient_scale: float = DEFAULT_GRADIENT_SCALE,
) -> dict[str, NDArray[Any]]:
    """Train AffectEstimator on synthetic valence/arousal targets.

    Target valence derived from desire norm; arousal from intention entropy.

    Returns:
        Weight dict with keys ``w1``, ``b1``.
    """
    rng = np.random.default_rng(45)
    desires = _encode_desires(observations, belief_weights, desire_weights)
    if intention_weights["w1"].shape[0] != _DESIRE_DIM:
        msg = "affect estimator requires the runtime 64→10 intention head, not causal features"
        raise ValueError(msg)
    intention_logits = desires @ intention_weights["w1"] + intention_weights["b1"]
    intention_probs = _softmax(intention_logits)

    desire_norms = np.linalg.norm(desires, axis=1)
    valence = np.tanh(desire_norms / (desire_norms.mean() + 1e-8) - 1.0)
    entropy = -np.sum(intention_probs * np.log(intention_probs + 1e-8), axis=1)
    max_entropy = np.log(_INTENTION_CLASSES)
    arousal = 2.0 * (entropy / max_entropy) - 1.0
    targets = np.stack([valence, arousal], axis=1).astype(np.float32)

    inputs = np.concatenate([desires, intention_probs], axis=1)
    input_dim = _DESIRE_DIM + _INTENTION_CLASSES
    weights: dict[str, NDArray[Any]] = {
        "w1": _init_weights(rng, input_dim, _AFFECT_DIM),
        "b1": np.zeros(_AFFECT_DIM, dtype=np.float32),
    }

    def _batch(idx: NDArray[np.intp]) -> tuple[float, dict[str, NDArray[Any]]]:
        batch_x = inputs[idx]
        batch_y = targets[idx]
        raw = batch_x @ weights["w1"] + weights["b1"]
        pred = np.tanh(raw)
        loss = float(np.mean((pred - batch_y) ** 2))
        batch = float(batch_y.shape[0])
        d_pred = (gradient_scale / batch) * (pred - batch_y)
        d_raw = d_pred * (1.0 - pred**2)
        grads = {"w1": batch_x.T @ d_raw, "b1": d_raw.sum(axis=0)}
        return loss, grads

    _adam_epochs(
        n=len(observations),
        batch_size=batch_size,
        epochs=epochs,
        rng=rng,
        weights=weights,
        lr=lr,
        event="affect_epoch",
        batch_fn=_batch,
    )
    return {"w1": weights["w1"], "b1": weights["b1"]}


def train_bdi(
    annotations_path: Path | str,
    output_dir: Path | str | None = None,
    lr: float = 3e-4,
    epochs: int = 100,
    batch_size: int = 32,
    gradient_scale: float = DEFAULT_GRADIENT_SCALE,
) -> Path:
    """Full Phase 2.3 BDI training pipeline.

    Args:
        annotations_path: Path to ``bdi_annotations.npz``.
        output_dir: Directory to save ``.npz`` weight files.
        lr: Learning rate for all sub-networks.
        epochs: Training epochs per sub-network.
        batch_size: Batch size.
        gradient_scale: Multiplier for manual numpy MSE gradients.

    Returns:
        Path to output directory with saved weights.
    """
    output_dir = Path(output_dir) if output_dir else Path("weights/bdi")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(annotations_path)
    observations = data["observations"].astype(np.float32)
    intentions = data["intentions"].astype(np.int64)
    causal_features = None
    if "intention_features" in data.files:
        causal_features = data["intention_features"].astype(np.float32)

    _log.info("bdi_training_start", n_samples=len(observations))

    belief_weights = train_belief_encoder(observations, lr, epochs, batch_size, gradient_scale)
    _save_weight_file(output_dir / "belief.npz", belief_weights)
    _log.info("belief_encoder_saved")

    desire_weights = train_desire_encoder(
        observations,
        belief_weights,
        lr,
        epochs,
        batch_size,
        gradient_scale,
    )
    _save_weight_file(output_dir / "desire.npz", desire_weights)
    _log.info("desire_encoder_saved")

    intention_weights = train_intention_predictor(
        observations,
        intentions,
        belief_weights,
        desire_weights,
        lr,
        epochs,
        batch_size,
    )
    _save_weight_file(output_dir / "intention.npz", intention_weights)
    _log.info("intention_predictor_saved")

    if causal_features is not None:
        causal_weights = train_intention_predictor(
            observations,
            intentions,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            features=causal_features,
        )
        _save_weight_file(output_dir / "intention_causal.npz", causal_weights)
        belief_mse = belief_reconstruction_mse(observations, belief_weights)
        pca_mse = pca_reconstruction_mse(observations)
        intent_acc = linear_classifier_accuracy(causal_features, intentions, causal_weights)
        majority_acc = majority_class_accuracy(intentions)
        publishable = passes_bdi_publish_bars(
            belief_mse=belief_mse,
            pca_mse=pca_mse,
            intention_acc=intent_acc,
            majority_acc=majority_acc,
        )
        _log.info(
            "bdi_publish_bars",
            belief_mse=round(belief_mse, 6),
            pca_mse=round(pca_mse, 6),
            intention_acc=round(intent_acc, 6),
            majority_acc=round(majority_acc, 6),
            publishable=publishable,
            ota_repo="ianshank/mousedroid-policy-v2",
        )
        if not publishable:
            _log.warning("bdi_hf_publish_blocked_until_bars_beaten")

    affect_weights = train_affect_estimator(
        observations,
        belief_weights,
        desire_weights,
        intention_weights,
        lr,
        epochs,
        batch_size,
        gradient_scale,
    )
    _save_weight_file(output_dir / "affect.npz", affect_weights)
    _log.info("affect_estimator_saved")

    _log.info("bdi_training_complete", output_dir=str(output_dir))
    return output_dir
