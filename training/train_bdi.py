"""Phase 2.3b — Train BDI sub-networks on annotated navigation data.

Trains BeliefEncoder, DesireEncoder, IntentionPredictor, and AffectEstimator
using numpy-only SGD, matching the existing numpy MLP design in bdi_model.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

from mousedroid.config.schema import BDITrainingConfig
from mousedroid.utils.numpy_ops import relu as _relu
from mousedroid.utils.numpy_ops import softmax as _softmax
from training.collect_annotations import INTENTION_LABELS

_log = structlog.get_logger(__name__)

# Dimensions matching bdi_model.py / ModelConfig defaults
_OBS_DIM = 256
_BELIEF_DIM = 128
_DESIRE_DIM = 64
_INTENTION_CLASSES = len(INTENTION_LABELS)
_AFFECT_DIM = 2


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Batch cross-entropy loss."""
    probs = _softmax(logits)
    n = len(labels)
    log_probs = np.log(probs[np.arange(n), labels] + 1e-8)
    return -float(np.mean(log_probs))


def train_belief_encoder(
    observations: np.ndarray,
    lr: float = 3e-4,
    epochs: int = 100,
    batch_size: int = 32,
    intentions: np.ndarray | None = None,
    probe_weight: float = 0.1,
) -> dict[str, np.ndarray]:
    """Train BeliefEncoder as an autoencoder (256 → 128 → 256).

    When *intentions* are provided, a classification probe (128 → n_classes)
    is co-trained so the belief representation captures discriminative
    features.  The probe weights are discarded; only encoder weights are
    returned.

    Returns:
        Weight dict with keys ``w1``, ``b1``, ``w2``, ``b2`` for the encoder
        (first two layers).
    """
    rng = np.random.default_rng(42)
    n = len(observations)

    # Encoder weights (256 → 128)
    w1 = rng.standard_normal((_OBS_DIM, _BELIEF_DIM)).astype(np.float32) * 0.01
    b1 = np.zeros(_BELIEF_DIM, dtype=np.float32)
    # Second encoder layer (128 → 128)
    w2 = rng.standard_normal((_BELIEF_DIM, _BELIEF_DIM)).astype(np.float32) * 0.01
    b2 = np.zeros(_BELIEF_DIM, dtype=np.float32)
    # Decoder weights (128 → 256)
    w_dec = rng.standard_normal((_BELIEF_DIM, _OBS_DIM)).astype(np.float32) * 0.01
    b_dec = np.zeros(_OBS_DIM, dtype=np.float32)

    # Optional classification probe (128 → n_classes)
    use_probe = intentions is not None and probe_weight > 0
    if use_probe:
        w_probe = rng.standard_normal((_BELIEF_DIM, _INTENTION_CLASSES)).astype(np.float32) * 0.01
        b_probe = np.zeros(_INTENTION_CLASSES, dtype=np.float32)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
            x = observations[idx]

            # Forward — autoencoder path
            h1 = _relu(x @ w1 + b1)
            h2 = _relu(h1 @ w2 + b2)
            recon = h2 @ w_dec + b_dec
            recon_loss = np.mean((recon - x) ** 2)

            # Forward — classification probe path
            if use_probe:
                probe_logits = h2 @ w_probe + b_probe
                cls_loss = _cross_entropy(probe_logits, intentions[idx])
                loss = recon_loss + probe_weight * cls_loss
            else:
                loss = recon_loss

            total_loss += loss

            # Backward — autoencoder gradients
            d_recon = 2.0 * (recon - x) / x.shape[0]
            d_w_dec = h2.T @ d_recon / batch_size
            d_b_dec = d_recon.mean(axis=0)

            d_h2 = d_recon @ w_dec.T

            # Backward — probe gradients (add to d_h2)
            if use_probe:
                probs = _softmax(probe_logits)
                d_logits = probs.copy()
                d_logits[np.arange(len(idx)), intentions[idx]] -= 1.0
                d_logits /= len(idx)
                d_w_probe = h2.T @ d_logits / batch_size
                d_b_probe = d_logits.mean(axis=0)
                d_h2 = d_h2 + probe_weight * (d_logits @ w_probe.T)

            d_h2 = d_h2 * (h2 > 0).astype(np.float32)
            d_w2 = h1.T @ d_h2 / batch_size
            d_b2 = d_h2.mean(axis=0)

            d_h1 = d_h2 @ w2.T
            d_h1 = d_h1 * (h1 > 0).astype(np.float32)
            d_w1 = x.T @ d_h1 / batch_size
            d_b1 = d_h1.mean(axis=0)

            # SGD update — encoder + decoder
            w1 -= lr * d_w1
            b1 -= lr * d_b1
            w2 -= lr * d_w2
            b2 -= lr * d_b2
            w_dec -= lr * d_w_dec
            b_dec -= lr * d_b_dec
            if use_probe:
                w_probe -= lr * d_w_probe
                b_probe -= lr * d_b_probe
            n_batches += 1

        if epoch % 20 == 0:
            _log.info("belief_epoch", epoch=epoch, loss=round(total_loss / max(n_batches, 1), 6))

    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def train_desire_encoder(
    observations: np.ndarray,
    belief_weights: dict[str, np.ndarray],
    lr: float = 3e-4,
    epochs: int = 100,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Train DesireEncoder to map belief → desire (reward-relevant features).

    Returns:
        Weight dict with keys ``w1``, ``b1``.
    """
    rng = np.random.default_rng(43)
    n = len(observations)

    # Compute belief embeddings using trained encoder
    h1 = _relu(observations @ belief_weights["w1"] + belief_weights["b1"])
    beliefs = _relu(h1 @ belief_weights["w2"] + belief_weights["b2"])

    # Desire encoder weights (128 → 64)
    w1 = rng.standard_normal((_BELIEF_DIM, _DESIRE_DIM)).astype(np.float32) * 0.01
    b1 = np.zeros(_DESIRE_DIM, dtype=np.float32)
    # Decoder for training (64 → 128)
    w_dec = rng.standard_normal((_DESIRE_DIM, _BELIEF_DIM)).astype(np.float32) * 0.01
    b_dec = np.zeros(_BELIEF_DIM, dtype=np.float32)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
            x = beliefs[idx]

            # Forward
            desire = _relu(x @ w1 + b1)
            recon = desire @ w_dec + b_dec
            loss = np.mean((recon - x) ** 2)
            total_loss += loss

            # Backward
            d_recon = 2.0 * (recon - x) / x.shape[0]
            d_w_dec = desire.T @ d_recon / batch_size
            d_b_dec = d_recon.mean(axis=0)

            d_desire = d_recon @ w_dec.T
            d_desire = d_desire * (desire > 0).astype(np.float32)
            d_w1 = x.T @ d_desire / batch_size
            d_b1 = d_desire.mean(axis=0)

            w1 -= lr * d_w1
            b1 -= lr * d_b1
            w_dec -= lr * d_w_dec
            b_dec -= lr * d_b_dec
            n_batches += 1

        if epoch % 20 == 0:
            _log.info("desire_epoch", epoch=epoch, loss=round(total_loss / max(n_batches, 1), 6))

    return {"w1": w1, "b1": b1}


def train_intention_predictor(
    observations: np.ndarray,
    intentions: np.ndarray,
    belief_weights: dict[str, np.ndarray],
    desire_weights: dict[str, np.ndarray],
    lr: float = 3e-4,
    epochs: int = 100,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Train IntentionPredictor with cross-entropy on labelled intentions.

    Returns:
        Weight dict with keys ``w1``, ``b1``.
    """
    rng = np.random.default_rng(44)
    n = len(observations)

    # Compute desire embeddings
    h1 = _relu(observations @ belief_weights["w1"] + belief_weights["b1"])
    beliefs = _relu(h1 @ belief_weights["w2"] + belief_weights["b2"])
    desires = _relu(beliefs @ desire_weights["w1"] + desire_weights["b1"])

    # Intention weights (64 → 8)
    w1 = rng.standard_normal((_DESIRE_DIM, _INTENTION_CLASSES)).astype(np.float32) * 0.01
    b1 = np.zeros(_INTENTION_CLASSES, dtype=np.float32)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
            x = desires[idx]
            y = intentions[idx]

            # Forward
            logits = x @ w1 + b1
            loss = _cross_entropy(logits, y)
            total_loss += loss

            # Backward (softmax + cross-entropy gradient)
            probs = _softmax(logits)
            d_logits = probs.copy()
            d_logits[np.arange(len(y)), y] -= 1.0
            d_logits /= len(y)

            d_w1 = x.T @ d_logits / batch_size
            d_b1 = d_logits.mean(axis=0)

            w1 -= lr * d_w1
            b1 -= lr * d_b1
            n_batches += 1

        if epoch % 20 == 0:
            _log.info("intention_epoch", epoch=epoch, loss=round(total_loss / max(n_batches, 1), 6))

    return {"w1": w1, "b1": b1}


def train_affect_estimator(
    observations: np.ndarray,
    belief_weights: dict[str, np.ndarray],
    desire_weights: dict[str, np.ndarray],
    intention_weights: dict[str, np.ndarray],
    lr: float = 3e-4,
    epochs: int = 100,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Train AffectEstimator on synthetic valence/arousal targets.

    Target valence derived from desire norm; arousal from intention entropy.

    Returns:
        Weight dict with keys ``w1``, ``b1``.
    """
    rng = np.random.default_rng(45)
    n = len(observations)

    # Compute pipeline outputs
    h1 = _relu(observations @ belief_weights["w1"] + belief_weights["b1"])
    beliefs = _relu(h1 @ belief_weights["w2"] + belief_weights["b2"])
    desires = _relu(beliefs @ desire_weights["w1"] + desire_weights["b1"])
    intention_logits = desires @ intention_weights["w1"] + intention_weights["b1"]
    intention_probs = _softmax(intention_logits)

    # Synthetic affect targets
    # Valence: normalised desire magnitude → [-1, 1]
    desire_norms = np.linalg.norm(desires, axis=1)
    valence = np.tanh(desire_norms / (desire_norms.mean() + 1e-8) - 1.0)
    # Arousal: intention entropy (high entropy → high arousal)
    entropy = -np.sum(intention_probs * np.log(intention_probs + 1e-8), axis=1)
    max_entropy = np.log(_INTENTION_CLASSES)
    arousal = 2.0 * (entropy / max_entropy) - 1.0
    targets = np.stack([valence, arousal], axis=1).astype(np.float32)

    # Combined input: desire (64) + intentions (8) = 72
    inputs = np.concatenate([desires, intention_probs], axis=1)
    input_dim = _DESIRE_DIM + _INTENTION_CLASSES

    # Affect weights (72 → 2)
    w1 = rng.standard_normal((input_dim, _AFFECT_DIM)).astype(np.float32) * 0.01
    b1 = np.zeros(_AFFECT_DIM, dtype=np.float32)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
            x = inputs[idx]
            y = targets[idx]

            # Forward (tanh output)
            raw = x @ w1 + b1
            pred = np.tanh(raw)
            loss = np.mean((pred - y) ** 2)
            total_loss += loss

            # Backward through tanh
            d_pred = 2.0 * (pred - y) / y.shape[0]
            d_raw = d_pred * (1.0 - pred**2)

            d_w1 = x.T @ d_raw / batch_size
            d_b1 = d_raw.mean(axis=0)

            w1 -= lr * d_w1
            b1 -= lr * d_b1
            n_batches += 1

        if epoch % 20 == 0:
            _log.info("affect_epoch", epoch=epoch, loss=round(total_loss / max(n_batches, 1), 6))

    return {"w1": w1, "b1": b1}


def train_bdi(
    annotations_path: Path | str,
    output_dir: Path | str | None = None,
    lr: float | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    bdi_config: BDITrainingConfig | None = None,
) -> Path:
    """Full Phase 2.3 BDI training pipeline.

    Args:
        annotations_path: Path to ``bdi_annotations.npz``.
        output_dir: Directory to save ``.npz`` weight files.
        lr: Learning rate (overrides ``bdi_config.learning_rate`` if set).
        epochs: Training epochs (overrides ``bdi_config.epochs`` if set).
        batch_size: Batch size (overrides ``bdi_config.batch_size`` if set).
        bdi_config: Optional BDI-specific training config. Falls back to
            ``BDITrainingConfig()`` defaults when ``None``.

    Returns:
        Path to output directory with saved weights.
    """
    cfg = bdi_config or BDITrainingConfig()
    _lr = lr if lr is not None else cfg.learning_rate
    _epochs = epochs if epochs is not None else cfg.epochs
    _batch_size = batch_size if batch_size is not None else cfg.batch_size

    output_dir = Path(output_dir) if output_dir else Path("weights/bdi")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(annotations_path)
    observations = data["observations"].astype(np.float32)
    intentions = data["intentions"].astype(np.int64)

    # Z-score normalise observations for better SGD convergence
    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0) + 1e-8
    observations = (observations - obs_mean) / obs_std

    # Persist norm stats so validation & runtime can reconstruct the transform
    np.savez(output_dir / "belief_norm_stats.npz", mean=obs_mean, std=obs_std)
    _log.info("norm_stats_saved", path=str(output_dir / "belief_norm_stats.npz"))

    _log.info(
        "bdi_training_start",
        n_samples=len(observations),
        epochs=_epochs,
        lr=_lr,
        batch_size=_batch_size,
        balance_classes=cfg.balance_classes,
        normalise=cfg.normalise_observations,
    )

    # Stage 1: BeliefEncoder (with classification probe for discriminative features)
    belief_weights = train_belief_encoder(
        observations, _lr, _epochs, _batch_size, intentions=intentions,
    )
    np.savez(output_dir / "belief.npz", **belief_weights)  # type: ignore[arg-type]
    _log.info("belief_encoder_saved")

    # Stage 2: DesireEncoder
    desire_weights = train_desire_encoder(observations, belief_weights, _lr, _epochs, _batch_size)
    np.savez(output_dir / "desire.npz", **desire_weights)  # type: ignore[arg-type]
    _log.info("desire_encoder_saved")

    # Stage 3: IntentionPredictor
    intention_weights = train_intention_predictor(
        observations,
        intentions,
        belief_weights,
        desire_weights,
        _lr,
        _epochs,
        _batch_size,
    )
    np.savez(output_dir / "intention.npz", **intention_weights)  # type: ignore[arg-type]
    _log.info("intention_predictor_saved")

    # Stage 4: AffectEstimator
    affect_weights = train_affect_estimator(
        observations,
        belief_weights,
        desire_weights,
        intention_weights,
        _lr,
        _epochs,
        _batch_size,
    )
    np.savez(output_dir / "affect.npz", **affect_weights)  # type: ignore[arg-type]
    _log.info("affect_estimator_saved")

    _log.info("bdi_training_complete", output_dir=str(output_dir))
    return output_dir

