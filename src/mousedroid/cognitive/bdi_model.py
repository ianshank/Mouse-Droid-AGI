"""Neural BDI (Belief-Desire-Intention) model — numpy-only inference.

Provides belief encoding, desire encoding, intention prediction and affect
estimation using lightweight numpy MLP networks.  Weights are loaded from
``.npz`` files at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_BAYESIAN_SUM_EPS: float = 1e-8
"""Floor added to Bayesian normalisation denominator."""

_SOFTMAX_EPS: float = 1e-8
"""Floor added to softmax denominator for numerical safety."""

_APPROACH_RATE_EPS: float = 1e-6
"""Minimum approach rate to avoid divide-by-zero."""

_BELIEF_DIM: int = 128
"""Dimensionality of the belief latent vector."""

_DESIRE_DIM: int = 64
"""Dimensionality of the desire latent vector."""

_INTENTION_CLASSES: int = 8
"""Number of discrete intention categories."""

_AFFECT_DIM: int = 2
"""Affect output dimensionality (valence, arousal)."""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _safe_softmax(logits: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Numerically stable softmax over a 1-D logit vector.

    Args:
        logits: 1-D array of raw logits.

    Returns:
        Probability vector summing to 1.
    """
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    total = np.sum(exp_vals) + _SOFTMAX_EPS
    return exp_vals / total


def _bayesian_normalise(
    values: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Normalise *values* so they sum to 1 (safe against all-zeros).

    Args:
        values: Non-negative 1-D array.

    Returns:
        Normalised array.
    """
    total = np.sum(values) + _BAYESIAN_SUM_EPS
    return values / total


def _relu(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Element-wise ReLU activation.

    Args:
        x: Input array.

    Returns:
        Array with negative values zeroed.
    """
    return np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# BDIInput dataclass
# ---------------------------------------------------------------------------


@dataclass
class BDIInput:
    """Pre-processed input for the Neural BDI pipeline.

    Attributes:
        belief_state: Observation vector fed to the belief encoder.
        intentions: Prior intention distribution (optional).
    """

    belief_state: NDArray[np.floating[Any]]
    intentions: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.zeros(_INTENTION_CLASSES, dtype=np.float32),
    )

    @classmethod
    def from_belief_state(cls, state: dict[str, object]) -> BDIInput:
        """Build a :class:`BDIInput` from a raw belief-state dictionary.

        Args:
            state: Dictionary with at least a ``"belief_state"`` key
                   containing a numpy array.  An optional ``"intentions"``
                   key supplies prior intention probabilities.

        Returns:
            Populated :class:`BDIInput`.
        """
        belief_arr = np.asarray(state["belief_state"], dtype=np.float32)
        intentions = np.asarray(
            state.get("intentions", np.zeros(_INTENTION_CLASSES, dtype=np.float32)),
            dtype=np.float32,
        )
        return cls(belief_state=belief_arr, intentions=intentions)


# ---------------------------------------------------------------------------
# Sub-networks (numpy MLP)
# ---------------------------------------------------------------------------


class BeliefEncoder:
    """Two-layer MLP that maps observations to a 128-d belief latent.

    Args:
        weights_path: Optional path to an ``.npz`` file with keys
            ``w1``, ``b1``, ``w2``, ``b2``.
    """

    def __init__(self, weights_path: Path | None = None) -> None:
        if weights_path is not None:
            data = np.load(weights_path)
            self._w1: NDArray[np.floating[Any]] = data["w1"]
            self._b1: NDArray[np.floating[Any]] = data["b1"]
            self._w2: NDArray[np.floating[Any]] = data["w2"]
            self._b2: NDArray[np.floating[Any]] = data["b2"]
        else:
            rng = np.random.default_rng(42)
            self._w1 = rng.standard_normal((256, _BELIEF_DIM)).astype(np.float32) * 0.01
            self._b1 = np.zeros(_BELIEF_DIM, dtype=np.float32)
            self._w2 = rng.standard_normal((_BELIEF_DIM, _BELIEF_DIM)).astype(np.float32) * 0.01
            self._b2 = np.zeros(_BELIEF_DIM, dtype=np.float32)

    def forward(self, x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Encode observation *x* to a belief vector.

        Args:
            x: 1-D observation array.

        Returns:
            128-d belief vector.
        """
        h = _relu(x @ self._w1 + self._b1)
        return _relu(h @ self._w2 + self._b2)


class DesireEncoder:
    """Single-layer MLP mapping beliefs to a 64-d desire vector.

    Args:
        weights_path: Optional ``.npz`` with ``w1``, ``b1``.
    """

    def __init__(self, weights_path: Path | None = None) -> None:
        if weights_path is not None:
            data = np.load(weights_path)
            self._w1: NDArray[np.floating[Any]] = data["w1"]
            self._b1: NDArray[np.floating[Any]] = data["b1"]
        else:
            rng = np.random.default_rng(43)
            self._w1 = rng.standard_normal((_BELIEF_DIM, _DESIRE_DIM)).astype(np.float32) * 0.01
            self._b1 = np.zeros(_DESIRE_DIM, dtype=np.float32)

    def forward(self, belief: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Map a belief vector to a desire vector.

        Args:
            belief: 128-d belief vector.

        Returns:
            64-d desire vector.
        """
        return _relu(belief @ self._w1 + self._b1)


class IntentionPredictor:
    """8-class softmax intention classifier.

    Args:
        weights_path: Optional ``.npz`` with ``w1``, ``b1``.
    """

    def __init__(self, weights_path: Path | None = None) -> None:
        if weights_path is not None:
            data = np.load(weights_path)
            self._w1: NDArray[np.floating[Any]] = data["w1"]
            self._b1: NDArray[np.floating[Any]] = data["b1"]
        else:
            rng = np.random.default_rng(44)
            self._w1 = (
                rng.standard_normal((_DESIRE_DIM, _INTENTION_CLASSES)).astype(np.float32) * 0.01
            )
            self._b1 = np.zeros(_INTENTION_CLASSES, dtype=np.float32)

    def forward(self, desire: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Predict intention probabilities from a desire vector.

        Args:
            desire: 64-d desire vector.

        Returns:
            8-class probability distribution.
        """
        logits = desire @ self._w1 + self._b1
        return _safe_softmax(logits)


class AffectEstimator:
    """Estimate valence/arousal from desire + intention signals.

    Args:
        weights_path: Optional ``.npz`` with ``w1``, ``b1``.
    """

    _INPUT_DIM: int = _DESIRE_DIM + _INTENTION_CLASSES

    def __init__(self, weights_path: Path | None = None) -> None:
        if weights_path is not None:
            data = np.load(weights_path)
            self._w1: NDArray[np.floating[Any]] = data["w1"]
            self._b1: NDArray[np.floating[Any]] = data["b1"]
        else:
            rng = np.random.default_rng(45)
            self._w1 = (
                rng.standard_normal((self._INPUT_DIM, _AFFECT_DIM)).astype(np.float32) * 0.01
            )
            self._b1 = np.zeros(_AFFECT_DIM, dtype=np.float32)

    def forward(
        self,
        desire: NDArray[np.floating[Any]],
        intentions: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Estimate affect (valence, arousal).

        Args:
            desire: 64-d desire vector.
            intentions: 8-class intention probabilities.

        Returns:
            2-d array ``[valence, arousal]`` in ``[-1, 1]``.
        """
        combined = np.concatenate([desire, intentions])
        raw = combined @ self._w1 + self._b1
        return np.tanh(raw)


# ---------------------------------------------------------------------------
# NeuralBDI — combined pipeline
# ---------------------------------------------------------------------------


class NeuralBDI:
    """Full Neural BDI inference pipeline.

    Chains :class:`BeliefEncoder` -> :class:`DesireEncoder` ->
    :class:`IntentionPredictor` + :class:`AffectEstimator`.

    Args:
        weights_dir: Optional directory containing ``belief.npz``,
            ``desire.npz``, ``intention.npz``, ``affect.npz``.
    """

    def __init__(self, weights_dir: Path | None = None) -> None:
        def _npz(name: str) -> Path | None:
            if weights_dir is None:
                return None
            p = weights_dir / f"{name}.npz"
            return p if p.exists() else None

        self._belief_enc = BeliefEncoder(_npz("belief"))
        self._desire_enc = DesireEncoder(_npz("desire"))
        self._intention_pred = IntentionPredictor(_npz("intention"))
        self._affect_est = AffectEstimator(_npz("affect"))
        _log.info("neural_bdi_init", weights_dir=str(weights_dir))

    def infer(self, belief_state: NDArray[np.floating[Any]]) -> dict[str, Any]:
        """Run full BDI inference from a raw observation vector.

        Args:
            belief_state: 1-D observation array (typically 256-d).

        Returns:
            Dictionary with keys ``belief``, ``desire``, ``intentions``,
            ``affect``, and ``approach_rate``.
        """
        belief = self._belief_enc.forward(belief_state)
        desire = self._desire_enc.forward(belief)
        intentions = self._intention_pred.forward(desire)
        affect = self._affect_est.forward(desire, intentions)

        # Compute scalar approach rate (desire magnitude, floored).
        approach_rate = float(np.linalg.norm(desire)) + _APPROACH_RATE_EPS

        return {
            "belief": belief,
            "desire": desire,
            "intentions": intentions,
            "affect": affect,
            "approach_rate": approach_rate,
        }
