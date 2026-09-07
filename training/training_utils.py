"""Shared training utilities for numpy-based SGD training loops.

Provides reusable primitives used across ``train_bdi.py``,
``train_constitutional_rl.py``, and any future training scripts.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray


def iter_batches(
    n: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[NDArray[np.intp]]:
    """Yield shuffled index arrays for one epoch of mini-batch training.

    Generates a random permutation of ``range(n)`` and slices it into
    consecutive batches of length ``batch_size``.  The final incomplete
    batch (if any) is silently dropped to keep batch sizes uniform.

    Args:
        n: Total number of samples.
        batch_size: Number of samples per batch.
        rng: NumPy random generator (provides reproducibility when seeded).

    Yields:
        1-D integer index arrays of length ``batch_size``.

    Example::

        rng = np.random.default_rng(0)
        for idx in iter_batches(200, 32, rng):
            x_batch = X[idx]
    """
    perm = rng.permutation(n)
    for start in range(0, n - batch_size + 1, batch_size):
        yield perm[start : start + batch_size]


def sgd_step(
    weights: dict[str, NDArray[Any]],
    grads: dict[str, NDArray[Any]],
    lr: float,
) -> None:
    """Apply one vanilla SGD update to a weight dictionary in-place.

    For each key present in *grads*, subtracts ``lr * grads[key]`` from
    ``weights[key]``.  Keys in *weights* that have no corresponding entry in
    *grads* are left unchanged.

    Args:
        weights: Mutable dictionary mapping parameter names to arrays.
        grads: Dictionary mapping parameter names to gradient arrays with
            the same shapes as the corresponding weight arrays.
        lr: Learning rate (step size).

    Example::

        weights = {"w": np.zeros((4, 2)), "b": np.zeros(2)}
        grads   = {"w": np.ones((4, 2)), "b": np.ones(2)}
        sgd_step(weights, grads, lr=0.01)
        # weights["w"] is now -0.01 everywhere
    """
    unknown_keys = grads.keys() - weights.keys()
    if unknown_keys:
        raise KeyError(f"Gradient provided for unknown parameter(s): {sorted(unknown_keys)}")
    for key, grad in grads.items():
        weights[key] -= lr * grad


class AdamOptimizer:
    """In-place Adam for a dict of numpy parameter arrays.

    Mutates the same arrays held in *weights* so training loops can keep
    using local names (``w1``, ``b1``, …) after wrapping them in a dict.
    """

    def __init__(
        self,
        weights: dict[str, NDArray[Any]],
        *,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Initialise first/second moment buffers.

        Args:
            weights: Parameter dict. Arrays are updated in-place by :meth:`step`.
            lr: Step size.
            beta1: Exponential decay for the first moment.
            beta2: Exponential decay for the second moment.
            eps: Numerical floor under the RMS denominator.
        """
        self._weights = weights
        self._m = {name: np.zeros_like(value) for name, value in weights.items()}
        self._v = {name: np.zeros_like(value) for name, value in weights.items()}
        self._lr = float(lr)
        self._beta1 = float(beta1)
        self._beta2 = float(beta2)
        self._eps = float(eps)
        self._t = 0

    def step(self, grads: dict[str, NDArray[Any]]) -> None:
        """Apply one Adam update to the wrapped parameter dict.

        Args:
            grads: Mapping of parameter name to gradient. Extra keys raise
                ``KeyError`` (same contract as :func:`sgd_step`).

        Raises:
            KeyError: A gradient name is not in the wrapped weights.
        """
        unknown_keys = grads.keys() - self._weights.keys()
        if unknown_keys:
            raise KeyError(f"Gradient provided for unknown parameter(s): {sorted(unknown_keys)}")
        self._t += 1
        beta1_correction = 1.0 - self._beta1**self._t
        beta2_correction = 1.0 - self._beta2**self._t
        for name, grad in grads.items():
            self._m[name] *= self._beta1
            self._m[name] += (1.0 - self._beta1) * grad
            self._v[name] *= self._beta2
            self._v[name] += (1.0 - self._beta2) * (grad * grad)
            m_hat = self._m[name] / beta1_correction
            v_hat = self._v[name] / beta2_correction
            self._weights[name] -= self._lr * m_hat / (np.sqrt(v_hat) + self._eps)


def log_epoch_loss(
    log_fn: Any,
    event: str,
    epoch: int,
    total_loss: float,
    n_batches: int,
    *,
    log_every: int = 20,
) -> None:
    """Emit a structured log line for a training epoch if due.

    Computes the mean batch loss and calls ``log_fn.info(event, ...)`` when
    ``epoch % log_every == 0``.

    Args:
        log_fn: A structlog (or compatible) logger instance.
        event: Log event name (e.g. ``"belief_epoch"``).
        epoch: Current epoch number (1-indexed).
        total_loss: Accumulated loss sum across all batches.
        n_batches: Number of batches processed (used for mean computation).
        log_every: Frequency at which to emit the log line.
    """
    if log_every <= 0:
        raise ValueError(f"log_every must be a positive integer, got {log_every!r}")
    if epoch <= 0:
        raise ValueError(f"epoch must be a positive integer (1-indexed), got {epoch!r}")
    if epoch % log_every == 0:
        mean_loss = round(total_loss / max(n_batches, 1), 6)
        log_fn.info(event, epoch=epoch, loss=mean_loss)
