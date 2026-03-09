"""Shared training utilities for numpy-based SGD training loops.

Provides reusable primitives used across ``train_bdi.py``,
``train_constitutional_rl.py``, and any future training scripts.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np


def iter_batches(
    n: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[np.ndarray]:
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
    weights: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
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
    for key, grad in grads.items():
        weights[key] -= lr * grad


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
    if epoch % log_every == 0:
        mean_loss = round(total_loss / max(n_batches, 1), 6)
        log_fn.info(event, epoch=epoch, loss=mean_loss)
