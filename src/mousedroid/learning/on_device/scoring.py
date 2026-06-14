"""World-model dynamics scoring harness for the on-device gate (Phase 6 WS-E3).

The Phase-6 ENABLEMENT safety gate scores a candidate **RSSM** against the live
baseline **RSSM** by their held-out **reconstruction+KL loss** —
:func:`score_dynamics`. This measures the world model's dynamics-prediction
quality on REAL held-out data and is LOWER-IS-BETTER. It is the authoritative
gate metric.

The pre-ENABLEMENT WS4 metric (*mean imagined return* under the RSSM reward
head) was RETIRED — it summed the model's OWN ``reward_head`` along a prior
rollout, so a candidate that inflated its reward head scored HIGHER while its
dynamics were actually unchanged or WORSE (self-gaming — proven in the
WS-E-SPIKE: an inflated-reward-head model scored imagined-return +57.8 while its
recon loss was byte-identical to baseline). The recon-loss gate cannot be gamed
that way and fully replaced it.

Determinism is the load-bearing property: the same ``seed`` + the same inputs +
the same weights ALWAYS produce a byte-identical score, so the promote/revert
decision in the regression gate is reproducible. The global torch RNG is seeded
immediately before the scored forward (the RSSM reparam / prior sampler draws
from it) and everything runs under ``torch.no_grad()`` + ``eval()``, with the
prior global RNG + train-mode state captured + restored.

The world-model API reused here (``world_model/rssm.py``):

* ``train_sequence(batch, decoders) -> {"loss", "recon", "kl", ...}`` — the
  gradient-enabled sequence rollout; under ``no_grad`` it still returns the
  scalar recon+KL ``loss`` used by :func:`score_dynamics`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mousedroid.logging.setup import get_logger
from mousedroid.world_model.protocol import WorldModelProtocol

if TYPE_CHECKING:
    from mousedroid.world_model.rssm import RawModalityDecoders

_log = get_logger(__name__)


def score_dynamics(
    world_model: WorldModelProtocol,
    batch: Mapping[str, Tensor],
    decoders: RawModalityDecoders,
    *,
    seed: int,
) -> float:
    """Score ``world_model`` by its held-out reconstruction+KL loss (LOWER better).

    This is the authoritative WS-E3 regression-gate metric. It runs the world
    model's ``train_sequence`` over a FIXED held-out ``(B, T, ...)`` batch and
    returns the scalar ``recon + kl_beta * kl`` loss as a Python ``float``. A
    lower loss means the candidate RSSM predicts the held-out dynamics better;
    a heavily-degraded candidate blows the KL up to a very-large / non-finite
    value, which the gate correctly treats as worse-than-baseline.

    Unlike the retired imagined-return metric (which summed the model's OWN
    ``reward_head`` and so self-gamed on reward-head inflation), this scores real
    dynamics quality on real data and cannot be gamed by inflating an unused head.

    Determinism: the global torch RNG is seeded with ``seed`` IMMEDIATELY before
    the ``train_sequence`` call (its reparameterisation noise draws from the
    global RNG), so the SAME ``seed`` + ``batch`` + ``decoders`` + weights ALWAYS
    yield a byte-identical loss. The gate uses the SAME ``seed`` and the SAME
    ``decoders`` instance for both baseline and candidate so the comparison is
    apples-to-apples (the recon heads are external to the RSSM ``state_dict``).

    Side-effect free: the scored model is forced into ``.eval()`` under
    ``torch.no_grad()`` for the loss computation and restored to its prior
    train-mode afterwards; the prior global RNG state is captured + restored so a
    caller sharing the process RNG is never perturbed. No autograd graph is built
    (``no_grad``), so the model parameters never accumulate a ``.grad``.

    Args:
        world_model: The :class:`~mousedroid.world_model.rssm.RSSM` (candidate or
            baseline) to score. The recon-loss gate is supported ONLY on the
            concrete ``RSSM`` (the only engine exposing ``train_sequence``); a
            non-RSSM engine raises :class:`TypeError`.
        batch: A FIXED held-out ``(B, T, ...)`` sequence-dict batch (built by the
            WS-E2 ``build_sequence_batch`` over a held-out replay slice DISJOINT
            from the refine batch).
        decoders: The SHARED reconstruction heads used to score BOTH baseline and
            candidate (recon heads live external to the RSSM ``state_dict``).
        seed: Fixed RNG seed for reproducibility (``cfg.scoring_seed``).

    Returns:
        The held-out ``recon + kl_beta * kl`` loss as a Python ``float`` (may be
        non-finite for a heavily-degraded candidate — the caller treats that as
        worse-than-baseline and reverts).

    Raises:
        TypeError: If ``world_model`` is not a concrete ``RSSM`` (only the RSSM
            exposes ``train_sequence``; ``DualStreamRSSM`` / the ONNX engine do
            not, so the recon-loss gate cannot score them).
    """
    # Import the concrete RSSM here (not at module top) so the heavy world-model
    # module is only pulled in when the gate actually scores. Narrowing to RSSM
    # gives a statically-typed ``train_sequence`` call (no suppression) — only the
    # RSSM exposes it.
    from mousedroid.world_model.rssm import RSSM

    if not isinstance(world_model, RSSM):
        msg = (
            "score_dynamics requires a concrete RSSM exposing train_sequence; "
            f"got {type(world_model).__name__}"
        )
        raise TypeError(msg)

    rng_state = torch.get_rng_state()
    was_training = world_model.training
    decoder_was_training = decoders.training

    try:
        torch.manual_seed(seed)
        world_model.eval()
        decoders.eval()
        with torch.no_grad():
            out = world_model.train_sequence(dict(batch), decoders)
            loss = float(out["loss"].item())
    finally:
        torch.set_rng_state(rng_state)
        if was_training:
            world_model.train()
        if decoder_was_training:
            decoders.train()

    _log.info(
        "on_device_dynamics_score_computed",
        loss=loss,
        finite=math.isfinite(loss),
        seed=seed,
    )
    return loss


__all__ = ["score_dynamics"]
