"""World-model dynamics + (diagnostic) rollout-return scoring harness (Phase 6).

The Phase-6 ENABLEMENT safety gate (WS-E3) scores a candidate **RSSM** against
the live baseline **RSSM** by their held-out **reconstruction+KL loss** —
:func:`score_dynamics`. This measures the world model's dynamics-prediction
quality on REAL held-out data and is LOWER-IS-BETTER. It is the authoritative
gate metric.

:func:`score_policy` (the pre-ENABLEMENT WS4 metric — *mean imagined return*
under the RSSM reward head) is RETIRED FROM THE GATE: it sums the model's OWN
``reward_head`` along a prior rollout, so a candidate that inflates its reward
head scores HIGHER while its dynamics are actually unchanged or WORSE
(self-gaming — proven in the WS-E-SPIKE: an inflated-reward-head model scores
imagined-return +57.8 while its recon loss is byte-identical to baseline). It is
kept ONLY as a non-gating diagnostic and is NOT used by :class:`RegressionGate`.

Determinism is the load-bearing property for both: the same ``seed`` + the same
inputs + the same weights ALWAYS produce a byte-identical score, so the
promote/revert decision in the regression gate is reproducible. The global torch
RNG is seeded immediately before the scored forward (the RSSM reparam / prior
sampler draws from it) and everything runs under ``torch.no_grad()`` + ``eval()``,
with the prior global RNG + train-mode state captured + restored.

The world-model APIs reused here (``world_model/rssm.py`` /
``world_model/protocol.py``):

* ``train_sequence(batch, decoders) -> {"loss", "recon", "kl", ...}`` — the
  gradient-enabled sequence rollout; under ``no_grad`` it still returns the
  scalar recon+KL ``loss`` used by :func:`score_dynamics`.
* ``imagine_step(action, h, z) -> (new_h, new_z, predicted_reward)`` — one
  prior-only imagined step used by the diagnostic :func:`score_policy`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.logging.setup import get_logger
from mousedroid.world_model.protocol import WorldModelProtocol

if TYPE_CHECKING:
    from mousedroid.world_model.rssm import RawModalityDecoders

_log = get_logger(__name__)

#: A seed state is the world model's ``(hidden, latent)`` pair from which an
#: imagined rollout begins. Both tensors are shaped ``(1, dim)``.
SeedState = tuple[Tensor, Tensor]


@runtime_checkable
class PolicyProtocol(Protocol):
    """Minimal policy interface for the WS4 scoring harness.

    Maps a world-model latent state ``(hidden, latent)`` to an action tensor.
    Deliberately tiny so the scoring harness is decoupled from whether the
    candidate is the WS2/WS3 config-sized stand-in or the WS5 live policy net.
    """

    def act(self, hidden: Tensor, latent: Tensor) -> Tensor:
        """Return an action for the given world-model latent state.

        Args:
            hidden: RSSM hidden state, shape ``(batch, hidden_dim)``.
            latent: RSSM latent sample, shape ``(batch, latent_dim)``.

        Returns:
            Action tensor, shape ``(batch, action_dim)``.
        """
        ...


class StateDictPolicyAdapter:
    """Wrap a candidate ``nn.Module`` as a :class:`PolicyProtocol`.

    The thin WS4 adapter that lets the scoring harness run end-to-end against
    the WS2/WS3 candidate model NOW, before the live policy net is wired (WS5).
    The wrapped module is fed the concatenated ``[hidden, latent]`` vector and
    its output is sliced to ``action_dim`` so a stand-in linear of arbitrary
    output width still yields a usable action.

    Args:
        module: The candidate network (e.g. the WS2 stand-in or, in WS5, the
            live policy). Put into ``eval()`` mode at construction.
        hidden_dim: Expected RSSM hidden-state width (for input validation).
        latent_dim: Expected RSSM latent width (for input validation).
        action_dim: Optional action width to slice the module output to. When
            ``None`` the full module output is returned as the action.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        hidden_dim: int,
        latent_dim: int,
        action_dim: int | None = None,
    ) -> None:
        self._module = module
        self._module.eval()
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self._action_dim = action_dim

    def act(self, hidden: Tensor, latent: Tensor) -> Tensor:
        """Map ``[hidden, latent]`` through the wrapped module to an action.

        Raises:
            ValueError: If ``hidden`` / ``latent`` do not match the configured
                widths, or the module output has fewer than ``action_dim``
                columns (a silent under-return would feed the world model a
                truncated, malformed action).
        """
        if hidden.shape[-1] != self._hidden_dim:
            msg = f"hidden width {hidden.shape[-1]} != expected hidden_dim {self._hidden_dim}"
            raise ValueError(msg)
        if latent.shape[-1] != self._latent_dim:
            msg = f"latent width {latent.shape[-1]} != expected latent_dim {self._latent_dim}"
            raise ValueError(msg)

        features = torch.cat([hidden, latent], dim=-1)
        output: Tensor = self._module(features)
        if self._action_dim is not None:
            if output.shape[-1] < self._action_dim:
                msg = (
                    f"module output width {output.shape[-1]} < requested "
                    f"action_dim {self._action_dim}; the stand-in net must emit "
                    "at least action_dim columns"
                )
                raise ValueError(msg)
            output = output[..., : self._action_dim]
        return output


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

    Unlike :func:`score_policy` (which sums the model's OWN ``reward_head`` and so
    self-games on reward-head inflation), this scores real dynamics quality on
    real data and cannot be gamed by inflating an unused head.

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


def score_policy(
    policy: PolicyProtocol,
    world_model: WorldModelProtocol,
    seed_states: Sequence[SeedState],
    *,
    horizon: int,
    n_rollouts: int,
    seed: int,
) -> float:
    """DIAGNOSTIC-ONLY: mean imagined return under ``world_model`` (RETIRED from gate).

    .. warning::
        NOT a safety-gate metric. This sums the world model's OWN ``reward_head``
        along a prior rollout, so a candidate that inflates its (unused-in-the-
        recon-graph) reward head scores HIGHER while its real dynamics are
        unchanged or worse — it SELF-GAMES the gate. The WS-E3 regression gate
        uses :func:`score_dynamics` (held-out recon+KL loss) instead. This is
        retained ONLY as a non-gating observability diagnostic.

    From each seed state, runs ``n_rollouts`` imagined rollouts of ``horizon``
    steps under ``policy`` through the world model's prior, summing the reward
    head's predicted reward at each step. The returned scalar is the mean of
    those per-rollout returns across all rollouts and all seed states.

    Determinism: the global torch RNG is seeded ONCE with ``seed`` before any
    sampling, and the whole computation runs under ``torch.no_grad()``. Same
    seed + same ``seed_states`` + same policy weights ⇒ identical score.

    Side-effect free: the prior global RNG state is captured and restored in a
    ``finally`` so a caller sharing the process RNG is never perturbed, and a
    world model passed in ``.train()`` is left in ``.train()`` afterwards. The
    return accumulator and every per-step tensor live on the seed-state's
    device, so a GPU-resident world model (the Jetson iGPU) never triggers a
    cross-device op.

    Args:
        policy: Candidate (or baseline) policy implementing :class:`PolicyProtocol`.
        world_model: REUSED RSSM world model implementing ``imagine_step``.
        seed_states: Fixed ``(hidden, latent)`` start states (each ``(1, dim)``).
        horizon: Number of imagined steps H per rollout (``> 0``).
        n_rollouts: Number of rollouts N averaged per seed state (``> 0``).
        seed: Fixed RNG seed for reproducibility.

    Returns:
        Mean imagined return as a Python ``float``. Returns ``0.0`` when
        ``seed_states`` is empty (a degenerate but well-defined input).
    """
    if not seed_states:
        _log.warning("on_device_score_empty_seed_states")
        return 0.0

    # The accumulator + every per-step tensor must live on the world-model /
    # seed-state device, not hardcoded CPU. Derive it from the first seed
    # state's hidden tensor (the world model rolls forward from there).
    device = seed_states[0][0].device

    # Capture global RNG + train-mode state so we can restore them in a
    # ``finally`` — seeding the global RNG for reproducibility and forcing
    # ``.eval()`` must NOT leak out of this call. ``wm_module`` holds the
    # narrowed ``nn.Module`` reference (``None`` for a non-module protocol impl)
    # so the restore branch can call ``.train()`` without a cross-protocol cast.
    rng_state = torch.get_rng_state()
    wm_module = world_model if isinstance(world_model, nn.Module) else None
    was_training = wm_module.training if wm_module is not None else False

    returns: list[float] = []
    try:
        torch.manual_seed(seed)
        if wm_module is not None:
            wm_module.eval()

        with torch.no_grad():
            for h0, z0 in seed_states:
                for _ in range(n_rollouts):
                    h = h0
                    z = z0
                    rollout_return = torch.zeros(1, 1, dtype=torch.float32, device=device)
                    for _ in range(horizon):
                        action = policy.act(h, z)
                        h, z, predicted_reward = world_model.imagine_step(action, h, z)
                        rollout_return = rollout_return + predicted_reward
                    returns.append(float(rollout_return.mean().item()))
    finally:
        torch.set_rng_state(rng_state)
        if wm_module is not None and was_training:
            wm_module.train()

    score = sum(returns) / len(returns)
    _log.info(
        "on_device_score_computed",
        score=score,
        horizon=horizon,
        n_rollouts=n_rollouts,
        n_seed_states=len(seed_states),
        seed=seed,
    )
    return score


__all__ = [
    "PolicyProtocol",
    "SeedState",
    "StateDictPolicyAdapter",
    "score_dynamics",
    "score_policy",
]
