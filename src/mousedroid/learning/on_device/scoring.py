"""World-model rollout-return scoring harness (Phase 6 WS4).

Scores a candidate policy by its *mean imagined return* under the REUSED RSSM /
Dreamer world model. This is the user-chosen safety-gate metric: from a FIXED
set of seed states, roll the world model forward ``n_rollouts`` times for
``horizon`` imagined steps under the candidate policy, summing the world model's
predicted reward at each step, then average across all rollouts and seed states
into a single scalar.

Determinism is the load-bearing property: the same ``seed`` + the same
``seed_states`` + the same policy weights ALWAYS produce a byte-identical score,
so the promote/revert decision in the regression gate is reproducible. We seed
the global torch RNG ONCE at entry (the RSSM prior sampler draws from it inside
``imagine_step``) and run everything under ``torch.no_grad()`` + ``eval()``.

The harness is DECOUPLED from the concrete policy via :class:`PolicyProtocol`
(``(hidden, latent) -> action``). WS4 runs end-to-end against a
config-sized stand-in adapter (:class:`StateDictPolicyAdapter`); WS5 swaps in
the live policy network behind the SAME protocol without touching this module.

The world-model API reused here (``world_model/rssm.py`` /
``world_model/protocol.py``):

* ``imagine_step(action, h, z) -> (new_h, new_z, predicted_reward)`` — one
  prior-only imagined step; ``predicted_reward`` is the RSSM reward head's
  scalar output, shape ``(batch, 1)``. Already ``@torch.no_grad`` decorated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.logging.setup import get_logger
from mousedroid.world_model.protocol import WorldModelProtocol

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
        """Map ``[hidden, latent]`` through the wrapped module to an action."""
        features = torch.cat([hidden, latent], dim=-1)
        output: Tensor = self._module(features)
        if self._action_dim is not None:
            output = output[..., : self._action_dim]
        return output


def score_policy(
    policy: PolicyProtocol,
    world_model: WorldModelProtocol,
    seed_states: Sequence[SeedState],
    *,
    horizon: int,
    n_rollouts: int,
    seed: int,
) -> float:
    """Score ``policy`` by its mean imagined return under ``world_model``.

    From each seed state, runs ``n_rollouts`` imagined rollouts of ``horizon``
    steps under ``policy`` through the world model's prior, summing the reward
    head's predicted reward at each step. The returned scalar is the mean of
    those per-rollout returns across all rollouts and all seed states.

    Determinism: the global torch RNG is seeded ONCE with ``seed`` before any
    sampling, and the whole computation runs under ``torch.no_grad()``. Same
    seed + same ``seed_states`` + same policy weights ⇒ identical score.

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

    torch.manual_seed(seed)

    if isinstance(world_model, nn.Module):
        world_model.eval()

    returns: list[float] = []
    with torch.no_grad():
        for h0, z0 in seed_states:
            for _ in range(n_rollouts):
                h = h0
                z = z0
                rollout_return = torch.zeros(1, 1, dtype=torch.float32)
                for _ in range(horizon):
                    action = policy.act(h, z)
                    h, z, predicted_reward = world_model.imagine_step(action, h, z)
                    rollout_return = rollout_return + predicted_reward
                returns.append(float(rollout_return.mean().item()))

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


__all__ = ["PolicyProtocol", "SeedState", "StateDictPolicyAdapter", "score_policy"]
