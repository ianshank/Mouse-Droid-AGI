"""Monte Carlo Tree Search planner over RSSM latent space."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from mousedroid.config.schema import MCTSConfig
from mousedroid.constants import (
    DEFAULT_ACTION_DIM,
    DEFAULT_ACTION_LIMIT,
    R_D_NEWTON_ITERATIONS,
    R_D_NEWTON_START,
    R_D_SEQUENCE_OFFSET,
)
from mousedroid.logging.setup import get_logger
from mousedroid.world_model.protocol import WorldModelProtocol

_log = get_logger(__name__)

_AXIS_SIGNS: tuple[float, ...] = (1.0, -1.0)
"""Directions emitted per action axis, in order, for the spanning candidate set.

Interleaving +/- per axis (rather than grouping all positives first) is what
makes truncation drop whole axes last-first instead of stripping every reverse
direction. The primitive count derives from ``len(...)``, so nothing downstream
restates "two directions per axis" as a literal.
"""


def _low_discrepancy_points(count: int, dim: int, device: torch.device) -> Tensor:
    """Deterministic quasirandom points in ``[0, 1)``, shape ``(count, dim)``.

    Uses the R_d additive recurrence (Roberts' generalisation of the golden
    ratio): ``x_n = frac(0.5 + n * alpha)`` where ``alpha_i = phi**-(i+1)`` and
    ``phi`` is the positive root of ``x**(dim+1) = x + 1``. Chosen over
    ``torch.quasirandom.SobolEngine`` because it is a handful of typed tensor
    ops with no untyped-stub dependency, needs no engine state, and is
    bit-reproducible across processes and platforms.

    Args:
        count: Number of points to generate.
        dim: Dimensionality of each point.
        device: Torch device for the returned tensor.

    Returns:
        Tensor of shape ``(count, dim)`` with values in ``[0, 1)``.
    """
    if dim < 1:
        msg = f"low-discrepancy points need dim >= 1, got {dim}"
        raise ValueError(msg)
    phi = R_D_NEWTON_START
    for _ in range(R_D_NEWTON_ITERATIONS):
        numerator = phi ** (dim + 1) - phi - 1.0
        denominator = (dim + 1) * phi**dim - 1.0
        phi -= numerator / denominator
    alpha = torch.tensor(
        [phi ** -(axis + 1) for axis in range(dim)], dtype=torch.float32, device=device
    )
    steps = torch.arange(1, count + 1, dtype=torch.float32, device=device).unsqueeze(-1)
    return torch.frac(R_D_SEQUENCE_OFFSET + steps * alpha)


@dataclass
class _Node:
    """Internal MCTS tree node."""

    action: Tensor
    h: Tensor
    z: Tensor
    visit_count: int = 0
    total_value: float = 0.0
    children: list[_Node] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        """Average backed-up value."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MCTSPlanner:
    """Monte Carlo Tree Search for action selection in latent space.

    Uses UCB1 for tree policy and the world model's ``imagine_step``
    for rollout simulation.

    Args:
        cfg: MCTS configuration (simulations, depth, UCB constant, etc.).
        world_model: A world model implementing ``WorldModelProtocol``.
    """

    def __init__(
        self,
        cfg: MCTSConfig,
        world_model: WorldModelProtocol,
        action_dim: int = DEFAULT_ACTION_DIM,
    ) -> None:
        self._cfg = cfg
        self._world_model = world_model
        self._action_dim = action_dim

        _log.info(
            "mcts_init",
            n_simulations_base=cfg.n_simulations_base,
            rollout_depth=cfg.rollout_depth,
            ucb_c=cfg.ucb_c,
            n_action_candidates=cfg.n_action_candidates,
            action_candidate_strategy=cfg.action_candidate_strategy,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_candidate_actions(self, device: torch.device) -> Tensor:
        """Build the candidate action set the tree policy selects between.

        Dispatches on ``MCTSConfig.action_candidate_strategy``. See that
        field's description for why the default is rank-deficient and why
        the spanning alternative is opt-in rather than a silent flip.

        Returns:
            Tensor of shape ``(n_action_candidates, action_dim)``, values in
            ``[-DEFAULT_ACTION_LIMIT, +DEFAULT_ACTION_LIMIT]``.
        """
        if self._cfg.action_candidate_strategy == "per_axis":
            return self._per_axis_candidates(device)
        return self._shared_axis_candidates(device)

    def _shared_axis_candidates(self, device: torch.device) -> Tensor:
        """Legacy candidate set: one linspace broadcast across every axis.

        Preserved byte-identically for backwards compatibility. Every row
        satisfies ``vx == vy == omega``, so the returned matrix has rank 1 and
        cannot express "drive straight" or "turn in place".

        Returns:
            Tensor of shape ``(n_action_candidates, action_dim)``.
        """
        n = self._cfg.n_action_candidates
        action_dim = self._action_dim
        raw = torch.linspace(-DEFAULT_ACTION_LIMIT, DEFAULT_ACTION_LIMIT, n, device=device)
        # Expand to full action space: repeat across action dims.
        actions = raw.unsqueeze(-1).expand(n, action_dim)
        return actions

    def _per_axis_candidates(self, device: torch.device) -> Tensor:
        """Spanning candidate set: stop, per-axis unit moves, quasirandom fill.

        Ordered by decreasing operational importance so that a
        ``n_action_candidates`` smaller than the primitive count degrades
        predictably (stop is never dropped) rather than arbitrarily:

        1. the zero action (full stop);
        2. ``+limit`` then ``-limit`` along each action axis in turn;
        3. a deterministic low-discrepancy fill for any remaining slots.

        Dimension-agnostic: no axis count, candidate count or bound is
        hardcoded here. Built functionally — no in-place tensor writes — per
        invariant 3 of this subsystem's contract.

        The fill needs no de-duplication. ``_low_discrepancy_points`` is an
        irrational rotation evaluated from step 1, so it never returns the
        cube centre (step 0, the one point that would map onto the stop
        action) and never repeats; the primitives sit exactly on 0 and
        +/-limit, which the rotation misses for the same reason. Uniqueness is
        asserted by the property tier rather than defended at runtime.

        Returns:
            Tensor of shape ``(n_action_candidates, action_dim)``, contiguous.
        """
        n = self._cfg.n_action_candidates
        action_dim = self._action_dim
        stop = torch.zeros(1, action_dim, device=device)
        axes = torch.eye(action_dim, device=device) * DEFAULT_ACTION_LIMIT
        # Interleave +axis / -axis so truncation drops whole axes last-first
        # rather than stripping every negative direction.
        signed = torch.stack([axes * sign for sign in _AXIS_SIGNS], dim=1).reshape(
            len(_AXIS_SIGNS) * action_dim, action_dim
        )
        blocks: list[Tensor] = [stop, signed]

        n_primitives = 1 + len(_AXIS_SIGNS) * action_dim
        if n < n_primitives:
            _log.warning(
                "mcts_candidates_truncated",
                n_action_candidates=n,
                n_primitives=n_primitives,
                action_dim=action_dim,
                detail="fewer candidates than motion primitives; some axes unreachable",
            )
        elif n > n_primitives:
            pool = _low_discrepancy_points(n - n_primitives, action_dim, device)
            lower, upper = -DEFAULT_ACTION_LIMIT, DEFAULT_ACTION_LIMIT
            blocks.append(pool * (upper - lower) + lower)

        # ``-axes`` yields -0.0 entries; ``+ 0.0`` normalises them so the
        # driver never emits a signed negative zero the legacy path could not.
        actions = (torch.cat(blocks, dim=0)[:n] + 0.0).contiguous()
        _log.debug(
            "mcts_candidates_built",
            strategy="per_axis",
            shape=tuple(actions.shape),
            n_primitives=min(n_primitives, n),
        )
        return actions

    def _ucb1(self, node: _Node, parent_visits: int) -> float:
        """Compute UCB1 score for child selection.

        Args:
            node: Child node to evaluate.
            parent_visits: Total visits of the parent.

        Returns:
            UCB1 score.
        """
        if node.visit_count == 0:
            return float("inf")
        exploitation = node.mean_value
        exploration = self._cfg.ucb_c * math.sqrt(math.log(parent_visits) / node.visit_count)
        return exploitation + exploration

    def _select_child(self, node: _Node) -> _Node:
        """Select best child via UCB1.

        Args:
            node: Parent node.

        Returns:
            Best child node.
        """
        best_score = -float("inf")
        best_child = node.children[0]
        for child in node.children:
            score = self._ucb1(child, node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _expand(self, node: _Node, device: torch.device) -> None:
        """Expand a leaf node by adding children for all candidate actions.

        Args:
            node: Leaf node to expand.
            device: Torch device for tensor creation.
        """
        candidates = self._generate_candidate_actions(device)
        for i in range(candidates.shape[0]):
            action = candidates[i : i + 1]
            new_h, new_z, _ = self._world_model.imagine_step(action, node.h, node.z)
            child = _Node(action=action, h=new_h, z=new_z)
            node.children.append(child)

    def _rollout(self, h: Tensor, z: Tensor, depth: int) -> float:
        """Random rollout from a node to estimate value.

        Args:
            h: Hidden state at rollout start.
            z: Latent state at rollout start.
            depth: Remaining rollout depth.

        Returns:
            Discounted cumulative reward.
        """
        total_reward = 0.0
        gamma_acc = 1.0
        cur_h, cur_z = h, z
        for _ in range(depth):
            random_action = torch.tanh(torch.randn(1, self._action_dim, device=h.device))
            cur_h, cur_z, reward = self._world_model.imagine_step(random_action, cur_h, cur_z)
            total_reward += gamma_acc * float(reward.item())
            gamma_acc *= self._cfg.gamma
        return total_reward

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def plan(self, h: Tensor, z: Tensor, *, n_simulations: int | None = None) -> Tensor:
        """Run MCTS simulations and return the best action.

        Args:
            h: Current hidden state, shape ``(1, hidden_dim)``.
            z: Current latent state, shape ``(1, latent_dim)``.
            n_simulations: Number of simulations to run. If ``None``,
                uses ``cfg.n_simulations_base``. Pass a surprise-adaptive
                budget via :func:`~mousedroid.agents._planning.compute_mcts_budget`.

        Returns:
            Best action tensor, shape ``(1, action_dim)``, values in ``[-1, 1]``.
        """
        budget = n_simulations if n_simulations is not None else self._cfg.n_simulations_base
        device = h.device
        dummy_action = torch.zeros(1, self._action_dim, device=device)
        root = _Node(action=dummy_action, h=h, z=z)

        # Initial expansion
        self._expand(root, device)

        for _ in range(budget):
            node = root
            path: list[_Node] = [node]

            # Selection: descend to a leaf.
            while node.children:
                node = self._select_child(node)
                path.append(node)

            # Expansion
            if node.visit_count > 0:
                self._expand(node, device)
                if node.children:
                    node = node.children[0]
                    path.append(node)

            # Rollout
            value = self._rollout(node.h, node.z, self._cfg.rollout_depth)

            # Backpropagation
            for ancestor in path:
                ancestor.visit_count += 1
                ancestor.total_value += value

        # Select most-visited root child.
        best_child = max(root.children, key=lambda c: c.visit_count)
        action = best_child.action
        _log.debug(
            "mcts_plan_complete",
            n_simulations=budget,
            visits=best_child.visit_count,
            mean_value=best_child.mean_value,
        )
        return action
