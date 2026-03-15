"""Monte Carlo Tree Search planner over RSSM latent space."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

from mousedroid.config.schema import MCTSConfig
from mousedroid.constants import DEFAULT_ACTION_DIM
from mousedroid.logging.setup import get_logger
from mousedroid.world_model.protocol import WorldModelProtocol

_log = get_logger(__name__)


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
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_candidate_actions(self, device: torch.device) -> Tensor:
        """Sample candidate actions according to the configured strategy.

        When ``action_sampling == "linspace"``, uses a 1-D grid broadcast
        (legacy behaviour).  ``"uniform"`` produces truly independent
        multi-dimensional samples for better action-space coverage.

        Returns:
            Tensor of shape ``(n_action_candidates, action_dim)``.
        """
        n = self._cfg.n_action_candidates
        action_dim = self._action_dim
        if self._cfg.action_sampling == "linspace":
            # Legacy: 1-D linspace broadcast.
            raw = torch.linspace(-1.0, 1.0, n, device=device)
            return raw.unsqueeze(-1).expand(n, action_dim)
        # Multi-dim uniform: independent samples in [-1, 1] per dimension.
        return torch.rand(n, action_dim, device=device) * 2.0 - 1.0

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
    def plan(self, h: Tensor, z: Tensor) -> Tensor:
        """Run MCTS simulations and return the best action.

        Supports three config-driven optimisations:

        1. **Early exit** — when the best-child value changes by less than
           ``early_exit_value_threshold`` for ``early_exit_patience``
           consecutive simulations, search terminates early.
        2. **Time budget** — when ``simulation_budget_ms > 0`` and elapsed
           wall-clock time exceeds the budget, search terminates early.
        3. **Action diversity** — controlled by ``action_sampling`` config.

        Args:
            h: Current hidden state, shape ``(1, hidden_dim)``.
            z: Current latent state, shape ``(1, latent_dim)``.

        Returns:
            Best action tensor, shape ``(1, action_dim)``, values in ``[-1, 1]``.
        """
        device = h.device
        dummy_action = torch.zeros(1, self._action_dim, device=device)
        root = _Node(action=dummy_action, h=h, z=z)

        # Initial expansion
        self._expand(root, device)

        # Early-exit tracking
        prev_best_value = 0.0
        stable_count = 0
        use_early_exit = self._cfg.early_exit_value_threshold > 0

        # Time-budget tracking
        use_budget = self._cfg.simulation_budget_ms > 0
        t_start = time.monotonic() if use_budget else 0.0

        for sim_idx in range(self._cfg.n_simulations_base):
            # --- Time-budget check ---
            if use_budget:
                elapsed_ms = (time.monotonic() - t_start) * 1000.0
                if elapsed_ms >= self._cfg.simulation_budget_ms:
                    _log.debug(
                        "mcts_time_budget_exit",
                        sim=sim_idx,
                        elapsed_ms=round(elapsed_ms, 1),
                    )
                    break

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

            # --- Early-exit convergence check ---
            if use_early_exit and root.children:
                best_child = max(root.children, key=lambda c: c.visit_count)
                current_best_value = best_child.mean_value
                if abs(current_best_value - prev_best_value) < self._cfg.early_exit_value_threshold:
                    stable_count += 1
                    if stable_count >= self._cfg.early_exit_patience:
                        _log.debug(
                            "mcts_early_exit",
                            sim=sim_idx,
                            value=round(current_best_value, 4),
                        )
                        break
                else:
                    stable_count = 0
                prev_best_value = current_best_value

        # Select most-visited root child.
        best_child = max(root.children, key=lambda c: c.visit_count)
        action = torch.tanh(best_child.action)
        _log.debug(
            "mcts_plan_complete",
            visits=best_child.visit_count,
            mean_value=best_child.mean_value,
        )
        return action
