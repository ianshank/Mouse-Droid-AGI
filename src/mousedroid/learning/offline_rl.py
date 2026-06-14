"""Offline RL algorithms — CQL and IQL for learning from static datasets.

Both algorithms learn Q-functions and policies from fixed datasets without
online environment interaction, addressing distribution shift via
conservative (CQL) or implicit (IQL) value estimation.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from mousedroid.constants import IQL_EXP_ADVANTAGE_CLAMP_MAX
from mousedroid.logging.setup import get_logger
from mousedroid.training.observability import (
    ExperimentLoggerProtocol,
    NoOpExperimentLogger,
    PhaseContext,
)

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared Q-network and policy
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    """Twin Q-network for offline RL algorithms.

    Two-layer MLP mapping ``(state, action)`` to a scalar Q-value.
    Twin architecture (Q1, Q2) for clipped double-Q learning.

    Args:
        state_dim: State vector dimension.
        action_dim: Action vector dimension.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        input_dim = state_dim + action_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        """Compute twin Q-values.

        Args:
            state: State tensor, shape ``(batch, state_dim)``.
            action: Action tensor, shape ``(batch, action_dim)``.

        Returns:
            Tuple of ``(q1, q2)``, each shape ``(batch, 1)``.
        """
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class DeterministicPolicy(nn.Module):
    """Deterministic policy network for offline RL.

    Two-layer MLP with tanh output, mapping state to action.

    Args:
        state_dim: State vector dimension.
        action_dim: Action vector dimension.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: Tensor) -> Tensor:
        """Compute action from state.

        Args:
            state: State tensor, shape ``(batch, state_dim)``.

        Returns:
            Action tensor in ``[-1, 1]``, shape ``(batch, action_dim)``.
        """
        return self.net(state)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Base offline RL trainer
# ---------------------------------------------------------------------------


class OfflineRLTrainer(abc.ABC):
    """Base class for offline RL algorithms.

    Subclasses implement ``update_step`` with algorithm-specific losses.

    Args:
        state_dim: State vector dimension.
        action_dim: Action vector dimension.
        hidden_dim: Hidden layer dimension.
        gamma: Discount factor.
        tau: Soft target update coefficient.
        lr: Learning rate.
        device: Torch device.
        bc_lr: Optional dedicated learning rate for the BC auxiliary loss. When
            ``None`` the policy optimizer is reused (byte-identical to the
            pre-Phase-2.1 path). When set, a separate ``bc_optimizer`` is built
            over policy parameters.
        bc_batch_size: Optional dedicated mini-batch size for the BC step.
            Stored on the trainer for downstream consumers and checkpoint
            visibility; currently informational (BC consumes the same batch as
            the actor-critic step).
        log_step_every_n: Per-update-step metric throttle. ``1`` (default)
            logs every call to ``_log_step_metrics``. Set higher (e.g. ``10``)
            for long training runs to reduce store-write overhead. The global
            step counter always increments regardless.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        device: torch.device | None = None,
        bc_lr: float | None = None,
        bc_batch_size: int | None = None,
        *,
        experiment_logger: ExperimentLoggerProtocol | None = None,
        log_phase: PhaseContext | None = None,
        log_step_every_n: int = 1,
    ) -> None:
        self._device = device or torch.device("cpu")
        self._gamma = gamma
        self._tau = tau
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._bc_lr = bc_lr
        self._bc_batch_size = bc_batch_size

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self._device)
        self.target_q_network = QNetwork(state_dim, action_dim, hidden_dim).to(
            self._device,
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.policy = DeterministicPolicy(state_dim, action_dim, hidden_dim).to(
            self._device,
        )

        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        # Phase 2.1: when ``bc_lr`` is None we alias the policy optimizer so
        # ``bc_optimizer is policy_optimizer`` and any BC step is byte-identical
        # to stepping the policy optimizer directly. When set, we build a
        # dedicated Adam over the same parameters so the BC and actor steps
        # can run at independent learning rates.
        self.bc_optimizer: torch.optim.Optimizer = (
            torch.optim.Adam(self.policy.parameters(), lr=bc_lr)
            if bc_lr is not None
            else self.policy_optimizer
        )
        self._experiment_logger: ExperimentLoggerProtocol = (
            experiment_logger or NoOpExperimentLogger()
        )
        self._log_phase = log_phase
        if log_step_every_n < 1:
            # Guard the modulo throttle: 0 would ZeroDivisionError, negatives are
            # meaningless. (The schema field is ``gt=0`` but this kwarg is
            # independently constructible, so validate at the boundary.)
            msg = f"log_step_every_n must be >= 1, got {log_step_every_n}"
            raise ValueError(msg)
        self._log_step_every_n = log_step_every_n
        self._global_step = 0
        _log.info(
            "offline_rl_bc_optimizer_built",
            bc_lr=bc_lr,
            bc_batch_size=bc_batch_size,
            shared_with_policy=self.bc_optimizer is self.policy_optimizer,
        )

    def _log_step_metrics(self, losses: dict[str, float]) -> None:
        """Forward per-update_step losses to the experiment logger.

        Called by ``update_step`` subclass implementations at the tail of
        each call. When the trainer was built without an
        ``experiment_logger`` OR without a ``log_phase`` context, this is a
        byte-identical no-op via the NoOp logger.

        The step counter ALWAYS increments (so step indices remain monotonic
        in the log), but only multiples of ``_log_step_every_n`` write to
        the backend. The default of ``1`` preserves byte-identical pre-fix
        behavior (every step is logged).
        """
        if self._log_phase is None:
            self._global_step += 1
            return
        if self._global_step % self._log_step_every_n == 0:
            for key, value in losses.items():
                self._experiment_logger.log_phase_metric(
                    self._log_phase, key, value, step=self._global_step
                )
        self._global_step += 1

    def _soft_update_targets(self) -> None:
        """Polyak-average target Q-network toward current Q-network."""
        with torch.no_grad():
            for target_param, param in zip(
                self.target_q_network.parameters(),
                self.q_network.parameters(),
                strict=True,
            ):
                target_param.mul_(1.0 - self._tau).add_(param, alpha=self._tau)

    @abc.abstractmethod
    def update_step(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> dict[str, float]:
        """Perform one gradient update step.

        Args:
            states: Batch of states, shape ``(batch, state_dim)``.
            actions: Batch of actions, shape ``(batch, action_dim)``.
            rewards: Batch of rewards, shape ``(batch,)``.
            next_states: Batch of next states, shape ``(batch, state_dim)``.
            dones: Batch of done flags, shape ``(batch,)``.

        Returns:
            Dict of loss statistics for logging.
        """

    def bc_update(
        self,
        states: Tensor,
        actions: Tensor,
        weight: float,
    ) -> dict[str, float]:
        """Apply a behavior-cloning auxiliary loss against real-replay actions.

        Computes ``weight * MSE(policy(s), a)`` and steps the policy optimizer.
        The Q-network is **not** updated, so this is a strict regularizer on
        the actor toward the demonstrator distribution.

        When ``weight <= 0`` this is a no-op and returns ``{"bc_loss": 0.0}``,
        guaranteeing byte-identical behavior for legacy training paths whose
        ``OfflineRLConfig.real_supervised_weight`` defaults to 0.0.

        Args:
            states: Real-replay states, shape ``(batch, state_dim)``.
            actions: Real-replay actions, shape ``(batch, action_dim)``.
            weight: Scalar multiplier on the BC loss. Must be ``>= 0``.

        Returns:
            ``{"bc_loss": <float>}``.
        """
        if weight <= 0.0:
            return {"bc_loss": 0.0}
        if states.shape[0] == 0:
            return {"bc_loss": 0.0}
        predicted = self.policy(states)
        bc_loss = F.mse_loss(predicted, actions)
        scaled = weight * bc_loss

        # Phase 2.1: ``bc_optimizer`` is aliased to ``policy_optimizer`` when
        # ``bc_lr is None`` (byte-identical path). When ``bc_lr`` is set, this
        # steps an independent Adam state without touching the actor's PPO
        # optimizer state.
        self.bc_optimizer.zero_grad()
        scaled.backward()  # type: ignore[no-untyped-call]
        self.bc_optimizer.step()

        return {"bc_loss": float(bc_loss.item())}

    def save(self, path: str) -> None:
        """Save all network weights.

        The ``bc_optimizer`` state is only checkpointed when it differs from
        ``policy_optimizer``. This keeps legacy (pre-Phase-2.1) checkpoints
        byte-identical when ``bc_lr is None`` and lets older code paths load
        new checkpoints without a key error.

        Args:
            path: File path for the checkpoint.
        """
        payload: dict[str, object] = {
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "policy": self.policy.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
        }
        if self.bc_optimizer is not self.policy_optimizer:
            payload["bc_optimizer"] = self.bc_optimizer.state_dict()
            payload["bc_lr"] = self._bc_lr
        torch.save(payload, path)
        _log.info(
            "offline_rl_saved",
            path=path,
            has_bc_optimizer=self.bc_optimizer is not self.policy_optimizer,
        )

    def load(self, path: str) -> None:
        """Load all network weights.

        Backwards-compatible with pre-Phase-2.1 checkpoints: ``bc_optimizer``
        state is only restored when both (a) the checkpoint contains it and
        (b) the current trainer has a dedicated ``bc_optimizer``.

        Args:
            path: File path to the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        has_bc_state = "bc_optimizer" in checkpoint
        if has_bc_state and self.bc_optimizer is not self.policy_optimizer:
            self.bc_optimizer.load_state_dict(checkpoint["bc_optimizer"])
        _log.info(
            "offline_rl_loaded",
            path=path,
            bc_optimizer_state_restored=has_bc_state
            and self.bc_optimizer is not self.policy_optimizer,
        )


# ---------------------------------------------------------------------------
# CQL — Conservative Q-Learning
# ---------------------------------------------------------------------------


class CQLTrainer(OfflineRLTrainer):
    """Conservative Q-Learning (CQL) for offline RL.

    Adds a conservative regularizer that penalises Q-values for
    out-of-distribution actions while maintaining Q-values for
    in-distribution (dataset) actions.

    Reference: Kumar et al., "Conservative Q-Learning for Offline
    Reinforcement Learning" (NeurIPS 2020).

    Args:
        state_dim: State vector dimension.
        action_dim: Action vector dimension.
        hidden_dim: Hidden layer dimension.
        gamma: Discount factor.
        tau: Soft target update coefficient.
        lr: Learning rate.
        cql_alpha: CQL regularization weight.
        n_random_actions: Number of random actions for logsumexp estimate.
        device: Torch device.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        cql_alpha: float = 1.0,
        n_random_actions: int = 10,
        device: torch.device | None = None,
        bc_lr: float | None = None,
        bc_batch_size: int | None = None,
        *,
        experiment_logger: ExperimentLoggerProtocol | None = None,
        log_phase: PhaseContext | None = None,
        log_step_every_n: int = 1,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            gamma=gamma,
            tau=tau,
            lr=lr,
            device=device,
            bc_lr=bc_lr,
            bc_batch_size=bc_batch_size,
            experiment_logger=experiment_logger,
            log_phase=log_phase,
            log_step_every_n=log_step_every_n,
        )
        self._cql_alpha = cql_alpha
        self._n_random_actions = n_random_actions

    def _cql_regularizer(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Compute CQL conservative penalty.

        Estimates ``log(sum_a exp(Q(s,a)))`` via uniform random actions,
        then subtracts ``Q(s, a_data)`` to penalise overestimation of
        OOD actions.

        Args:
            states: Batch of states.
            actions: Batch of dataset actions.

        Returns:
            Scalar CQL penalty.
        """
        batch_size = states.shape[0]

        # Sample random actions uniformly in [-1, 1]
        random_actions = (
            torch.rand(
                batch_size,
                self._n_random_actions,
                self._action_dim,
                device=self._device,
            )
            * 2.0
            - 1.0
        )

        # Q-values for random actions
        states_rep = states.unsqueeze(1).expand(-1, self._n_random_actions, -1)
        states_flat = states_rep.reshape(-1, self._state_dim)
        actions_flat = random_actions.reshape(-1, self._action_dim)

        q1_random, q2_random = self.q_network(states_flat, actions_flat)
        q1_random = q1_random.reshape(batch_size, self._n_random_actions)
        q2_random = q2_random.reshape(batch_size, self._n_random_actions)

        # LogSumExp over random actions
        q1_logsumexp = torch.logsumexp(q1_random, dim=1).mean()
        q2_logsumexp = torch.logsumexp(q2_random, dim=1).mean()

        # Q-values for dataset actions
        q1_data, q2_data = self.q_network(states, actions)

        # CQL penalty: push down random, push up dataset
        cql_loss: Tensor = q1_logsumexp - q1_data.mean() + q2_logsumexp - q2_data.mean()

        return cql_loss

    def update_step(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> dict[str, float]:
        """Perform one CQL update step.

        Combines standard Bellman backup with CQL conservative regularizer.
        """
        # --- Q-function update ---
        with torch.no_grad():
            next_actions = self.policy(next_states)
            target_q1, target_q2 = self.target_q_network(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards.unsqueeze(-1) + self._gamma * (1.0 - dones.unsqueeze(-1)) * target_q

        q1, q2 = self.q_network(states, actions)
        bellman_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        cql_loss = self._cql_regularizer(states, actions)
        q_loss = bellman_loss + self._cql_alpha * cql_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()  # type: ignore[no-untyped-call]
        self.q_optimizer.step()

        # --- Policy update ---
        policy_actions = self.policy(states)
        q1_pi, _ = self.q_network(states, policy_actions)
        policy_loss = -q1_pi.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # --- Target update ---
        self._soft_update_targets()

        losses = {
            "q_loss": q_loss.item(),
            "bellman_loss": bellman_loss.item(),
            "cql_loss": cql_loss.item(),
            "policy_loss": policy_loss.item(),
        }
        self._log_step_metrics(losses)
        return losses


# ---------------------------------------------------------------------------
# IQL — Implicit Q-Learning
# ---------------------------------------------------------------------------


class ValueNetwork(nn.Module):
    """State value network for IQL.

    Args:
        state_dim: State vector dimension.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: Tensor) -> Tensor:
        """Compute state value.

        Args:
            state: State tensor, shape ``(batch, state_dim)``.

        Returns:
            Value tensor, shape ``(batch, 1)``.
        """
        return self.net(state)  # type: ignore[no-any-return]


class IQLTrainer(OfflineRLTrainer):
    """Implicit Q-Learning (IQL) for offline RL.

    Avoids querying OOD actions entirely by using expectile regression
    on the value function and advantage-weighted policy extraction.

    Reference: Kostrikov et al., "Offline Reinforcement Learning with
    Implicit Q-Learning" (ICLR 2022).

    Args:
        state_dim: State vector dimension.
        action_dim: Action vector dimension.
        hidden_dim: Hidden layer dimension.
        gamma: Discount factor.
        tau: Soft target update coefficient.
        lr: Learning rate.
        iql_tau: Expectile for asymmetric value loss (higher = more optimistic).
        beta: Inverse temperature for advantage-weighted policy extraction.
        device: Torch device.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        device: torch.device | None = None,
        bc_lr: float | None = None,
        bc_batch_size: int | None = None,
        *,
        experiment_logger: ExperimentLoggerProtocol | None = None,
        log_phase: PhaseContext | None = None,
        log_step_every_n: int = 1,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            gamma=gamma,
            tau=tau,
            lr=lr,
            device=device,
            bc_lr=bc_lr,
            bc_batch_size=bc_batch_size,
            experiment_logger=experiment_logger,
            log_phase=log_phase,
            log_step_every_n=log_step_every_n,
        )
        self._iql_tau = iql_tau
        self._beta = beta

        self.value_network = ValueNetwork(state_dim, hidden_dim).to(self._device)
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=lr,
        )

    def _expectile_loss(self, diff: Tensor) -> Tensor:
        """Asymmetric (expectile) squared loss.

        Args:
            diff: ``Q - V`` difference tensor.

        Returns:
            Expectile loss scalar.
        """
        weight = torch.where(diff > 0, self._iql_tau, 1.0 - self._iql_tau)
        return (weight * diff.pow(2)).mean()

    def update_step(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> dict[str, float]:
        """Perform one IQL update step.

        Three-phase update: value function (expectile), Q-functions
        (Bellman with V-targets), policy (advantage-weighted regression).
        """
        # --- Value function update (expectile regression) ---
        with torch.no_grad():
            target_q1, target_q2 = self.target_q_network(states, actions)
            target_q = torch.min(target_q1, target_q2)

        v = self.value_network(states)
        value_loss = self._expectile_loss(target_q - v)

        self.value_optimizer.zero_grad()
        value_loss.backward()  # type: ignore[no-untyped-call]
        self.value_optimizer.step()

        # --- Q-function update (Bellman with V-targets) ---
        with torch.no_grad():
            next_v = self.value_network(next_states)
            target = rewards.unsqueeze(-1) + self._gamma * (1.0 - dones.unsqueeze(-1)) * next_v

        q1, q2 = self.q_network(states, actions)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.q_optimizer.zero_grad()
        q_loss.backward()  # type: ignore[no-untyped-call]
        self.q_optimizer.step()

        # --- Policy update (advantage-weighted regression) ---
        with torch.no_grad():
            v_for_adv = self.value_network(states)
            q1_for_adv, q2_for_adv = self.target_q_network(states, actions)
            q_for_adv = torch.min(q1_for_adv, q2_for_adv)
            advantage = q_for_adv - v_for_adv

            # Clip advantage weights for stability
            exp_advantage = torch.exp(self._beta * advantage).clamp(
                max=IQL_EXP_ADVANTAGE_CLAMP_MAX,
            )

        policy_actions = self.policy(states)
        mse = F.mse_loss(policy_actions, actions, reduction="none").sum(dim=-1, keepdim=True)
        policy_loss = (exp_advantage * mse).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()  # type: ignore[no-untyped-call]
        self.policy_optimizer.step()

        # --- Target update ---
        self._soft_update_targets()

        losses = {
            "q_loss": q_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
        }
        self._log_step_metrics(losses)
        return losses

    def save(self, path: str) -> None:
        """Save all network weights including value network.

        Args:
            path: File path for the checkpoint.
        """
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
                "policy": self.policy.state_dict(),
                "value_network": self.value_network.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
            },
            path,
        )
        _log.info("iql_saved", path=path)

    def load(self, path: str) -> None:
        """Load all network weights including value network.

        Args:
            path: File path to the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.value_network.load_state_dict(checkpoint["value_network"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        _log.info("iql_loaded", path=path)
