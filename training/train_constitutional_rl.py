"""Phase 2.4 — Constitutional RL fine-tuning with PPO.

Trains PolicyMLP and ValueMLP using PPO in RSSM latent space with
ConstitutionalChecker as a safety constraint layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
from numpy.typing import NDArray

from mousedroid.cognitive.constitutional_rl import (
    ConstitutionalChecker,
    ConstitutionalRLConfig,
    PolicyMLP,
    ValueMLP,
)
from mousedroid.config.schema import Settings
from mousedroid.reward.model import MultiObjectiveRewardModel
from mousedroid.safety.three_laws import RoboticsLawChecker
from mousedroid.world_model.rssm import RSSM

_log = structlog.get_logger(__name__)


def _gae(
    rewards: list[float],
    values: list[float],
    gamma: float,
    gae_lambda: float,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Compute Generalised Advantage Estimation.

    Args:
        rewards: Per-step rewards.
        values: Per-step value estimates.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        Tuple of ``(advantages, returns)``.
    """
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n_steps)):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[t] = last_gae

    returns = advantages + np.array(values[:n_steps], dtype=np.float32)
    return advantages, returns


def _ppo_update(
    policy: PolicyMLP,
    value_fn: ValueMLP,
    states: NDArray[Any],
    actions: NDArray[Any],
    old_log_probs: NDArray[Any],
    advantages: NDArray[Any],
    returns: NDArray[Any],
    clip_epsilon: float,
    lr: float,
    n_epochs: int,
) -> dict[str, float]:
    """Perform PPO clipped surrogate update on numpy MLP weights.

    Uses numerical gradient approximation for the simple numpy MLPs.

    Args:
        policy: Policy network.
        value_fn: Value network.
        states: Batch of state vectors.
        actions: Batch of actions taken.
        old_log_probs: Log probabilities under old policy.
        advantages: GAE advantages.
        returns: Discounted returns.
        clip_epsilon: PPO clipping range.
        lr: Learning rate.
        n_epochs: Update epochs.

    Returns:
        Dict with loss statistics.
    """
    eps = 1e-8
    total_policy_loss = 0.0
    total_value_loss = 0.0

    for _epoch in range(n_epochs):
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            advantage = advantages[i]
            ret = returns[i]

            # Current policy output
            pred_action = policy.forward(state)
            # Gaussian log prob (simplified: unit variance)
            new_log_prob = -0.5 * np.sum((action - pred_action) ** 2)
            old_lp = old_log_probs[i]

            # PPO ratio
            ratio = np.exp(new_log_prob - old_lp)
            clipped = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            policy_loss = -min(ratio * advantage, clipped * advantage)
            total_policy_loss += policy_loss

            # Value loss
            v_pred = value_fn.forward(state)
            value_loss = (v_pred - ret) ** 2
            total_value_loss += value_loss

            # Numerical gradient update for policy
            grad_scale = lr * advantage
            direction = (action - pred_action) / (np.linalg.norm(action - pred_action) + eps)

            # Simple weight perturbation toward better actions
            h_pre = state @ policy._w1 + policy._b1
            h_mask = (h_pre > 0).astype(np.float32)
            hidden_act = h_mask * h_pre * (h_pre > 0)
            policy._w2 += grad_scale * np.outer(hidden_act, direction) * 0.001
            policy._b2 += grad_scale * direction * 0.001

            # Value update
            v_grad = 2.0 * (v_pred - ret)
            h_v = state @ value_fn._w1 + value_fn._b1
            h_v_mask = (h_v > 0).astype(np.float32)
            value_fn._w2 -= lr * v_grad * (h_v * h_v_mask).reshape(-1, 1) * 0.001
            value_fn._b2 -= lr * v_grad * np.ones(1, dtype=np.float32) * 0.001

    n = max(len(states) * n_epochs, 1)
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
    }


def train_constitutional_rl(
    cfg: Settings,
    rssm_checkpoint: Path,
    policy_init_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Full Phase 2.4 pipeline: PPO with constitutional constraints.

    Args:
        cfg: Root settings.
        rssm_checkpoint: Path to pretrained RSSM checkpoint.
        policy_init_path: Optional warm-started policy weights.
        output_dir: Output directory for final weights.

    Returns:
        Tuple of ``(output_dir, validation_results)``.
    """
    device = torch.device("cpu")
    output_dir = output_dir or Path(cfg.training.weights_dir) / "constitutional_rl"
    output_dir.mkdir(parents=True, exist_ok=True)

    ppo_cfg = cfg.ppo
    gamma = cfg.mcts.gamma

    # Load RSSM
    rssm = RSSM(cfg.model).to(device)
    rssm.load_state_dict(torch.load(rssm_checkpoint, map_location=device, weights_only=True))
    rssm.eval()

    # Build reward model with Three Laws integration
    reward_model = MultiObjectiveRewardModel(
        cfg.model,
        cfg.reward,
        law_cfg=cfg.three_laws,
    ).to(device)

    # Init policy and value networks
    policy = PolicyMLP(input_dim=cfg.model.latent_dim, action_dim=cfg.model.action_dim)
    value_fn = ValueMLP(input_dim=cfg.model.latent_dim)

    if policy_init_path and policy_init_path.exists():
        policy.load(policy_init_path)
        _log.info("policy_loaded", path=str(policy_init_path))

    # Three Laws checker (runs first)
    law_checker = RoboticsLawChecker.from_config(cfg.three_laws)

    # Constitutional checker (delegates to law checker)
    checker = ConstitutionalChecker(ConstitutionalRLConfig(), law_checker=law_checker)

    # Training loop
    n_episodes = ppo_cfg.n_training_episodes
    rollout_steps = ppo_cfg.n_rollout_steps

    for episode in range(1, n_episodes + 1):
        # Collect rollout in latent space
        h = torch.zeros(1, cfg.model.hidden_dim, device=device)
        z = torch.zeros(1, cfg.model.latent_dim, device=device)

        states: list[NDArray[Any]] = []
        actions_taken: list[NDArray[Any]] = []
        log_probs: list[float] = []
        rewards: list[float] = []
        values: list[float] = []

        for _step in range(rollout_steps):
            state_np = z.detach().cpu().numpy().flatten()
            states.append(state_np)

            # Policy action
            action_np = policy.forward(state_np)

            # Log probability (unit Gaussian)
            log_prob = -0.5 * float(np.sum(action_np**2))
            log_probs.append(log_prob)

            # Constitutional check
            context = {"battery_v": 12.0, "obstacle_dist_m": 2.0, "mcts_sims": 50}
            safe_action, violations = checker.check(action_np, context)

            actions_taken.append(safe_action)

            # Step in world model
            action_tensor = torch.as_tensor(
                safe_action,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            h, z, pred_reward = rssm.imagine_step(action_tensor, h, z)

            # Multi-objective reward
            with torch.no_grad():
                obs_recon = rssm.decode(h, z)
                reward_scalar = reward_model(obs_recon).item()

            # Stricter penalties for law violations
            law1_violations = [v for v in violations if v.startswith("[Law 1]")]
            if law1_violations:
                reward_scalar = -1.0  # Large negative for harm violations
            elif violations:
                reward_scalar = 0.0

            rewards.append(reward_scalar)
            values.append(value_fn.forward(state_np))

        # GAE
        advantages, returns = _gae(rewards, values, gamma, ppo_cfg.gae_lambda)

        # PPO update
        losses = _ppo_update(
            policy,
            value_fn,
            np.array(states),
            np.array(actions_taken),
            np.array(log_probs, dtype=np.float32),
            advantages,
            returns,
            ppo_cfg.clip_epsilon,
            cfg.training.learning_rate,
            ppo_cfg.ppo_epochs,
        )

        if episode % 100 == 0:
            _log.info(
                "ppo_episode",
                episode=episode,
                mean_reward=round(float(np.mean(rewards)), 4),
                **{k: round(v, 6) for k, v in losses.items()},
            )

    # Validation
    _log.info("validation_starting", n_episodes=ppo_cfg.n_validation_episodes)
    total_violations = 0
    total_law1_violations = 0
    total_law2_violations = 0
    total_law3_violations = 0
    val_rewards: list[float] = []

    for _ep in range(ppo_cfg.n_validation_episodes):
        h = torch.zeros(1, cfg.model.hidden_dim, device=device)
        z = torch.zeros(1, cfg.model.latent_dim, device=device)
        ep_reward = 0.0

        for _step in range(rollout_steps):
            state_np = z.detach().cpu().numpy().flatten()
            action_np = policy.forward(state_np)

            context = {"battery_v": 12.0, "obstacle_dist_m": 2.0, "mcts_sims": 50}
            safe_action, violations = checker.check(action_np, context)
            total_violations += len(violations)
            total_law1_violations += sum(1 for v in violations if v.startswith("[Law 1]"))
            total_law2_violations += sum(1 for v in violations if v.startswith("[Law 2]"))
            total_law3_violations += sum(1 for v in violations if v.startswith("[Law 3]"))

            action_tensor = torch.as_tensor(
                safe_action,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            h, z, pred_reward = rssm.imagine_step(action_tensor, h, z)
            ep_reward += pred_reward.item()

        val_rewards.append(ep_reward)

    validation_results: dict[str, object] = {
        "total_violations": total_violations,
        "law1_violations": total_law1_violations,
        "law2_violations": total_law2_violations,
        "law3_violations": total_law3_violations,
        "mean_reward": round(float(np.mean(val_rewards)), 4),
        "n_episodes": ppo_cfg.n_validation_episodes,
    }

    _log.info("validation_complete", **validation_results)

    # Save weights
    policy.save(output_dir / "policy.npz")
    value_fn.save(output_dir / "value.npz")
    _log.info("constitutional_rl_complete", output_dir=str(output_dir))

    return output_dir, validation_results
