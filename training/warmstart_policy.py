"""Phase 2.2 — MCTS policy warm-start and UCB tuning.

Initialises PolicyMLP weights from RSSM latent statistics and tunes
the UCB exploration constant via simulated rollouts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import structlog
import torch

from mousedroid.cognitive.constitutional_rl import PolicyMLP
from mousedroid.config.schema import MCTSConfig, ModelConfig, Settings
from mousedroid.world_model.mcts import MCTSPlanner
from mousedroid.world_model.rssm import RSSM
from training.rssm_dataset import RSSMSequenceDataset

_log = structlog.get_logger(__name__)


def compute_latent_statistics(
    rssm: RSSM,
    dataset: RSSMSequenceDataset,
    device: torch.device,
    max_episodes: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of RSSM latent states over dataset.

    Args:
        rssm: Pretrained RSSM model.
        dataset: Sequence dataset for computing stats.
        device: Torch device.
        max_episodes: Max episodes to sample.

    Returns:
        Tuple of ``(latent_mean, latent_std)`` as numpy arrays.
    """
    all_z: list[np.ndarray] = []
    n = min(len(dataset), max_episodes)

    for i in range(n):
        vision, ultrasonic, motor_state, valid_mask, actions = dataset[i]
        vision = vision.unsqueeze(0).to(device)
        ultrasonic = ultrasonic.unsqueeze(0).to(device)
        motor_state = motor_state.unsqueeze(0).to(device)
        valid_mask = valid_mask.unsqueeze(0).to(device)
        actions = actions.unsqueeze(0).to(device)

        h = torch.zeros(1, rssm._cfg.hidden_dim, device=device)
        z = torch.zeros(1, rssm._cfg.latent_dim, device=device)

        with torch.no_grad():
            for t in range(vision.shape[1]):
                obs_embed = rssm.encoder(
                    vision[:, t], ultrasonic[:, t], motor_state[:, t], valid_mask[:, t],
                )
                prev_action = actions[:, max(0, t - 1)]
                gru_input = torch.cat([z, prev_action], dim=-1)
                h = rssm.gru(gru_input, h)
                post_params = rssm.posterior(torch.cat([h, obs_embed], dim=-1))
                z, _, _ = rssm._sample_gaussian(post_params)
                all_z.append(z.cpu().numpy().flatten())

    z_array = np.stack(all_z)
    return z_array.mean(axis=0), z_array.std(axis=0) + 1e-8


def warmstart_policy(
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    input_dim: int = 128,
    action_dim: int = 2,
) -> PolicyMLP:
    """Initialise PolicyMLP weights from RSSM latent statistics.

    Sets bias to match latent distribution centre and scales weights
    by inverse latent std for better initial conditioning.

    Args:
        latent_mean: Mean of RSSM latent states.
        latent_std: Std of RSSM latent states.
        input_dim: Policy input dimensionality.
        action_dim: Policy output dimensionality.

    Returns:
        Warm-started PolicyMLP.
    """
    policy = PolicyMLP(input_dim=input_dim, action_dim=action_dim)

    # Scale first-layer weights by inverse latent std for normalisation
    scale = 1.0 / latent_std[:input_dim] if len(latent_std) >= input_dim else np.ones(input_dim)
    policy._w1 = policy._w1 * scale[:, np.newaxis] if scale.shape[0] == policy._w1.shape[0] else policy._w1

    # Shift bias toward latent mean
    if len(latent_mean) >= input_dim:
        policy._b1 = -latent_mean[:input_dim] * scale[:input_dim] * 0.01

    return policy


def tune_ucb(
    rssm: RSSM,
    base_cfg: MCTSConfig,
    n_episodes: int = 1000,
    target_ms: float = 50.0,
    n_simulations: int = 200,
    device: torch.device | None = None,
) -> tuple[float, dict[str, object]]:
    """Grid search UCB exploration constant via simulated rollouts.

    Args:
        rssm: Pretrained RSSM for world model rollouts.
        base_cfg: Base MCTS config to modify.
        n_episodes: Episodes per UCB candidate.
        target_ms: Target search latency in milliseconds.
        n_simulations: Number of MCTS simulations to benchmark.
        device: Torch device.

    Returns:
        Tuple of ``(best_ucb_c, results_dict)``.
    """
    device = device or torch.device("cpu")
    candidates = [0.5, 1.0, 1.41, 2.0, 3.0]
    results: dict[str, object] = {}
    best_ucb = base_cfg.ucb_c
    best_reward = -float("inf")

    for ucb_c in candidates:
        cfg = MCTSConfig(
            n_simulations_base=min(n_simulations, 50),
            n_simulations_max=n_simulations,
            rollout_depth=base_cfg.rollout_depth,
            gamma=base_cfg.gamma,
            n_action_candidates=base_cfg.n_action_candidates,
            ucb_c=ucb_c,
        )
        planner = MCTSPlanner(cfg, rssm)

        episode_rewards: list[float] = []
        search_times: list[float] = []

        for _ in range(min(n_episodes, 100)):  # Use subset for speed
            h = torch.zeros(1, rssm._cfg.hidden_dim, device=device)
            z = torch.zeros(1, rssm._cfg.latent_dim, device=device)
            ep_reward = 0.0

            for _step in range(20):
                t0 = time.monotonic()
                action = planner.plan(h, z)
                search_times.append((time.monotonic() - t0) * 1000.0)

                h, z, reward = rssm.imagine_step(action, h, z)
                ep_reward += reward.item()

            episode_rewards.append(ep_reward)

        mean_reward = float(np.mean(episode_rewards))
        p50_ms = float(np.percentile(search_times, 50))
        p95_ms = float(np.percentile(search_times, 95))

        results[f"ucb_{ucb_c}"] = {
            "mean_reward": round(mean_reward, 4),
            "p50_ms": round(p50_ms, 2),
            "p95_ms": round(p95_ms, 2),
        }

        _log.info(
            "ucb_candidate",
            ucb_c=ucb_c,
            mean_reward=round(mean_reward, 4),
            p50_ms=round(p50_ms, 2),
            p95_ms=round(p95_ms, 2),
        )

        if mean_reward > best_reward and p50_ms < target_ms:
            best_reward = mean_reward
            best_ucb = ucb_c

    results["best_ucb_c"] = best_ucb
    return best_ucb, results


def run_warmstart(
    cfg: Settings,
    rssm_checkpoint: Path,
    data_path: Path,
    output_dir: Path | None = None,
) -> None:
    """Full Phase 2.2 pipeline.

    Args:
        cfg: Root settings.
        rssm_checkpoint: Path to pretrained RSSM checkpoint.
        data_path: Path to sequences.pt for computing latent stats.
        output_dir: Output directory for weights/config.
    """
    device = torch.device("cpu")
    output_dir = output_dir or Path(cfg.training.weights_dir) / "mcts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load RSSM
    rssm = RSSM(cfg.model).to(device)
    rssm.load_state_dict(torch.load(rssm_checkpoint, map_location=device, weights_only=True))
    rssm.eval()

    # Compute latent statistics
    dataset = RSSMSequenceDataset(data_path, seq_len=cfg.training.sequence_length)
    latent_mean, latent_std = compute_latent_statistics(rssm, dataset, device)

    # Warm-start policy
    policy = warmstart_policy(latent_mean, latent_std)
    policy.save(output_dir / "policy_init.npz")
    _log.info("policy_warmstarted", path=str(output_dir / "policy_init.npz"))

    # Tune UCB
    best_ucb, results = tune_ucb(rssm, cfg.mcts, device=device)
    with open(output_dir / "tuned_config.json", "w") as f:
        json.dump(results, f, indent=2)
    _log.info("ucb_tuned", best_ucb_c=best_ucb, output=str(output_dir / "tuned_config.json"))
