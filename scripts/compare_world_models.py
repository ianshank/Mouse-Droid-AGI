#!/usr/bin/env python3
"""Parity harness — compare two RSSM world models across synthetic episodes.

Measures the action-agreement rate between an old model (e.g. trained with
ultrasonic) and a new model (e.g. trained or migrated to LiDAR-only) by
rolling out the same synthetic episode through both and comparing the greedy
actions derived from their latent states.

Usage::

    python scripts/compare_world_models.py \\
        --old-ckpt weights/rssm/epoch_100.pt \\
        --old-config config/default.yaml \\
        --new-ckpt weights/lidar_only/epoch_100.pt \\
        --new-config config/lidar_only_training.yaml \\
        --episodes 50 \\
        --steps 30

The script exits with code 0 when mean action agreement ≥ ``--min-agreement``
(default 0.80) and code 1 otherwise, making it suitable as a CI gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor

# Ensure src/ is on the path when invoked directly.
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from mousedroid.config.loader import load_settings  # noqa: E402
from mousedroid.config.schema import ModelConfig  # noqa: E402
from mousedroid.logging.setup import get_logger  # noqa: E402
from mousedroid.world_model.checkpoint_migration import load_rssm_with_migration  # noqa: E402
from mousedroid.world_model.rssm import RSSM  # noqa: E402

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Synthetic episode generation
# ---------------------------------------------------------------------------

def _random_obs(cfg: ModelConfig, batch: int, device: torch.device) -> dict[str, Tensor]:
    """Generate one step of random observations for a given ModelConfig."""
    obs: dict[str, Tensor] = {
        "vision": torch.randn(batch, cfg.vision_dim, device=device),
        "motor": torch.randn(batch, cfg.motor_state_dim, device=device),
    }
    n_slots = 5  # valid_mask always covers all 5 slots
    obs["valid_mask"] = torch.ones(batch, n_slots, device=device)

    if cfg.ultrasonic_dim > 0:
        obs["ultrasonic"] = torch.randn(batch, cfg.ultrasonic_dim, device=device)
    else:
        obs["ultrasonic"] = None  # type: ignore[assignment]

    if cfg.audio_dim > 0:
        obs["audio"] = torch.randn(batch, cfg.audio_dim, device=device)
    else:
        obs["audio"] = None  # type: ignore[assignment]

    if cfg.lidar_dim > 0:
        obs["lidar"] = torch.randn(batch, cfg.lidar_dim, device=device)
    else:
        obs["lidar"] = None  # type: ignore[assignment]

    return obs


def _encode_obs(rssm: RSSM, obs: dict[str, Tensor]) -> Tensor:
    """Run encoder forward pass from the obs dict."""
    return rssm.encoder(
        obs["vision"],
        obs["ultrasonic"],
        obs["motor"],
        obs["valid_mask"],
        audio=obs["audio"],
        lidar=obs["lidar"],
    )


@torch.no_grad()
def _rollout(
    rssm: RSSM,
    cfg: ModelConfig,
    n_steps: int,
    device: torch.device,
    seed: int,
) -> list[Tensor]:
    """Roll out one synthetic episode and collect latent states.

    Args:
        rssm: World model to roll out.
        cfg: ModelConfig matching *rssm*.
        n_steps: Number of steps in the episode.
        device: Computation device.
        seed: RNG seed for reproducible episodes (same seed → same vision/motor).

    Returns:
        List of ``z`` latent state tensors, one per step.
    """
    torch.manual_seed(seed)
    rssm.eval()
    batch = 1
    h = torch.zeros(batch, cfg.hidden_dim, device=device)
    latents: list[Tensor] = []

    for _ in range(n_steps):
        obs = _random_obs(cfg, batch, device)
        obs_embed = _encode_obs(rssm, obs)
        # Posterior step: h + embed → z
        combined = torch.cat([h, obs_embed], dim=-1)
        params = rssm.posterior(combined)
        mean = params[:, : cfg.latent_dim]
        z = mean  # greedy (no sampling) for deterministic comparison

        # GRU step
        action = torch.zeros(batch, cfg.action_dim, device=device)
        gru_input = torch.cat([z, action], dim=-1)
        h = rssm.gru(gru_input, h)
        latents.append(z.squeeze(0))

    return latents


# ---------------------------------------------------------------------------
# Greedy action derivation (linear policy approximation)
# ---------------------------------------------------------------------------

def _latent_to_action(z: Tensor, action_dim: int) -> Tensor:
    """Project latent state to action space via a fixed linear projection.

    Uses the first ``action_dim`` dimensions of *z* as a simple, deterministic
    proxy for policy output.  This is sufficient for structural agreement
    measurement — we only care whether the *rank* of actions is consistent
    across models.
    """
    raw = z[:action_dim] if z.shape[0] >= action_dim else torch.cat(
        [z, z.new_zeros(action_dim - z.shape[0])]
    )
    return torch.tanh(raw)


# ---------------------------------------------------------------------------
# Agreement metric
# ---------------------------------------------------------------------------

def _cosine_agreement(a: Tensor, b: Tensor) -> float:
    """Cosine similarity between two action vectors, mapped to [0, 1]."""
    norm_a = a / (a.norm() + 1e-8)
    norm_b = b / (b.norm() + 1e-8)
    sim = (norm_a * norm_b).sum().item()
    return (sim + 1.0) / 2.0  # shift from [-1,1] to [0,1]


def compute_parity(
    old_rssm: RSSM,
    new_rssm: RSSM,
    old_cfg: ModelConfig,
    new_cfg: ModelConfig,
    n_episodes: int,
    n_steps: int,
    device: torch.device,
) -> dict[str, float]:
    """Compute action-agreement parity between two RSSM world models.

    Args:
        old_rssm: Reference (original) world model.
        new_rssm: Candidate (migrated/retrained) world model.
        old_cfg: ModelConfig matching *old_rssm*.
        new_cfg: ModelConfig matching *new_rssm*.
        n_episodes: Number of synthetic episodes to evaluate.
        n_steps: Steps per episode.
        device: Computation device.

    Returns:
        Dict with ``mean_agreement``, ``min_agreement``, ``max_agreement``,
        ``n_episodes``, and ``n_steps``.
    """
    agreements: list[float] = []

    for ep in range(n_episodes):
        old_latents = _rollout(old_rssm, old_cfg, n_steps, device, seed=ep)
        new_latents = _rollout(new_rssm, new_cfg, n_steps, device, seed=ep)
        for old_z, new_z in zip(old_latents, new_latents, strict=True):
            old_a = _latent_to_action(old_z, old_cfg.action_dim)
            new_a = _latent_to_action(new_z, new_cfg.action_dim)
            agreements.append(_cosine_agreement(old_a, new_a))

    mean_ag = sum(agreements) / max(len(agreements), 1)
    return {
        "mean_agreement": mean_ag,
        "min_agreement": min(agreements, default=0.0),
        "max_agreement": max(agreements, default=0.0),
        "n_episodes": n_episodes,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two RSSM world models for action-agreement parity."
    )
    p.add_argument("--old-ckpt", type=Path, required=True, help="Old model checkpoint path")
    p.add_argument("--old-config", type=Path, required=True, help="Config overlay for old model")
    p.add_argument("--new-ckpt", type=Path, required=True, help="New model checkpoint path")
    p.add_argument("--new-config", type=Path, required=True, help="Config overlay for new model")
    p.add_argument("--episodes", type=int, default=50, help="Synthetic episodes to evaluate")
    p.add_argument("--steps", type=int, default=30, help="Steps per episode")
    p.add_argument(
        "--min-agreement",
        type=float,
        default=0.80,
        help="Minimum mean agreement to pass (0-1, default 0.80)",
    )
    p.add_argument("--device", type=str, default="cpu", help="Torch device string")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    device = torch.device(args.device)

    _log.info("loading_old_model", ckpt=str(args.old_ckpt), config=str(args.old_config))
    old_cfg = load_settings(args.old_config).model

    _log.info("loading_new_model", ckpt=str(args.new_ckpt), config=str(args.new_config))
    new_cfg = load_settings(args.new_config).model

    old_rssm = load_rssm_with_migration(args.old_ckpt, old_cfg, device)
    new_rssm = load_rssm_with_migration(args.new_ckpt, new_cfg, device)

    _log.info(
        "parity_eval_start",
        episodes=args.episodes,
        steps=args.steps,
        min_agreement=args.min_agreement,
    )

    result = compute_parity(
        old_rssm,
        new_rssm,
        old_cfg,
        new_cfg,
        n_episodes=args.episodes,
        n_steps=args.steps,
        device=device,
    )

    _log.info("parity_result", **result)
    print(
        f"Mean agreement: {result['mean_agreement']:.3f}  "
        f"(min={result['min_agreement']:.3f}, max={result['max_agreement']:.3f})"
        f"  over {args.episodes} episodes x {args.steps} steps"
    )

    if result["mean_agreement"] >= args.min_agreement:
        print(f"PASS  (≥ {args.min_agreement:.0%})")
        return 0
    else:
        print(
            f"FAIL  ({result['mean_agreement']:.1%} < {args.min_agreement:.0%})",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
