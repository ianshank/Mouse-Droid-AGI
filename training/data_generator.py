"""Synthetic observation sequence generator using mock hardware.

Generates training data by running the MouseDroidOrchestrator in mock mode
and collecting (observation, action, reward) tuples.

Phase 1 — Domain Randomization: when ``cfg.domain_randomization.enabled`` is
true, per-episode :class:`EpisodeParams` are sampled from a seeded
:class:`numpy.random.Generator` and applied to the observation tensors
(feature noise on vision; additive noise + dropout on the ultrasonic). When
disabled, the generator falls back to the legacy ``torch.randn`` action path
so existing artifacts and golden hashes remain byte-identical.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mousedroid.config.schema import ModelConfig, Settings
from mousedroid.factory import build_orchestrator
from mousedroid.logging.setup import get_logger
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.training.domain_randomization import (
    DomainRandomizer,
    EpisodeParams,
    apply_feature_noise,
    apply_range_sensor_randomization,
)

_log = get_logger(__name__)


def _bundle_to_tensors(
    obs: MouseDroidObservationBundle,
    model_cfg: ModelConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Convert an observation bundle to a dict of tensors."""
    lidar_dim = model_cfg.lidar_dim if model_cfg is not None else 0
    if lidar_dim > 0:
        lidar_source = obs.lidar_features
        lidar = torch.zeros(lidar_dim, dtype=torch.float32)
        if lidar_source is not None:
            lidar_tensor = torch.as_tensor(lidar_source, dtype=torch.float32)
            lidar[: min(lidar_dim, lidar_tensor.shape[0])] = lidar_tensor[:lidar_dim]
    else:
        lidar = torch.zeros(0, dtype=torch.float32)

    ultrasonic = (
        torch.tensor([obs.distance_m], dtype=torch.float32)
        if model_cfg is None or model_cfg.ultrasonic_dim > 0
        else torch.zeros(0, dtype=torch.float32)
    )

    return {
        "vision": torch.as_tensor(obs.vision_features, dtype=torch.float32),
        "ultrasonic": ultrasonic,
        "motor_state": torch.as_tensor(obs.motor_state, dtype=torch.float32),
        "valid_mask": torch.as_tensor(obs.valid_mask, dtype=torch.float32),
        "lidar": lidar,
    }


def _apply_episode_randomization(
    obs_tensors: dict[str, torch.Tensor],
    ep_params: EpisodeParams,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    """Apply :class:`EpisodeParams` to an already-tensorised observation dict.

    Empty ``ep_params`` returns ``obs_tensors`` unchanged so the disabled-DR
    code path is byte-identical to the pre-feature behaviour.

    Args:
        obs_tensors: Output of :func:`_bundle_to_tensors`.
        ep_params: Per-episode randomization bundle.
        rng: Per-step RNG.

    Returns:
        New dict with feature-noise vision and noisy/dropped ultrasonic
        tensors. Other modalities are forwarded unchanged.
    """
    if ep_params.is_empty:
        return obs_tensors

    randomized: dict[str, torch.Tensor] = dict(obs_tensors)

    if ep_params.feature:
        vision_np = obs_tensors["vision"].detach().cpu().numpy()
        noisy = apply_feature_noise(vision_np, ep_params.feature, rng)
        randomized["vision"] = torch.from_numpy(np.ascontiguousarray(noisy)).to(torch.float32)

    if ep_params.range_sensor and obs_tensors["ultrasonic"].numel() > 0:
        nominal = float(obs_tensors["ultrasonic"][0].item())
        sampled = apply_range_sensor_randomization(nominal, ep_params.range_sensor, rng)
        if not np.isnan(sampled):
            randomized["ultrasonic"] = torch.tensor([sampled], dtype=torch.float32)

    return randomized


class SyntheticSequenceGenerator:
    """Generate synthetic observation sequences via mock orchestrator.

    Args:
        cfg: Root settings (must have ``mock_hardware=True``).
        seed: Optional integer seed for the per-run RNG. ``None`` defers to
            ``numpy.random.default_rng()`` (non-deterministic). Only consumed
            when ``cfg.domain_randomization.enabled`` is true; otherwise the
            legacy ``torch.randn`` action path is preserved verbatim.
    """

    def __init__(self, cfg: Settings, *, seed: int | None = None) -> None:
        if not cfg.mock_hardware:
            msg = "SyntheticSequenceGenerator requires mock_hardware=True"
            raise ValueError(msg)
        self._cfg = cfg
        self._seed = seed
        self._randomizer = DomainRandomizer(cfg.domain_randomization)

    async def _run_episode(
        self,
        max_steps: int,
        ep_params: EpisodeParams,
        rng: np.random.Generator | None,
    ) -> list[dict[str, Any]]:
        """Run a single episode and collect transitions.

        Args:
            max_steps: Maximum steps per episode.
            ep_params: Per-episode randomization bundle (empty disables DR).
            rng: Numpy RNG used for randomization + action sampling. ``None``
                preserves the legacy ``torch.randn`` action path.

        Returns:
            List of transition dicts with keys: ``vision``, ``ultrasonic``,
            ``motor_state``, ``valid_mask``, ``lidar``, ``action``.
        """
        orchestrator = build_orchestrator(self._cfg)
        await orchestrator.start()  # type: ignore[attr-defined]

        transitions: list[dict[str, Any]] = []
        action_dim = self._cfg.model.action_dim

        for _ in range(max_steps):
            # Use the sensor manager to read observations
            obs = await orchestrator._sensor_manager.read_all()  # type: ignore[attr-defined]
            obs_tensors = _bundle_to_tensors(obs, self._cfg.model)

            if rng is not None and not ep_params.is_empty:
                obs_tensors = _apply_episode_randomization(obs_tensors, ep_params, rng)

            if rng is None:
                # Legacy path — preserves byte-identical output when DR off.
                action = torch.tanh(torch.randn(1, action_dim))
            else:
                action_np = np.tanh(rng.standard_normal(action_dim, dtype=np.float32))
                action = torch.from_numpy(action_np).unsqueeze(0)

            transitions.append(
                {
                    "vision": obs_tensors["vision"],
                    "ultrasonic": obs_tensors["ultrasonic"],
                    "motor_state": obs_tensors["motor_state"],
                    "valid_mask": obs_tensors["valid_mask"],
                    "lidar": obs_tensors["lidar"],
                    "action": action.squeeze(0),
                }
            )

        await orchestrator.stop()  # type: ignore[attr-defined]
        return transitions

    async def _generate_sequences_async(
        self,
        n_episodes: int,
        max_steps: int,
    ) -> list[list[dict[str, Any]]]:
        """Generate all episodes within a single event loop."""
        all_episodes: list[list[dict[str, Any]]] = []
        log_every = self._cfg.training.generation.log_every_n_episodes

        if self._randomizer.enabled:
            master_rng = np.random.default_rng(self._seed)
            _log.info(
                "dr_enabled_for_generation",
                seed=self._seed,
                n_episodes=n_episodes,
            )
        else:
            master_rng = None
            if self._seed is not None:
                _log.info("dr_disabled_seed_ignored", seed=self._seed)

        for ep in range(n_episodes):
            if master_rng is not None:
                episode_rng = np.random.default_rng(master_rng.integers(0, np.iinfo(np.int64).max))
                ep_params = self._randomizer.sample(episode_rng)
            else:
                episode_rng = None
                ep_params = EpisodeParams()

            transitions = await self._run_episode(max_steps, ep_params, episode_rng)
            all_episodes.append(transitions)
            if (ep + 1) % log_every == 0 or ep + 1 == n_episodes:
                _log.info("episodes_generated", count=ep + 1, total=n_episodes)

        return all_episodes

    def generate_sequences(
        self,
        n_episodes: int,
        max_steps: int,
        output_dir: Path | str,
    ) -> Path:
        """Generate and save synthetic observation sequences.

        Args:
            n_episodes: Number of episodes to generate.
            max_steps: Maximum steps per episode.
            output_dir: Directory to save ``.pt`` files.

        Returns:
            Path to the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_episodes = asyncio.run(self._generate_sequences_async(n_episodes, max_steps))

        torch.save(all_episodes, output_dir / "sequences.pt")
        _log.info(
            "sequences_saved",
            n_episodes=n_episodes,
            max_steps=max_steps,
            path=str(output_dir),
            domain_randomization_enabled=self._randomizer.enabled,
        )
        return output_dir
