"""Synthetic observation sequence generator using mock hardware.

Generates training data by running the MouseDroidOrchestrator in mock mode
and collecting (observation, action, reward) tuples.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import torch

from mousedroid.config.schema import ModelConfig, Settings
from mousedroid.factory import build_orchestrator
from mousedroid.logging.setup import get_logger
from mousedroid.sensing.bundle import MouseDroidObservationBundle

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


class SyntheticSequenceGenerator:
    """Generate synthetic observation sequences via mock orchestrator.

    Args:
        cfg: Root settings (must have ``mock_hardware=True``).
    """

    def __init__(self, cfg: Settings) -> None:
        if not cfg.mock_hardware:
            msg = "SyntheticSequenceGenerator requires mock_hardware=True"
            raise ValueError(msg)
        self._cfg = cfg

    async def _run_episode(
        self,
        max_steps: int,
    ) -> list[dict[str, Any]]:
        """Run a single episode and collect transitions.

        Args:
            max_steps: Maximum steps per episode.

        Returns:
            List of transition dicts with keys: obs, action, reward.
        """
        orchestrator = build_orchestrator(self._cfg)
        await orchestrator.start()  # type: ignore[attr-defined]

        transitions: list[dict[str, Any]] = []

        for _ in range(max_steps):
            # Use the sensor manager to read observations
            obs = await orchestrator._sensor_manager.read_all()  # type: ignore[attr-defined]
            obs_tensors = _bundle_to_tensors(obs, self._cfg.model)

            # Random action for data collection
            action = torch.tanh(torch.randn(1, self._cfg.model.action_dim))

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

        for ep in range(n_episodes):
            transitions = await self._run_episode(max_steps)
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
        )
        return output_dir
