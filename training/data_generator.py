"""Synthetic observation sequence generator using mock hardware.

Generates training data by running the MouseDroidOrchestrator in mock mode
and collecting (observation, action, reward) tuples.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import torch

from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator
from mousedroid.logging.setup import get_logger
from mousedroid.sensing.bundle import MouseDroidObservationBundle

_log = get_logger(__name__)


def _bundle_to_tensors(
    obs: MouseDroidObservationBundle,
) -> dict[str, torch.Tensor]:
    """Convert an observation bundle to a dict of tensors."""
    return {
        "vision": torch.as_tensor(obs.vision_features, dtype=torch.float32),
        "ultrasonic": torch.tensor([obs.distance_m], dtype=torch.float32),
        "motor_state": torch.as_tensor(obs.motor_state, dtype=torch.float32),
        "valid_mask": torch.as_tensor(obs.valid_mask, dtype=torch.float32),
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
        await orchestrator.start()

        transitions: list[dict[str, Any]] = []

        for _ in range(max_steps):
            obs = await orchestrator._sense()
            obs_tensors = _bundle_to_tensors(obs)

            # Random action for data collection
            action = torch.tanh(torch.randn(1, self._cfg.model.action_dim))

            transitions.append(
                {
                    "vision": obs_tensors["vision"],
                    "ultrasonic": obs_tensors["ultrasonic"],
                    "motor_state": obs_tensors["motor_state"],
                    "valid_mask": obs_tensors["valid_mask"],
                    "action": action.squeeze(0),
                }
            )

        await orchestrator.stop()
        return transitions

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

        all_episodes: list[list[dict[str, Any]]] = []
        for ep in range(n_episodes):
            transitions = asyncio.run(self._run_episode(max_steps))
            all_episodes.append(transitions)
            if (ep + 1) % 100 == 0:
                _log.info("episodes_generated", count=ep + 1, total=n_episodes)

        torch.save(all_episodes, output_dir / "sequences.pt")
        _log.info(
            "sequences_saved",
            n_episodes=n_episodes,
            max_steps=max_steps,
            path=str(output_dir),
        )
        return output_dir
