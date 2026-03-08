"""Phase 2.3a — Collect labelled intention annotations from navigation episodes.

Runs 500 episodes in mock mode and auto-labels each step with one of 8
intention classes using heuristic rules derived from action and sensor context.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator
from mousedroid.sensing.bundle import MouseDroidObservationBundle

_log = structlog.get_logger(__name__)

# 8 intention classes
INTENTION_LABELS = [
    "explore",           # 0
    "approach_target",   # 1
    "avoid_obstacle",    # 2
    "backtrack",         # 3
    "wait",              # 4
    "turn",              # 5
    "charge",            # 6
    "idle",              # 7
]


def label_intention(
    action: np.ndarray,
    obs: MouseDroidObservationBundle,
) -> int:
    """Assign an intention label based on heuristic rules.

    Args:
        action: Action vector ``[vx, vy, omega]``.
        obs: Observation bundle with sensor readings.

    Returns:
        Integer intention class index (0-7).
    """
    speed = float(np.linalg.norm(action[:2])) if len(action) >= 2 else abs(float(action[0]))
    omega = abs(float(action[2])) if len(action) >= 3 else 0.0
    distance = obs.distance_m
    battery = obs.motor_state[3] if len(obs.motor_state) > 3 else 12.0

    # Low battery → charge intention
    if battery < 10.8:
        return 6  # charge

    # Very close obstacle → avoid
    if distance < 0.25:
        return 2  # avoid_obstacle

    # Mostly stationary
    if speed < 0.05 and omega < 0.1:
        return 7  # idle

    # Waiting (low speed, no rotation)
    if speed < 0.1 and omega < 0.05:
        return 4  # wait

    # High rotation → turn
    if omega > 0.5:
        return 5  # turn

    # Moving backward → backtrack
    if len(action) >= 1 and float(action[0]) < -0.2:
        return 3  # backtrack

    # Moving forward with clear path → approach or explore
    if distance > 1.0 and speed > 0.2:
        return 1  # approach_target

    return 0  # explore


async def _collect_episode(
    cfg: Settings,
    max_steps: int,
) -> list[dict[str, Any]]:
    """Run one episode and collect annotated transitions."""
    orchestrator = build_orchestrator(cfg)
    await orchestrator.start()

    annotations: list[dict[str, Any]] = []
    rng = np.random.default_rng()

    for _ in range(max_steps):
        obs = await orchestrator._sense()

        # Random policy for diverse data collection
        action = np.tanh(rng.standard_normal(cfg.model.action_dim).astype(np.float32))

        intention = label_intention(action, obs)

        annotations.append({
            "observation": obs.vision_features.copy(),
            "action": action.copy(),
            "intention_label": intention,
            "distance_m": obs.distance_m,
            "motor_state": obs.motor_state.copy(),
        })

    await orchestrator.stop()
    return annotations


def collect_annotations(
    cfg: Settings,
    n_episodes: int = 500,
    max_steps: int = 50,
    output_path: Path | str | None = None,
) -> Path:
    """Collect labelled intention annotations from navigation episodes.

    Args:
        cfg: Root settings (must have ``mock_hardware=True``).
        n_episodes: Number of episodes to collect.
        max_steps: Steps per episode.
        output_path: Path to save annotations ``.npz``.

    Returns:
        Path to saved annotations file.
    """
    if not cfg.mock_hardware:
        msg = "Annotation collection requires mock_hardware=True"
        raise ValueError(msg)

    output_path = Path(output_path) if output_path else Path(cfg.training.data_dir) / "bdi_annotations.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_observations: list[np.ndarray] = []
    all_intentions: list[int] = []

    for ep in range(n_episodes):
        annotations = asyncio.get_event_loop().run_until_complete(
            _collect_episode(cfg, max_steps),
        )
        for ann in annotations:
            all_observations.append(ann["observation"])
            all_intentions.append(ann["intention_label"])

        if (ep + 1) % 100 == 0:
            _log.info("annotation_episodes", count=ep + 1, total=n_episodes)

    observations = np.stack(all_observations)
    intentions = np.array(all_intentions, dtype=np.int64)

    np.savez(
        output_path,
        observations=observations,
        intentions=intentions,
    )
    _log.info(
        "annotations_saved",
        path=str(output_path),
        n_samples=len(intentions),
        class_distribution={
            INTENTION_LABELS[i]: int(np.sum(intentions == i))
            for i in range(len(INTENTION_LABELS))
        },
    )
    return output_path
