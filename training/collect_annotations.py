"""Phase 2.3a — Collect labelled intention annotations from navigation episodes.

Runs 500 episodes in mock mode and auto-labels each step with one of 10
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

# 10 intention classes (expanded for Three Laws)
INTENTION_LABELS = [
    "explore",  # 0
    "approach_target",  # 1
    "avoid_obstacle",  # 2
    "backtrack",  # 3
    "wait",  # 4
    "turn",  # 5
    "charge",  # 6
    "idle",  # 7
    "protect_human",  # 8  — Law 1: human proximity triggered
    "obey_command",  # 9  — Law 2: following commanded action
]


def label_intention(
    action: np.ndarray,
    obs: MouseDroidObservationBundle,
    *,
    human_detected: bool = False,
    human_dist_m: float = float("inf"),
    commanded_action: np.ndarray | None = None,
    human_safety_radius_m: float = 0.5,
    battery_warn_v: float = 10.8,
    obstacle_clearance_m: float = 0.25,
) -> int:
    """Assign an intention label based on heuristic rules.

    Args:
        action: Action vector ``[vx, vy, omega]``.
        obs: Observation bundle with sensor readings.
        human_detected: Whether a human is detected nearby.
        human_dist_m: Distance to nearest human in metres.
        commanded_action: Externally commanded action, if any.
        human_safety_radius_m: Law 1 human safety distance threshold.
        battery_warn_v: Battery voltage threshold for charge intention.
        obstacle_clearance_m: Obstacle distance threshold for avoidance.

    Returns:
        Integer intention class index (0-9).
    """
    speed = float(np.linalg.norm(action[:2])) if len(action) >= 2 else abs(float(action[0]))
    omega = abs(float(action[2])) if len(action) >= 3 else 0.0
    distance = obs.distance_m
    battery = obs.motor_state[3] if len(obs.motor_state) > 3 else 12.0

    # Law 1: Human proximity → protect_human
    if human_detected and human_dist_m < human_safety_radius_m:
        return 8  # protect_human

    # Law 2: Commanded action → obey_command
    if commanded_action is not None:
        return 9  # obey_command

    # Low battery → charge intention
    if battery < battery_warn_v:
        return 6  # charge

    # Very close obstacle → avoid
    if distance < obstacle_clearance_m:
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


def _build_state_observation(
    vision_features: np.ndarray,
    action: np.ndarray,
    distance_m: float,
    motor_state: np.ndarray,
    human_detected: bool,
    commanded: bool,
) -> np.ndarray:
    """Build a 256d observation embedding label-relevant state.

    At runtime the BDI receives the RSSM latent which encodes the full
    robot state.  In mock mode we simulate this by placing the
    intention-determining signals (action, distance, motor, flags) into
    the first dimensions so the encoder can learn from them.  The
    remaining dimensions are filled from the (noisy) vision features.
    """
    # Structured prefix: action(3) + distance(1) + motor(4) + flags(2) = 10
    prefix = np.array(
        [
            *action[:3],
            distance_m,
            *motor_state[:4],
            float(human_detected),
            float(commanded),
        ],
        dtype=np.float32,
    )
    n_prefix = len(prefix)
    obs = vision_features.copy()
    obs[:n_prefix] = prefix
    return obs


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

        # Inject edge conditions to ensure all 10 intention classes are represented
        human_detected = rng.random() < 0.10
        human_dist_m = float(rng.uniform(0.1, 0.4)) if human_detected else float("inf")
        commanded_action = action.copy() if rng.random() < 0.10 else None

        intention = label_intention(
            action,
            obs,
            human_detected=human_detected,
            human_dist_m=human_dist_m,
            commanded_action=commanded_action,
        )

        # Build observation with label-relevant state in the first dims
        state_obs = _build_state_observation(
            obs.vision_features,
            action,
            obs.distance_m,
            obs.motor_state,
            human_detected,
            commanded_action is not None,
        )

        annotations.append(
            {
                "observation": state_obs,
                "action": action.copy(),
                "intention_label": intention,
                "distance_m": obs.distance_m,
                "motor_state": obs.motor_state.copy(),
            }
        )

    await orchestrator.stop()
    return annotations


def collect_annotations(
    cfg: Settings,
    n_episodes: int = 500,
    max_steps: int = 50,
    output_path: Path | str | None = None,
    balance_dataset: bool = False,
) -> Path:
    """Collect labelled intention annotations from navigation episodes.

    Args:
        cfg: Root settings (must have ``mock_hardware=True``).
        n_episodes: Number of episodes to collect.
        max_steps: Steps per episode.
        output_path: Path to save annotations ``.npz``.
        balance_dataset: Oversample minority classes to within 20% of majority.

    Returns:
        Path to saved annotations file.
    """
    if not cfg.mock_hardware:
        msg = "Annotation collection requires mock_hardware=True"
        raise ValueError(msg)

    if output_path:
        output_path = Path(output_path)
    else:
        output_path = Path(cfg.training.data_dir) / "bdi_annotations.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_observations: list[np.ndarray] = []
    all_intentions: list[int] = []

    for ep in range(n_episodes):
        annotations = asyncio.run(_collect_episode(cfg, max_steps))
        for ann in annotations:
            all_observations.append(ann["observation"])
            all_intentions.append(ann["intention_label"])

        if (ep + 1) % 100 == 0:
            _log.info("annotation_episodes", count=ep + 1, total=n_episodes)

    observations = np.stack(all_observations)
    intentions = np.array(all_intentions, dtype=np.int64)

    # Always audit and log class distribution
    class_counts = audit_class_balance(intentions)

    # Optionally balance classes via oversampling
    if balance_dataset:
        observations, intentions = balance_classes(observations, intentions)
        _log.info(
            "class_balance_applied",
            post_balance_counts={
                INTENTION_LABELS[i]: int(np.sum(intentions == i))
                for i in range(len(INTENTION_LABELS))
            },
        )

    np.savez(
        output_path,
        observations=observations,
        intentions=intentions,
    )
    _log.info(
        "annotations_saved",
        path=str(output_path),
        n_samples=len(intentions),
        class_distribution=class_counts,
    )
    return output_path


def audit_class_balance(
    intentions: np.ndarray,
) -> dict[str, int]:
    """Log per-class sample counts and imbalance ratio.

    Args:
        intentions: 1-D array of intention class indices.

    Returns:
        Dictionary mapping class name → count.
    """
    counts: dict[str, int] = {}
    for i, label in enumerate(INTENTION_LABELS):
        counts[label] = int(np.sum(intentions == i))

    values = [c for c in counts.values() if c > 0]
    if values:
        imbalance_ratio = max(values) / max(min(values), 1)
    else:
        imbalance_ratio = 0.0

    _log.info(
        "class_balance_audit",
        counts=counts,
        n_classes_present=len(values),
        imbalance_ratio=round(imbalance_ratio, 2),
    )
    return counts


def balance_classes(
    observations: np.ndarray,
    intentions: np.ndarray,
    *,
    max_ratio: float = 1.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes to within ``max_ratio`` of the majority.

    Args:
        observations: Array of shape ``(n, obs_dim)``.
        intentions: 1-D array of intention labels.
        max_ratio: Maximum allowed ratio between majority and each minority class.
            Defaults to 1.2 (within 20% of majority).
        seed: RNG seed for reproducible oversampling.

    Returns:
        Tuple of (balanced_observations, balanced_intentions).
    """
    rng = np.random.default_rng(seed)
    n_classes = len(INTENTION_LABELS)
    class_counts = np.array(
        [int(np.sum(intentions == i)) for i in range(n_classes)]
    )
    majority_count = int(class_counts.max())
    target_count = int(majority_count / max_ratio)

    balanced_obs_parts: list[np.ndarray] = []
    balanced_int_parts: list[np.ndarray] = []

    for cls_idx in range(n_classes):
        mask = intentions == cls_idx
        cls_obs = observations[mask]
        cls_count = len(cls_obs)

        if cls_count == 0:
            continue

        balanced_obs_parts.append(cls_obs)
        balanced_int_parts.append(intentions[mask])

        # Oversample if below target
        if cls_count < target_count:
            n_extra = target_count - cls_count
            extra_indices = rng.choice(cls_count, size=n_extra, replace=True)
            balanced_obs_parts.append(cls_obs[extra_indices])
            balanced_int_parts.append(np.full(n_extra, cls_idx, dtype=np.int64))

    balanced_obs = np.concatenate(balanced_obs_parts, axis=0)
    balanced_int = np.concatenate(balanced_int_parts, axis=0)

    # Shuffle
    perm = rng.permutation(len(balanced_obs))
    return balanced_obs[perm], balanced_int[perm]

