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
from numpy.typing import NDArray

from mousedroid.config.schema import Settings, TrainingAnnotationConfig
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

# Columns of ``intention_features`` — must stay aligned with :func:`label_intention`.
INTENTION_FEATURE_NAMES: tuple[str, ...] = (
    "vx",
    "vy",
    "omega",
    "speed",
    "abs_omega",
    "distance_m",
    "battery_v",
    "human_detected",
    "human_dist_m",
    "commanded",
)
INTENTION_FEATURE_DIM: int = len(INTENTION_FEATURE_NAMES)
_HUMAN_DIST_ABSENT: float = 10.0


def intention_feature_vector(
    action: NDArray[Any],
    obs: MouseDroidObservationBundle,
    *,
    human_detected: bool = False,
    human_dist_m: float = float("inf"),
    commanded_action: NDArray[Any] | None = None,
) -> NDArray[np.float32]:
    """Pack the exact scalars :func:`label_intention` reads into a feature row.

    Vision is omitted on purpose: the 10-class heuristic is a function of
    action, range, battery, and Law 1/2 flags, not of ``obs.vision_features``.
    """
    vx = float(action[0]) if len(action) >= 1 else 0.0
    vy = float(action[1]) if len(action) >= 2 else 0.0
    omega = float(action[2]) if len(action) >= 3 else 0.0
    speed = float(np.linalg.norm(action[:2])) if len(action) >= 2 else abs(vx)
    battery = float(obs.motor_state[3]) if len(obs.motor_state) > 3 else 12.0
    dist = (
        min(float(human_dist_m), _HUMAN_DIST_ABSENT)
        if np.isfinite(human_dist_m)
        else _HUMAN_DIST_ABSENT
    )
    features = np.array(
        [
            vx,
            vy,
            omega,
            speed,
            abs(omega),
            float(obs.distance_m),
            battery,
            1.0 if human_detected else 0.0,
            dist,
            1.0 if commanded_action is not None else 0.0,
        ],
        dtype=np.float32,
    )
    return features


def _sample_label_context(
    rng: np.random.Generator,
    action: NDArray[Any],
) -> tuple[MouseDroidObservationBundle, bool, float, NDArray[Any] | None]:
    """Sample causal fields so all 10 labels can appear under mock sensors."""
    distance_m = float(rng.uniform(0.05, 3.5))
    battery_v = float(rng.uniform(9.0, 12.6))
    human_detected = bool(rng.random() < 0.08)
    human_dist_m = float(rng.uniform(0.05, 1.5)) if human_detected else float("inf")
    commanded_action = action.copy() if rng.random() < 0.08 else None
    obs = MouseDroidObservationBundle(
        _distance_m=distance_m,
        _motor_state=np.array([0.0, 0.0, 0.0, battery_v], dtype=np.float32),
    )
    return obs, human_detected, human_dist_m, commanded_action


def label_intention(
    action: NDArray[Any],
    obs: MouseDroidObservationBundle,
    *,
    human_detected: bool = False,
    human_dist_m: float = float("inf"),
    commanded_action: NDArray[Any] | None = None,
    human_safety_radius_m: float = 0.5,
    battery_warn_v: float = 10.8,
    obstacle_clearance_m: float = 0.25,
    idle_speed_threshold: float = 0.05,
    idle_omega_threshold: float = 0.1,
    wait_speed_threshold: float = 0.1,
    wait_omega_threshold: float = 0.05,
    turn_omega_threshold: float = 0.5,
    backtrack_speed_threshold: float = -0.2,
    approach_clear_distance_m: float = 1.0,
    approach_speed_threshold: float = 0.2,
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
        idle_speed_threshold: Planar speed threshold for the idle label.
        idle_omega_threshold: Angular speed threshold for the idle label.
        wait_speed_threshold: Planar speed threshold for the wait label.
        wait_omega_threshold: Angular speed threshold for the wait label.
        turn_omega_threshold: Angular speed threshold for the turn label.
        backtrack_speed_threshold: Forward velocity threshold for the backtrack label.
        approach_clear_distance_m: Distance threshold for the approach_target label.
        approach_speed_threshold: Planar speed threshold for the approach_target label.

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
    if speed < idle_speed_threshold and omega < idle_omega_threshold:
        return 7  # idle

    # Waiting (low speed, no rotation)
    if speed < wait_speed_threshold and omega < wait_omega_threshold:
        return 4  # wait

    # High rotation → turn
    if omega > turn_omega_threshold:
        return 5  # turn

    # Moving backward → backtrack
    if len(action) >= 1 and float(action[0]) < backtrack_speed_threshold:
        return 3  # backtrack

    # Moving forward with clear path → approach or explore
    if distance > approach_clear_distance_m and speed > approach_speed_threshold:
        return 1  # approach_target

    return 0  # explore


def _label_intention_from_config(
    action: NDArray[Any],
    obs: MouseDroidObservationBundle,
    annotation_cfg: TrainingAnnotationConfig,
    *,
    human_detected: bool = False,
    human_dist_m: float = float("inf"),
    commanded_action: NDArray[Any] | None = None,
) -> int:
    """Apply annotation heuristics using config-backed thresholds."""
    return label_intention(
        action,
        obs,
        human_detected=human_detected,
        human_dist_m=human_dist_m,
        commanded_action=commanded_action,
        human_safety_radius_m=annotation_cfg.human_safety_radius_m,
        battery_warn_v=annotation_cfg.battery_warn_v,
        obstacle_clearance_m=annotation_cfg.obstacle_clearance_m,
        idle_speed_threshold=annotation_cfg.idle_speed_threshold,
        idle_omega_threshold=annotation_cfg.idle_omega_threshold,
        wait_speed_threshold=annotation_cfg.wait_speed_threshold,
        wait_omega_threshold=annotation_cfg.wait_omega_threshold,
        turn_omega_threshold=annotation_cfg.turn_omega_threshold,
        backtrack_speed_threshold=annotation_cfg.backtrack_speed_threshold,
        approach_clear_distance_m=annotation_cfg.approach_clear_distance_m,
        approach_speed_threshold=annotation_cfg.approach_speed_threshold,
    )


async def _collect_episode(
    cfg: Settings,
    max_steps: int,
    annotation_cfg: TrainingAnnotationConfig,
) -> list[dict[str, Any]]:
    """Run one episode and collect annotated transitions."""
    orchestrator = build_orchestrator(cfg)
    await orchestrator.start()  # type: ignore[attr-defined]

    annotations: list[dict[str, Any]] = []
    rng = np.random.default_rng()

    for _ in range(max_steps):
        # Use the sensor manager to read observations
        obs = await orchestrator._sensor_manager.read_all()  # type: ignore[attr-defined]

        # Random policy for diverse data collection
        action = np.tanh(rng.standard_normal(cfg.model.action_dim).astype(np.float32))
        label_obs, human_detected, human_dist_m, commanded_action = _sample_label_context(
            rng, action
        )
        intention = _label_intention_from_config(
            action,
            label_obs,
            annotation_cfg,
            human_detected=human_detected,
            human_dist_m=human_dist_m,
            commanded_action=commanded_action,
        )
        features = intention_feature_vector(
            action,
            label_obs,
            human_detected=human_detected,
            human_dist_m=human_dist_m,
            commanded_action=commanded_action,
        )

        annotations.append(
            {
                "observation": obs.vision_features.copy(),
                "intention_features": features,
                "action": action.copy(),
                "intention_label": intention,
                "distance_m": label_obs.distance_m,
                "motor_state": label_obs.motor_state.copy(),
            }
        )

    await orchestrator.stop()  # type: ignore[attr-defined]
    return annotations


async def _collect_annotations_async(
    cfg: Settings,
    n_episodes: int,
    max_steps: int,
    annotation_cfg: TrainingAnnotationConfig,
) -> tuple[list[NDArray[Any]], list[NDArray[Any]], list[int]]:
    """Collect all annotations within a single event loop."""
    all_observations: list[NDArray[Any]] = []
    all_features: list[NDArray[Any]] = []
    all_intentions: list[int] = []

    for ep in range(n_episodes):
        annotations = await _collect_episode(cfg, max_steps, annotation_cfg)
        for ann in annotations:
            all_observations.append(ann["observation"])
            all_features.append(ann["intention_features"])
            all_intentions.append(ann["intention_label"])

        if (ep + 1) % annotation_cfg.log_every_n_episodes == 0 or ep + 1 == n_episodes:
            _log.info("annotation_episodes", count=ep + 1, total=n_episodes)

    return all_observations, all_features, all_intentions


def collect_annotations(
    cfg: Settings,
    n_episodes: int | None = None,
    max_steps: int | None = None,
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

    annotation_cfg = cfg.training.annotation
    n_episodes = n_episodes if n_episodes is not None else annotation_cfg.n_episodes
    max_steps = max_steps if max_steps is not None else annotation_cfg.max_steps

    if output_path:
        output_path = Path(output_path)
    else:
        output_path = Path(cfg.training.data_dir) / "bdi_annotations.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _log.info(
        "annotation_collection_start",
        n_episodes=n_episodes,
        max_steps=max_steps,
        output_path=str(output_path),
    )

    all_observations, all_features, all_intentions = asyncio.run(
        _collect_annotations_async(cfg, n_episodes, max_steps, annotation_cfg)
    )

    observations = np.stack(all_observations)
    intention_features = np.stack(all_features)
    intentions = np.array(all_intentions, dtype=np.int64)

    np.savez(
        output_path,
        observations=observations,
        intention_features=intention_features,
        intentions=intentions,
    )
    _log.info(
        "annotations_saved",
        path=str(output_path),
        n_samples=len(intentions),
        class_distribution={
            INTENTION_LABELS[i]: int(np.sum(intentions == i)) for i in range(len(INTENTION_LABELS))
        },
    )
    return output_path
