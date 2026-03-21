"""Constitutional RL — safety-constrained policy with curiosity aggregation.

All computation is numpy-only.  The constitutional checker clips actions that
violate hard safety principles before they reach actuators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from mousedroid.common.math.numpy_ops import layer_norm, relu
from mousedroid.constants import (
    DEFAULT_ACTION_DIM,
    DEFAULT_BELIEF_DIM,
    DEFAULT_POLICY_HIDDEN_DIM,
    POLICY_MLP_SEED,
    VALUE_MLP_SEED,
    WEIGHT_INIT_SCALE,
)
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.safety.three_laws import RoboticsLawChecker

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_SPEED_CEILING_DEFAULT: float = 0.5
"""Default maximum speed in m/s."""

_BATTERY_MIN_DEFAULT: float = 10.8
"""Default minimum battery voltage (V)."""

_MCTS_MIN_SIMS_DEFAULT: int = 16
"""Default minimum MCTS simulations before trusting the tree."""

_OBSTACLE_CLEARANCE_DEFAULT: float = 0.25
"""Default minimum obstacle clearance in metres."""

_CURIOSITY_CHANNELS: tuple[str, ...] = (
    "social",
    "epistemic",
    "perceptual",
    "metacognitive",
)
"""Named curiosity channels for aggregation."""

_POLICY_HIDDEN_DIM: int = DEFAULT_POLICY_HIDDEN_DIM
"""Hidden layer dimensionality for PolicyMLP and ValueMLP networks."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstitutionalRLConfig:
    """Safety thresholds for constitutional checking.

    Attributes:
        speed_ceiling_mps: Maximum allowed speed (m/s).
        battery_min_v: Minimum battery voltage before halting.
        mcts_min_sims: Minimum MCTS simulations required.
        obstacle_clearance_floor_m: Minimum obstacle clearance (m).
    """

    speed_ceiling_mps: float = _SPEED_CEILING_DEFAULT
    battery_min_v: float = _BATTERY_MIN_DEFAULT
    mcts_min_sims: int = _MCTS_MIN_SIMS_DEFAULT
    obstacle_clearance_floor_m: float = _OBSTACLE_CLEARANCE_DEFAULT


# ---------------------------------------------------------------------------
# Constitutional Checker
# ---------------------------------------------------------------------------


class ConstitutionalChecker:
    """Check and clip actions against constitutional safety principles.

    Optionally integrates :class:`~mousedroid.safety.three_laws.RoboticsLawChecker`
    to enforce the Three Laws of Robotics *before* constitutional checks.

    Args:
        config: Safety thresholds.
        law_checker: Optional Three Laws checker (runs first).
    """

    def __init__(
        self,
        config: ConstitutionalRLConfig | None = None,
        law_checker: RoboticsLawChecker | None = None,
    ) -> None:
        self._cfg = config or ConstitutionalRLConfig()
        self._law_checker = law_checker

    def check(
        self,
        action: NDArray[np.floating[Any]],
        context: dict[str, Any],
    ) -> tuple[NDArray[np.floating[Any]], list[str]]:
        """Validate *action* against constitutional principles.

        If a :class:`~mousedroid.safety.three_laws.RoboticsLawChecker` is
        configured, it runs **first** and its violations are prepended.

        Args:
            action: Raw action vector (at minimum ``[speed, steering]``).
            context: Environment context with optional keys
                ``battery_v``, ``obstacle_dist_m``, ``mcts_sims``,
                ``human_detected``, ``human_dist_m``, etc.

        Returns:
            Tuple of ``(safe_action, violations)`` where *violations* is
            an empty list when no principles are breached.
        """
        # --- Three Laws (highest priority) ---
        law_violation_strs: list[str] = []
        if self._law_checker is not None:
            safe, law_violations = self._law_checker.check(action, context)
            law_violation_strs = [f"[Law {v.law.value}] {v.description}" for v in law_violations]
        else:
            safe = action.copy().astype(np.float64)

        violations: list[str] = list(law_violation_strs)

        # --- Speed ceiling ---
        if safe.size > 0 and float(np.abs(safe[0])) > self._cfg.speed_ceiling_mps:
            violations.append(
                f"speed {float(safe[0]):.2f} exceeds ceiling {self._cfg.speed_ceiling_mps:.2f} m/s"
            )
            safe[0] = np.clip(safe[0], -self._cfg.speed_ceiling_mps, self._cfg.speed_ceiling_mps)

        # --- Battery floor ---
        battery_v: float = float(context.get("battery_v", self._cfg.battery_min_v + 1.0))
        if battery_v < self._cfg.battery_min_v:
            violations.append(
                f"battery {battery_v:.2f}V below minimum {self._cfg.battery_min_v:.2f}V"
            )
            safe[:] = 0.0  # full stop

        # --- Obstacle clearance ---
        obstacle_dist: float = float(context.get("obstacle_dist_m", float("inf")))
        if obstacle_dist < self._cfg.obstacle_clearance_floor_m:
            violations.append(
                f"obstacle at {obstacle_dist:.2f}m < clearance "
                f"{self._cfg.obstacle_clearance_floor_m:.2f}m"
            )
            if safe.size > 0:
                safe[0] = min(float(safe[0]), 0.0)  # no forward motion

        # --- MCTS simulation count ---
        mcts_sims: int = int(context.get("mcts_sims", self._cfg.mcts_min_sims))
        if mcts_sims < self._cfg.mcts_min_sims:
            violations.append(f"mcts_sims {mcts_sims} < minimum {self._cfg.mcts_min_sims}")

        if violations:
            _log.warning("constitutional_violations", violations=violations)

        return safe, violations


# ---------------------------------------------------------------------------
# Flow Calculator
# ---------------------------------------------------------------------------


class FlowCalculator:
    """Compute a flow score based on channel width and observed value."""

    def compute(self, channel_width: float, value: float) -> float:
        """Compute flow score.

        When *channel_width* is zero, returns ``1.0`` if *value* is also
        exactly zero, else ``0.0``.

        Args:
            channel_width: Width of the flow channel.
            value: Observed value.

        Returns:
            Flow score in ``[0, 1]``.
        """
        if channel_width == 0.0:
            return 1.0 if value == 0.0 else 0.0
        raw = 1.0 - abs(value) / abs(channel_width)
        return float(np.clip(raw, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Curiosity Aggregator
# ---------------------------------------------------------------------------


class CuriosityAggregator:
    """Aggregate 4 curiosity channels into a scalar drive signal.

    Channels: social, epistemic, perceptual, metacognitive.

    Args:
        weights: Optional per-channel weights (default: uniform).
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        if weights is not None:
            self._weights = {ch: weights.get(ch, 1.0) for ch in _CURIOSITY_CHANNELS}
        else:
            self._weights = dict.fromkeys(_CURIOSITY_CHANNELS, 1.0)

    def aggregate(self, scores: dict[str, float]) -> float:
        """Weighted mean of curiosity channel scores.

        Args:
            scores: Mapping from channel name to score ``[0, 1]``.

        Returns:
            Aggregated curiosity drive scalar.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        for ch in _CURIOSITY_CHANNELS:
            w = self._weights[ch]
            weighted_sum += w * scores.get(ch, 0.0)
            total_weight += w
        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Policy / Value MLP networks (numpy)
# ---------------------------------------------------------------------------


class PolicyMLP:
    """Lightweight numpy policy network.

    Two hidden layers with ReLU + layer-norm, tanh output.

    Args:
        input_dim: Dimension of the state input.
        action_dim: Dimension of the action output.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_BELIEF_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
    ) -> None:
        rng = np.random.default_rng(POLICY_MLP_SEED)
        hidden = _POLICY_HIDDEN_DIM
        self._w1 = rng.standard_normal((input_dim, hidden)).astype(np.float32) * WEIGHT_INIT_SCALE
        self._b1 = np.zeros(hidden, dtype=np.float32)
        self._w2 = rng.standard_normal((hidden, action_dim)).astype(np.float32) * WEIGHT_INIT_SCALE
        self._b2 = np.zeros(action_dim, dtype=np.float32)

    def forward(self, state: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Compute action from state.

        Args:
            state: 1-D state vector.

        Returns:
            Action vector in ``[-1, 1]``.
        """
        h = relu(layer_norm(state @ self._w1 + self._b1))
        return cast(NDArray[np.floating[Any]], np.tanh(h @ self._w2 + self._b2))

    def save(self, path: Path | str) -> None:
        """Save weights to ``.npz`` file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w1=self._w1, b1=self._b1, w2=self._w2, b2=self._b2)

    def load(self, path: Path | str) -> None:
        """Load weights from ``.npz`` file.

        Args:
            path: Source file path.
        """
        data = np.load(path)
        self._w1 = data["w1"]
        self._b1 = data["b1"]
        self._w2 = data["w2"]
        self._b2 = data["b2"]


class ValueMLP:
    """Lightweight numpy value network.

    Two hidden layers with ReLU + layer-norm, scalar output.

    Args:
        input_dim: Dimension of the state input.
    """

    def __init__(self, input_dim: int = DEFAULT_BELIEF_DIM) -> None:
        rng = np.random.default_rng(VALUE_MLP_SEED)
        hidden = _POLICY_HIDDEN_DIM
        self._w1 = rng.standard_normal((input_dim, hidden)).astype(np.float32) * WEIGHT_INIT_SCALE
        self._b1 = np.zeros(hidden, dtype=np.float32)
        self._w2 = rng.standard_normal((hidden, 1)).astype(np.float32) * WEIGHT_INIT_SCALE
        self._b2 = np.zeros(1, dtype=np.float32)

    def forward(self, state: NDArray[np.floating[Any]]) -> float:
        """Compute value estimate from state.

        Args:
            state: 1-D state vector.

        Returns:
            Scalar value estimate.
        """
        h = relu(layer_norm(state @ self._w1 + self._b1))
        return float((h @ self._w2 + self._b2)[0])

    def save(self, path: Path | str) -> None:
        """Save weights to ``.npz`` file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w1=self._w1, b1=self._b1, w2=self._w2, b2=self._b2)

    def load(self, path: Path | str) -> None:
        """Load weights from ``.npz`` file.

        Args:
            path: Source file path.
        """
        data = np.load(path)
        self._w1 = data["w1"]
        self._b1 = data["b1"]
        self._w2 = data["w2"]
        self._b2 = data["b2"]
