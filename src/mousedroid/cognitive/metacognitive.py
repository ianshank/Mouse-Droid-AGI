"""Metacognitive self-model — numpy-only capability tracking.

Maintains an 8-dimensional capability vector updated via exponential moving
average and a causal knowledge graph for degradation reasoning.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_GEO_MEAN_FLOOR: float = 1e-10
"""Floor applied before geometric-mean computation to avoid log(0)."""

_BATTERY_NORM_EPS: float = 1e-6
"""Epsilon for battery voltage normalisation denominator."""

_N_CAPABILITIES: int = 8
"""Number of tracked capability dimensions."""

_EMA_DEFAULT_ALPHA: float = 0.1
"""Default EMA blending factor (higher = faster tracking)."""

_TARGET_LOOP_MS: float = 33.0
"""Target control loop duration in milliseconds (30 Hz)."""

_LOOP_SCORE_SCALE: float = 100.0
"""Scaling factor for loop-timing capability degradation."""

_CAPABILITY_NAMES: tuple[str, ...] = (
    "navigation",
    "obstacle_avoidance",
    "battery_management",
    "vision",
    "mcts",
    "bdi",
    "loop_timing",
    "communication",
)
"""Ordered names for each capability dimension."""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _ema_blend(
    old: NDArray[np.floating[Any]],
    new: NDArray[np.floating[Any]],
    alpha: float,
) -> NDArray[np.floating[Any]]:
    """Compute exponential moving average blend.

    Args:
        old: Previous value array.
        new: New observation array.
        alpha: Blending coefficient in ``[0, 1]``.

    Returns:
        Blended array ``(1 - alpha) * old + alpha * new``.
    """
    return (1.0 - alpha) * old + alpha * new


def _apply_capped_penalty(
    value: NDArray[np.floating[Any]],
    penalty: NDArray[np.floating[Any]],
    floor: float = 0.0,
) -> NDArray[np.floating[Any]]:
    """Subtract *penalty* from *value*, capping at *floor*.

    Args:
        value: Current values.
        penalty: Non-negative penalty to subtract.
        floor: Minimum allowed value.

    Returns:
        Penalised array, element-wise floored.
    """
    return np.maximum(value - penalty, floor)


# ---------------------------------------------------------------------------
# MetacognitiveModel
# ---------------------------------------------------------------------------


class MetacognitiveModel:
    """Self-model tracking 8 capability dimensions with EMA updates.

    Also maintains a :mod:`networkx` directed graph for causal reasoning
    about capability degradation.

    Args:
        alpha: EMA blending factor.
        battery_nominal_v: Nominal battery voltage for normalisation.
    """

    def __init__(
        self,
        alpha: float = _EMA_DEFAULT_ALPHA,
        battery_nominal_v: float = 12.6,
    ) -> None:
        self._alpha = alpha
        self._battery_nominal_v = battery_nominal_v

        # Capability vector — starts at 1.0 (fully capable).
        self._capabilities: NDArray[np.float32] = np.ones(
            _N_CAPABILITIES,
            dtype=np.float32,
        )

        # Causal knowledge graph.
        self._graph: nx.DiGraph[str] = nx.DiGraph()
        self._build_causal_graph()

        _log.info("metacognitive_init", alpha=alpha, n_caps=_N_CAPABILITIES)

    # -- Public API ---------------------------------------------------------

    def update(self, metrics: dict[str, float]) -> None:
        """Update capability estimates from latest metrics.

        Recognised metric keys (all optional):

        * ``nav_score`` — navigation quality ``[0, 1]``
        * ``obstacle_score`` — obstacle avoidance quality ``[0, 1]``
        * ``battery_v`` — current battery voltage
        * ``vision_score`` — vision confidence ``[0, 1]``
        * ``mcts_score`` — MCTS search quality ``[0, 1]``
        * ``bdi_score`` — BDI confidence ``[0, 1]``
        * ``loop_time_ms`` — control loop duration (ms)
        * ``comm_score`` — communication health ``[0, 1]``

        Args:
            metrics: Mapping of metric names to float values.
        """
        new_caps = self._capabilities.copy()

        idx_map: dict[str, int] = {name: i for i, name in enumerate(_CAPABILITY_NAMES)}

        if "nav_score" in metrics:
            new_caps[idx_map["navigation"]] = float(metrics["nav_score"])
        if "obstacle_score" in metrics:
            new_caps[idx_map["obstacle_avoidance"]] = float(metrics["obstacle_score"])
        if "battery_v" in metrics:
            normed = float(metrics["battery_v"]) / (self._battery_nominal_v + _BATTERY_NORM_EPS)
            new_caps[idx_map["battery_management"]] = float(np.clip(normed, 0.0, 1.0))
        if "vision_score" in metrics:
            new_caps[idx_map["vision"]] = float(metrics["vision_score"])
        if "mcts_score" in metrics:
            new_caps[idx_map["mcts"]] = float(metrics["mcts_score"])
        if "bdi_score" in metrics:
            new_caps[idx_map["bdi"]] = float(metrics["bdi_score"])
        if "loop_time_ms" in metrics:
            # Score drops linearly beyond _TARGET_LOOP_MS.
            score = float(
                np.clip(
                    1.0 - (metrics["loop_time_ms"] - _TARGET_LOOP_MS) / _LOOP_SCORE_SCALE,
                    0.0,
                    1.0,
                )
            )
            new_caps[idx_map["loop_timing"]] = score
        if "comm_score" in metrics:
            new_caps[idx_map["communication"]] = float(metrics["comm_score"])

        self._capabilities = _ema_blend(self._capabilities, new_caps, self._alpha)
        _log.debug("metacognitive_update", caps=self._capabilities.tolist())

    def get_capability_summary(self) -> dict[str, float]:
        """Return current capability vector as a name-keyed dictionary.

        Returns:
            Mapping from capability name to current score ``[0, 1]``.
        """
        return {
            name: float(self._capabilities[i]) for i, name in enumerate(_CAPABILITY_NAMES)
        }

    def geometric_mean(self) -> float:
        """Compute the geometric mean of all capability scores.

        Returns:
            Scalar geometric mean, floored at :data:`_GEO_MEAN_FLOOR`.
        """
        clamped = np.clip(self._capabilities, _GEO_MEAN_FLOOR, None)
        return float(np.exp(np.mean(np.log(clamped))))

    def propagate_degradation(self, failed_cap: str) -> list[str]:
        """Identify capabilities causally downstream of *failed_cap*.

        Uses BFS on the internal causal graph.

        Args:
            failed_cap: Name of the capability that failed.

        Returns:
            List of downstream capability names (may be empty).
        """
        if failed_cap not in self._graph:
            return []
        return list(nx.descendants(self._graph, failed_cap))

    # -- Internal -----------------------------------------------------------

    def _build_causal_graph(self) -> None:
        """Populate the causal degradation graph with domain edges."""
        edges: list[tuple[str, str]] = [
            ("vision", "navigation"),
            ("vision", "obstacle_avoidance"),
            ("battery_management", "vision"),
            ("battery_management", "communication"),
            ("communication", "mcts"),
            ("loop_timing", "navigation"),
            ("loop_timing", "obstacle_avoidance"),
            ("bdi", "navigation"),
        ]
        self._graph.add_edges_from(edges)
