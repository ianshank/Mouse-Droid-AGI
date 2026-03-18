"""CognitiveCore — dual-cadence cognitive loop.

Runs a fast tick at 30 Hz (<1 ms target) for constitutional checking and
curiosity, plus a slow loop at ~1 Hz for BDI inference and metacognitive
updates.  All computation is numpy-only.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.cognitive.bdi_model import NeuralBDI
from mousedroid.cognitive.constitutional_rl import (
    ConstitutionalChecker,
    CuriosityAggregator,
    PolicyMLP,
)
from mousedroid.cognitive.metacognitive import MetacognitiveModel
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_SLOW_LOOP_INTERVAL_S: float = 1.0
"""Target interval for the slow (BDI + metacognitive) loop."""

_SLOW_QUEUE_MAXSIZE: int = 2
"""Maximum backlog for the slow-loop work queue."""

_FAST_STATE_DIM: int = 128
"""Expected dimensionality of fast-tick state vectors."""


class CognitiveCore:
    """Dual-cadence cognitive controller.

    * **Fast path** (``tick_fast``, 30 Hz): constitutional check +
      curiosity aggregation.
    * **Slow path** (``_slow_loop``, ~1 Hz): BDI inference +
      metacognitive self-model update, offloaded via
      :func:`asyncio.to_thread`.

    Args:
        bdi: Neural BDI model for intention inference.
        metacog: Metacognitive self-model.
        checker: Constitutional safety checker.
        policy: Optional policy MLP (defaults to a fresh instance).
    """

    def __init__(
        self,
        bdi: NeuralBDI,
        metacog: MetacognitiveModel,
        checker: ConstitutionalChecker,
        policy: PolicyMLP | None = None,
    ) -> None:
        self._bdi = bdi
        self._metacog = metacog
        self._checker = checker
        self._rl = _RLBundle(policy=policy or PolicyMLP())
        self._curiosity = CuriosityAggregator()

        self._slow_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=_SLOW_QUEUE_MAXSIZE,
        )
        self._slow_task: asyncio.Task[None] | None = None
        self._latest_bdi: dict[str, Any] = {}

        _log.info("cognitive_core_init")

    # -- Fast path (30 Hz) --------------------------------------------------

    def tick_fast(
        self,
        observation_dict: dict[str, Any],
    ) -> tuple[NDArray[np.floating[Any]], list[str]]:
        """Run one fast-cadence tick: policy -> constitutional check.

        Args:
            observation_dict: Observation with at least ``"state"``
                (NDArray) and optional ``"curiosity"`` (dict of channel
                scores) and context keys for the constitutional checker.

        Returns:
            Tuple of ``(safe_action, violations)``.
        """
        default_state = np.zeros(_FAST_STATE_DIM)
        state = np.asarray(observation_dict.get("state", default_state), dtype=np.float32)

        # Policy forward pass.
        raw_action = self._rl._policy.forward(state)

        # Curiosity signal (informational, logged but not blocking).
        curiosity_scores: dict[str, float] = observation_dict.get("curiosity", {})
        if curiosity_scores:
            drive = self._rl._curiosity.aggregate(curiosity_scores)
            _log.debug("curiosity_drive", drive=drive)

        # Constitutional safety check.
        context: dict[str, Any] = {
            k: observation_dict[k]
            for k in (
                "battery_v",
                "obstacle_dist_m",
                "mcts_sims",
                "human_detected",
                "human_dist_m",
                "commanded_action",
            )
            if k in observation_dict
        }
        safe_action, violations = self._checker.check(raw_action, context)

        # Enqueue for slow loop (non-blocking, drop if full).
        with contextlib.suppress(asyncio.QueueFull):
            self._slow_queue.put_nowait(observation_dict)

        return safe_action, violations

    # -- Slow path (~1 Hz) --------------------------------------------------

    async def start(self) -> None:
        """Start the background slow-loop task.

        Must be called inside a running event loop.
        """
        if self._slow_task is None or self._slow_task.done():
            self._slow_task = asyncio.create_task(self._slow_loop())
            _log.info("cognitive_slow_loop_started")

    async def stop(self) -> None:
        """Cancel the slow-loop task and wait for it to finish."""
        if self._slow_task is not None and not self._slow_task.done():
            self._slow_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._slow_task
            _log.info("cognitive_slow_loop_stopped")

    async def _slow_loop(self) -> None:
        """Background loop: BDI inference + metacognitive update at ~1 Hz."""
        while True:
            try:
                obs = await asyncio.wait_for(
                    self._slow_queue.get(),
                    timeout=_SLOW_LOOP_INTERVAL_S,
                )
            except TimeoutError:
                continue

            state = np.asarray(
                obs.get("state", np.zeros(_FAST_STATE_DIM)),
                dtype=np.float32,
            )

            # Offload heavy numpy work to a thread.
            bdi_result = await asyncio.to_thread(self._bdi.infer, state)
            self._latest_bdi = bdi_result

            # Build metrics for metacognitive update.
            metrics: dict[str, float] = {}
            if "battery_v" in obs:
                metrics["battery_v"] = float(obs["battery_v"])
            if "loop_time_ms" in obs:
                metrics["loop_time_ms"] = float(obs["loop_time_ms"])
            bdi_intentions = bdi_result.get("intentions")
            if bdi_intentions is not None:
                metrics["bdi_score"] = float(np.max(bdi_intentions))

            await asyncio.to_thread(self._metacog.update, metrics)
            _log.debug(
                "slow_loop_tick",
                bdi_keys=list(bdi_result.keys()),
                metacog=self._metacog.get_capability_summary(),
            )


# ---------------------------------------------------------------------------
# Internal RL bundle (groups policy + curiosity for private access)
# ---------------------------------------------------------------------------


class _RLBundle:
    """Internal grouping of RL components.

    Args:
        policy: Policy MLP network.
    """

    def __init__(self, policy: PolicyMLP) -> None:
        self._policy = policy
        self._curiosity = CuriosityAggregator()
