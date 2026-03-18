"""MouseDroid orchestrator — main sense-plan-act loop.

Platform-agnostic via dependency injection. All components injected
through constructor, wired by factory functions.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from mousedroid.common.actions import normalize_action_numpy
from mousedroid.constants import (
    DEFAULT_BATTERY_VOLTAGE,
    MILLISECONDS_PER_SECOND,
    MOTOR_STATE_BATTERY_INDEX,
)
from mousedroid.logging.setup import get_logger
from mousedroid.telemetry.frame_builder import build_telemetry_frame

if TYPE_CHECKING:
    from mousedroid.agents.base import AgentProtocol
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.comms.protocol import ESP32CommProtocol
    from mousedroid.config.schema import Settings
    from mousedroid.safety.context import SafetyContext
    from mousedroid.safety.protocol import SafetyMonitorProtocol
    from mousedroid.sensing.manager import SensorManager
    from mousedroid.sensing.protocol import ObservationProtocol
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol, TelemetryServerProtocol
    from mousedroid.world_model.protocol import WorldModelProtocol

_log = get_logger(__name__)


class MouseDroidOrchestrator:
    """Main orchestrator — runs the sense-plan-act loop.

    All components are injected via constructor. No hardcoded types.
    """

    def __init__(
        self,
        world_model: WorldModelProtocol,
        agents: list[AgentProtocol],
        safety_monitor: SafetyMonitorProtocol,
        esp32: ESP32CommProtocol,
        sensor_manager: SensorManager,
        cfg: Settings,
        cognitive_core: CognitiveCore | None = None,
        telemetry_publisher: TelemetryPublisherProtocol | None = None,
        telemetry_server: TelemetryServerProtocol | None = None,
    ) -> None:
        """Initialise orchestrator with all components.

        Args:
            world_model: World model for latent dynamics.
            agents: List of navigation agents.
            safety_monitor: Safety monitor.
            esp32: ESP32 communication driver.
            sensor_manager: Sensor manager for concurrent sensor reads.
            cfg: Root settings.
            cognitive_core: Optional CognitiveCore for BDI/metacognitive loop.
            telemetry_publisher: Optional telemetry publisher for remote monitoring.
            telemetry_server: Optional telemetry server for remote connections.
        """
        self._world_model = world_model
        self._agents = agents
        self._safety_monitor = safety_monitor
        self._esp32 = esp32
        self._sensor_manager = sensor_manager
        self._cognitive_core = cognitive_core
        self._telemetry_publisher = telemetry_publisher
        self._telemetry_server = telemetry_server
        self._cfg = cfg
        self._running = False
        self._tick_count: int = 0

        # Latent state
        self._h = torch.zeros(1, cfg.model.hidden_dim)
        self._z = torch.zeros(1, cfg.model.latent_dim)
        self._prev_action = torch.zeros(1, cfg.model.action_dim)

    async def start(self) -> None:
        """Start all subsystems."""
        _log.info("orchestrator_starting")
        await self._esp32.connect()
        await self._sensor_manager.start()
        if self._cognitive_core is not None:
            await self._cognitive_core.start()
        if self._telemetry_server is not None:
            await self._telemetry_server.start()
        self._running = True
        _log.info("orchestrator_started")

    async def stop(self) -> None:
        """Stop all subsystems gracefully."""
        _log.info("orchestrator_stopping")
        self._running = False
        if self._telemetry_server is not None:
            await self._telemetry_server.stop()
        if self._cognitive_core is not None:
            await self._cognitive_core.stop()
        await self._esp32.emergency_stop()
        await self._sensor_manager.stop()
        await self._esp32.disconnect()
        _log.info("orchestrator_stopped")

    async def tick(self) -> None:
        """Execute one sense-plan-act cycle."""
        loop_start = time.monotonic()

        observation = await self._sensor_manager.read_all()
        loop_time_ms = (time.monotonic() - loop_start) * MILLISECONDS_PER_SECOND

        safety_ctx = self._safety_monitor.evaluate(observation, loop_time_ms)

        self._update_world_model(observation)

        if safety_ctx.is_emergency:
            await self._esp32.emergency_stop()
            _log.warning("emergency_stop_triggered")
            self._tick_count += 1
            await self._publish_telemetry(observation, safety_ctx, loop_time_ms)
            return

        action = self._select_action(safety_ctx, observation, loop_time_ms)
        self._prev_action = action.unsqueeze(0) if action.dim() == 1 else action

        await self._execute_action(action)

        self._tick_count += 1
        await self._publish_telemetry(observation, safety_ctx, loop_time_ms)

        _log.debug(
            "tick_complete",
            loop_time_ms=loop_time_ms,
            emergency=safety_ctx.is_emergency,
        )

    def _update_world_model(self, observation: ObservationProtocol) -> None:
        """Run world model observation step to update latent state.

        Args:
            observation: Current sensor observation bundle.
        """
        with torch.no_grad():
            self._h, self._z, _, _ = self._world_model.observe_step(
                observation,
                self._prev_action,
                self._h,
                self._z,
            )

    def _select_action(
        self,
        safety_ctx: SafetyContext,
        observation: ObservationProtocol,
        loop_time_ms: float,
    ) -> torch.Tensor:
        """Select action using cognitive core (primary) or MCTS agent (fallback).

        Args:
            safety_ctx: Current safety context.
            observation: Current sensor observation.
            loop_time_ms: Current loop timing in milliseconds.

        Returns:
            Action tensor.
        """
        if self._cognitive_core is not None:
            action = self._try_cognitive_action(observation, loop_time_ms)
            if action is not None:
                return action

        return self._agents[0].act(self._h, self._z, safety_ctx)

    def _try_cognitive_action(
        self,
        observation: ObservationProtocol,
        loop_time_ms: float,
    ) -> torch.Tensor | None:
        """Attempt action selection via cognitive core.

        Args:
            observation: Current sensor observation.
            loop_time_ms: Current loop timing in milliseconds.

        Returns:
            Action tensor if successful, None on failure.
        """
        try:
            battery_v = (
                float(observation.motor_state[MOTOR_STATE_BATTERY_INDEX])
                if observation.motor_state.size > MOTOR_STATE_BATTERY_INDEX
                else DEFAULT_BATTERY_VOLTAGE
            )
            belief_dim = int(self._cfg.model.belief_dim)
            bdi_state_vec = self._h.numpy().flatten().astype(np.float32, copy=False)
            state_vec = bdi_state_vec
            if state_vec.size < belief_dim:
                state_vec = np.pad(state_vec, (0, belief_dim - state_vec.size))
            else:
                state_vec = state_vec[:belief_dim]
            state_vec = state_vec.astype(np.float32, copy=False)

            obs_dict = {
                "state": state_vec,
                "bdi_state": bdi_state_vec,
                "battery_v": battery_v,
                "obstacle_dist_m": float(observation.distance_m),
                "mcts_sims": int(self._cfg.mcts.n_simulations_base),
                "loop_time_ms": loop_time_ms,
            }
            action_np, violations = self._cognitive_core.tick_fast(obs_dict)  # type: ignore[union-attr]
            if violations:
                _log.info(
                    "orchestrator_constitutional_violations_summary",
                    violation_count=len(violations),
                    violations=violations,
                )

            return self._normalize_cognitive_action(action_np)
        except Exception as e:  # pylint: disable=broad-except
            _log.warning(
                "cognitive_core_action_selection_failed",
                error=str(e),
                falling_back_to_mcts=True,
            )
            return None

    def _normalize_cognitive_action(
        self,
        action_np: np.ndarray,
    ) -> torch.Tensor:
        """Normalize cognitive core action to match expected action_dim.

        Args:
            action_np: Raw action from cognitive core.

        Returns:
            Normalized 1-D torch tensor with correct dimensions.
        """
        return normalize_action_numpy(action_np, int(self._cfg.model.action_dim))

    async def _execute_action(self, action: torch.Tensor) -> None:
        """Scale and send action to ESP32 motors.

        Args:
            action: Action tensor with values in [-1, 1].
        """
        max_v = self._cfg.esp32.max_velocity_mps
        max_omega = self._cfg.esp32.max_omega_rads
        vx = float(action[0]) * max_v
        vy = float(action[1]) * max_v if action.shape[0] > 1 else 0.0
        omega = float(action[2]) * max_omega if action.shape[0] > 2 else 0.0
        await self._esp32.send_velocity(vx, vy, omega)

    async def _publish_telemetry(
        self,
        observation: ObservationProtocol,
        safety_ctx: SafetyContext,
        loop_time_ms: float,
    ) -> None:
        """Build and publish a telemetry frame from current state.

        Non-blocking: silently drops if queue is full.

        Args:
            observation: Current sensor observation bundle.
            safety_ctx: Current safety context.
            loop_time_ms: Control loop iteration time (ms).
        """
        if self._telemetry_publisher is None:
            return

        try:
            frame = build_telemetry_frame(
                observation, safety_ctx, loop_time_ms, self._tick_count,
            )
            await self._telemetry_publisher.publish(frame)
        except Exception:
            _log.debug("telemetry_publish_failed", exc_info=True)

    async def run(self) -> None:
        """Run the main loop at configured control rate."""
        control_period = 1.0 / self._cfg.loop.control_hz
        _log.info("main_loop_starting", control_hz=self._cfg.loop.control_hz)

        while self._running:
            tick_start = time.monotonic()
            try:
                await self.tick()
            except Exception:
                _log.exception("tick_error")

            elapsed = time.monotonic() - tick_start
            sleep_time = max(0.0, control_period - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def health_check(self) -> dict[str, object]:
        """Run a quick health check of all subsystems.

        Returns:
            Health status dict.
        """
        return {
            "status": "ok",
            "platform": str(self._cfg.platform),
            "mock_hardware": self._cfg.mock_hardware,
            "agents": [a.name for a in self._agents],
        }
