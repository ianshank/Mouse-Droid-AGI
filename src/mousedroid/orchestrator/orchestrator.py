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

from mousedroid.constants import (
    DEFAULT_AUDIO_CHUNK_SIZE,
    DEFAULT_MOTOR_STATE_DIM,
    N_SENSOR_MODALITIES,
)
from mousedroid.logging.setup import get_logger
from mousedroid.sensing.bundle import MouseDroidObservationBundle

if TYPE_CHECKING:
    from mousedroid.agents.base import AgentProtocol
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.comms.protocol import ESP32CommProtocol
    from mousedroid.config.schema import Settings
    from mousedroid.hardware.protocols import AudioProtocol, DistanceSensorProtocol, VisionProtocol
    from mousedroid.safety.protocol import SafetyMonitorProtocol
    from mousedroid.world_model.protocol import WorldModelProtocol

_log = get_logger(__name__)

_SECONDS_TO_MS: float = 1000.0


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
        camera: VisionProtocol,
        distance_sensor: DistanceSensorProtocol,
        cfg: Settings,
        cognitive_core: CognitiveCore | None = None,
        microphone: AudioProtocol | None = None,
    ) -> None:
        """Initialise orchestrator with all components.

        Args:
            world_model: World model for latent dynamics.
            agents: List of navigation agents.
            safety_monitor: Safety monitor.
            esp32: ESP32 communication driver.
            camera: Vision driver.
            distance_sensor: Distance sensor driver.
            cfg: Root settings.
            cognitive_core: Optional CognitiveCore for BDI/metacognitive loop.
            microphone: Optional USB microphone driver.
        """
        self._world_model = world_model
        self._agents = agents
        self._safety_monitor = safety_monitor
        self._esp32 = esp32
        self._camera = camera
        self._distance_sensor = distance_sensor
        self._cognitive_core = cognitive_core
        self._microphone = microphone
        self._cfg = cfg
        self._running = False

        # Latent state
        self._h = torch.zeros(1, cfg.model.hidden_dim)
        self._z = torch.zeros(1, cfg.model.latent_dim)
        self._prev_action = torch.zeros(1, cfg.model.action_dim)

    async def start(self) -> None:
        """Start all subsystems."""
        _log.info("orchestrator_starting")
        await self._esp32.connect()
        await self._camera.start()
        if self._microphone is not None:
            await self._microphone.start()
        if self._cognitive_core is not None:
            await self._cognitive_core.start()
        self._running = True
        _log.info("orchestrator_started")

    async def stop(self) -> None:
        """Stop all subsystems gracefully."""
        _log.info("orchestrator_stopping")
        self._running = False
        if self._cognitive_core is not None:
            await self._cognitive_core.stop()
        await self._esp32.emergency_stop()
        await self._camera.stop()
        if self._microphone is not None:
            await self._microphone.stop()
        await self._esp32.disconnect()
        _log.info("orchestrator_stopped")

    async def tick(self) -> None:
        """Execute one sense-plan-act cycle."""
        loop_start = time.monotonic()

        # Sense
        observation = await self._sense()

        loop_time_ms = (time.monotonic() - loop_start) * _SECONDS_TO_MS

        # Evaluate safety
        safety_ctx = self._safety_monitor.evaluate(observation, loop_time_ms)

        # Plan + Act
        with torch.no_grad():
            self._h, self._z, _, surprise = self._world_model.observe_step(
                observation,
                self._prev_action,
                self._h,
                self._z,
            )

        if safety_ctx.is_emergency:
            await self._esp32.emergency_stop()
            _log.warning("emergency_stop_triggered", surprise=surprise)
            return

        # Select action — cognitive core primary, MCTS fallback
        action = None
        if self._cognitive_core is not None:
            try:
                # Build observation dict for cognitive core fast path
                battery_v = (
                    float(observation.motor_state[3]) if observation.motor_state.size > 3 else 12.6
                )
                belief_dim = int(self._cfg.model.belief_dim)
                state_vec = self._h.numpy().flatten()
                if state_vec.size < belief_dim:
                    state_vec = np.pad(state_vec, (0, belief_dim - state_vec.size))
                else:
                    state_vec = state_vec[:belief_dim]
                obs_dict = {
                    "state": state_vec,
                    "battery_v": battery_v,
                    "obstacle_dist_m": float(observation.distance_m),
                    "mcts_sims": int(self._cfg.mcts.n_simulations_base),
                    "loop_time_ms": loop_time_ms,
                }
                # Cognitive core returns (safe_action, violations)
                action_np, violations = self._cognitive_core.tick_fast(obs_dict)
                if violations:
                    _log.info(
                        "orchestrator_constitutional_violations_summary",
                        violation_count=len(violations),
                        violations=violations,
                    )
                # Ensure cognitive action matches expected action_dim
                action_np = np.asarray(action_np, dtype=np.float32).flatten()
                expected_action_dim = int(self._cfg.model.action_dim)
                if action_np.size < expected_action_dim:
                    _log.warning(
                        "cognitive_core_action_padded",
                        received_dim=int(action_np.size),
                        expected_dim=expected_action_dim,
                    )
                    action_np = np.pad(action_np, (0, expected_action_dim - action_np.size))
                elif action_np.size > expected_action_dim:
                    _log.warning(
                        "cognitive_core_action_truncated",
                        received_dim=int(action_np.size),
                        expected_dim=expected_action_dim,
                    )
                    action_np = action_np[:expected_action_dim]
                # Convert numpy array to 1D torch tensor for execution
                action = torch.from_numpy(action_np).float()
            except Exception as e:  # pylint: disable=broad-except
                _log.warning(
                    "cognitive_core_action_selection_failed",
                    error=str(e),
                    falling_back_to_mcts=True,
                )
                action = None  # Will fall back to MCTS below

        # Fallback to MCTS if cognitive not available or failed
        if action is None:
            action = self._agents[0].act(self._h, self._z, safety_ctx)

        self._prev_action = action.unsqueeze(0) if action.dim() == 1 else action

        # Execute
        max_v = self._cfg.esp32.max_velocity_mps
        max_omega = self._cfg.esp32.max_omega_rads
        vx = float(action[0]) * max_v
        vy = float(action[1]) * max_v if action.shape[0] > 1 else 0.0
        omega = float(action[2]) * max_omega if action.shape[0] > 2 else 0.0

        await self._esp32.send_velocity(vx, vy, omega)

        _log.debug(
            "tick_complete",
            loop_time_ms=loop_time_ms,
            emergency=safety_ctx.is_emergency,
        )

    async def _sense(self) -> MouseDroidObservationBundle:
        """Read all sensors and build observation bundle.

        Returns:
            Fused observation bundle.
        """
        vision_features = np.zeros(self._cfg.camera.feature_dim, dtype=np.float32)
        distance_m = self._distance_sensor.max_range_m
        motor_state = np.zeros(DEFAULT_MOTOR_STATE_DIM, dtype=np.float32)
        audio_chunk_size = (
            self._microphone.chunk_size * self._microphone.channels
            if self._microphone is not None
            else DEFAULT_AUDIO_CHUNK_SIZE
        )
        audio_chunk = np.zeros(audio_chunk_size, dtype=np.float32)
        valid_mask = np.zeros(N_SENSOR_MODALITIES, dtype=np.float32)

        try:
            vision_features = await self._camera.capture_features()
            valid_mask[0] = 1.0
        except Exception:
            _log.warning("vision_capture_failed", exc_info=True)

        try:
            distance_m = await self._distance_sensor.read_distance_m()
            valid_mask[1] = 1.0
        except Exception:
            _log.warning("distance_read_failed", exc_info=True)

        try:
            encoders = await self._esp32.read_encoders()
            battery = await self._esp32.get_battery_voltage()
            motor_state = np.array(
                [encoders.left_velocity_mps, encoders.right_velocity_mps, 0.0, battery],
                dtype=np.float32,
            )
            valid_mask[2] = 1.0
        except Exception:
            _log.warning("motor_state_read_failed", exc_info=True)

        if self._microphone is not None:
            try:
                audio_chunk = await self._microphone.read_chunk()
                valid_mask[3] = 1.0
            except Exception:
                _log.warning("audio_capture_failed", exc_info=True)

        return MouseDroidObservationBundle(
            _timestamp=time.monotonic(),
            _vision_features=vision_features,
            _distance_m=distance_m,
            _motor_state=motor_state,
            _audio_chunk=audio_chunk,
            _valid_mask=valid_mask,
        )

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
