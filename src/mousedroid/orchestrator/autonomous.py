"""Autonomous orchestrator control loop coordinating perception, reasoning, and actuation."""

from __future__ import annotations

import asyncio
import time

from mousedroid.config.schema.root import Settings
from mousedroid.constants import (
    DEFAULT_CONTROL_LOOP_INTERVAL_S,
    DEFAULT_LIDAR_MIN_RANGE_M,
    MILLISECONDS_PER_SECOND,
)
from mousedroid.interfaces.protocols import (
    CameraProtocol,
    GoalVector,
    LiDARProtocol,
    LLMGatewayProtocol,
    MetricsRegistryProtocol,
    MotorControllerProtocol,
)
from mousedroid.logging.setup import get_logger

_log = get_logger("mousedroid.orchestrator.autonomous")


class AutonomousOrchestrator:
    """Master autonomous mission control loop coordinating perception, safety, and actuation."""

    def __init__(
        self,
        cfg: Settings,
        motor: MotorControllerProtocol,
        camera: CameraProtocol,
        lidar: LiDARProtocol,
        llm: LLMGatewayProtocol,
        metrics: MetricsRegistryProtocol | None = None,
    ) -> None:
        self._cfg = cfg
        self._motor = motor
        self._camera = camera
        self._lidar = lidar
        self._llm = llm
        self._metrics = metrics
        self._running: bool = False
        self._step_count: int = 0
        self._safety_latched: bool = False
        _log.info("autonomous_orchestrator_initialized")

    @property
    def is_running(self) -> bool:
        """Return True if orchestrator loop is active."""
        return self._running

    def validate_sensors(self) -> bool:
        """Perform pre-flight health check across all perception and actuation subsystems."""
        motor_ok = getattr(self._motor, "is_healthy", lambda: True)()
        lidar_ok = getattr(self._lidar, "is_healthy", lambda: True)()
        cam_ok = getattr(self._camera, "is_healthy", lambda: True)()
        all_healthy = bool(motor_ok and lidar_ok and cam_ok)
        _log.info(
            "preflight_sensor_validation",
            motor=motor_ok,
            lidar=lidar_ok,
            camera=cam_ok,
            passed=all_healthy,
        )
        return all_healthy

    async def execute_mission_step(self, mission_command: str) -> bool:
        """Execute a single perception-reasoning-actuation cycle.

        Args:
            mission_command: Operator natural language command.

        Returns:
            True if cycle executed safely, False if safety intervention occurred or error happened.
        """
        self._step_count += 1
        t_start = time.perf_counter()

        try:
            # 1. Perception Ingestion
            scan = await self._lidar.get_latest_scan()
            _ = await self._camera.capture_frame()

            # 2. Safety Interlock: Proximity Obstacle Check
            min_distance_threshold = (
                self._cfg.lidar.min_range_m if self._cfg.lidar else DEFAULT_LIDAR_MIN_RANGE_M
            )
            min_obstacle_dist = min(scan) if scan else float("inf")

            if min_obstacle_dist < min_distance_threshold:
                _log.warning(
                    "obstacle_too_close_emergency_stop",
                    distance=min_obstacle_dist,
                    threshold=min_distance_threshold,
                )
                self._safety_latched = True
                await self._motor.emergency_stop()
                if self._metrics is not None:
                    self._metrics.record_counter(
                        "mousedroid_safety_interventions_total",
                        value=1.0,
                        labels={"cause": "proximity"},
                    )
                return False

            self._safety_latched = False

            # 3. Mission Synthesis via Hybrid LLM Gateway
            goal: GoalVector = await self._llm.translate_mission(mission_command)

            # 4. Actuation
            if not goal.is_safe or goal.arm_action == "e_stop":
                _log.warning("goal_vector_unsafe_triggering_estop", confidence=goal.confidence)
                await self._motor.emergency_stop()
                if self._metrics is not None:
                    self._metrics.record_counter(
                        "mousedroid_safety_interventions_total",
                        value=1.0,
                        labels={"cause": "unsafe_goal"},
                    )
                return False

            success = await self._motor.set_velocity(goal.linear_velocity, goal.angular_velocity)
            return success
        except asyncio.CancelledError:
            _log.warning("mission_step_cancelled_executing_emergency_stop")
            await self._motor.emergency_stop()
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t_start) * MILLISECONDS_PER_SECOND
            if self._metrics is not None:
                self._metrics.record_histogram(
                    "mousedroid_orchestrator_tick_latency_ms",
                    value=elapsed_ms,
                    labels={"status": "executed"},
                )

    async def run_loop(
        self,
        mission_command: str,
        iterations: int = 1,
        interval_s: float = DEFAULT_CONTROL_LOOP_INTERVAL_S,
    ) -> int:
        """Run mission loop for N steps or until stopped.

        Args:
            mission_command: Command string for the mission.
            iterations: Number of loop iterations.
            interval_s: Sleep duration between loop ticks.

        Returns:
            Number of successfully and safely executed steps.
        """
        self._running = True
        safe_steps = 0
        try:
            for _ in range(iterations):
                if not self._running:
                    break
                ok = await self.execute_mission_step(mission_command)
                if ok:
                    safe_steps += 1
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            _log.info("orchestrator_loop_cancelled")
            await self._motor.emergency_stop()
            raise
        finally:
            self._running = False
        return safe_steps

    async def stop(self) -> None:
        """Graceful teardown of all subsystem tasks."""
        self._running = False
        await self._motor.emergency_stop()
        await self._motor.close()
        await self._camera.close()
        await self._lidar.close()
        await self._llm.stop()
        _log.info("autonomous_orchestrator_stopped_safely", total_steps=self._step_count)
