"""Integration test: orchestrator drives the face controller from BDI affect."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import torch

from mousedroid.config.schema import FaceDisplayConfig, Settings
from mousedroid.hardware.display.expressions import Expression
from mousedroid.hardware.display.mock_face_driver import MockFaceDriver
from mousedroid.orchestrator.face_controller import FaceController
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle


def _make_observation(cfg: Settings) -> MouseDroidObservationBundle:
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(1024, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def _make_orchestrator(
    *,
    emergency: bool = False,
    affect: np.ndarray | None = None,
    action: torch.Tensor | None = None,
) -> tuple[MouseDroidOrchestrator, MockFaceDriver]:
    cfg = Settings(
        mock_hardware=True,
        face_display=FaceDisplayConfig(enabled=True, min_dwell_s=0.0),
    )

    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )

    selected_action = action if action is not None else torch.tensor([0.1, 0.0, 0.0])
    agent = MagicMock()
    agent.name = "test_agent"
    agent.act.return_value = selected_action

    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=emergency)

    esp32 = AsyncMock()
    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _make_observation(cfg)
    sensor_manager.recovery_attempt.return_value = 0

    cognitive_core = MagicMock()
    cognitive_core._latest_bdi = {"affect": affect} if affect is not None else {}

    drv = MockFaceDriver(cfg.face_display)
    face_controller = FaceController(drv, cfg.face_display, clock=lambda: 1000.0)

    orch = MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        cognitive_core=cognitive_core,
        face_controller=face_controller,
    )
    return orch, drv


async def test_tick_drives_face_from_bdi_affect() -> None:
    affect = np.array([0.6, 0.2], dtype=np.float32)
    orch, drv = _make_orchestrator(affect=affect)
    await orch._face_controller.start()
    await orch.tick()
    assert drv.current is Expression.HAPPY


async def test_tick_emergency_path_renders_emergency() -> None:
    orch, drv = _make_orchestrator(emergency=True)
    await orch._face_controller.start()
    await orch.tick()
    assert drv.current is Expression.EMERGENCY


async def test_tick_handles_missing_affect_gracefully() -> None:
    orch, drv = _make_orchestrator(affect=None)
    await orch._face_controller.start()
    # Affect missing → controller treats as zero valence/arousal → NEUTRAL.
    await orch.tick()
    assert drv.current is Expression.NEUTRAL


async def test_orchestrator_without_face_controller_is_noop() -> None:
    cfg = Settings(mock_hardware=True)
    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )
    agent = MagicMock()
    agent.name = "test_agent"
    agent.act.return_value = torch.tensor([0.0, 0.0, 0.0])
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)
    esp32 = AsyncMock()
    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _make_observation(cfg)
    sensor_manager.recovery_attempt.return_value = 0

    orch = MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        face_controller=None,
    )
    # Tick must run without raising despite no face controller.
    await orch.tick()
