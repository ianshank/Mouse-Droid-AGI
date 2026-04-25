"""Unit-level tests for ``MouseDroidOrchestrator._update_face``.

These tests focus on the edge cases that the integration test in
``tests/integration/test_orchestrator_face_integration.py`` cannot cover
cleanly, namely:

* missing or wrong-shape ``affect`` defaults to neutral;
* ``cognitive_core=None`` defaults to neutral;
* face-controller exceptions are swallowed and logged (the orchestrator
  must never let a flaky display crash the loop).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.config.schema import FaceDisplayConfig, Settings
from mousedroid.hardware.display.expressions import Expression
from mousedroid.hardware.display.mock_face_driver import MockFaceDriver
from mousedroid.orchestrator.face_controller import FaceController
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext


def _make_orch(
    *, affect: object | None, cognitive_core_present: bool = True
) -> tuple[MouseDroidOrchestrator, MockFaceDriver]:
    cfg = Settings(
        mock_hardware=True,
        face_display=FaceDisplayConfig(enabled=True, min_dwell_s=0.0),
    )

    cognitive_core: MagicMock | None = None
    if cognitive_core_present:
        cognitive_core = MagicMock()
        cognitive_core._latest_bdi = {"affect": affect} if affect is not None else {}
        # Mirror CognitiveCore.get_latest_affect semantics in the mock.
        if isinstance(affect, np.ndarray) and affect.shape == (2,):
            cognitive_core.get_latest_affect.return_value = (float(affect[0]), float(affect[1]))
        else:
            cognitive_core.get_latest_affect.return_value = (0.0, 0.0)

    drv = MockFaceDriver(cfg.face_display)
    fc = FaceController(drv, cfg.face_display, clock=lambda: 1000.0)

    orch = MouseDroidOrchestrator(
        world_model=MagicMock(),
        agents=[MagicMock()],
        safety_monitor=MagicMock(),
        esp32=AsyncMock(),
        sensor_manager=AsyncMock(),
        cfg=cfg,
        cognitive_core=cognitive_core,
        face_controller=fc,
    )
    return orch, drv


async def test_update_face_uses_cognitive_affect() -> None:
    affect = np.array([0.6, 0.0], dtype=np.float32)
    orch, drv = _make_orch(affect=affect)
    await orch._face_controller.start()
    await orch._update_face(safety_ctx=SafetyContext(is_emergency=False), action=None)
    assert drv.current is Expression.HAPPY


async def test_update_face_handles_missing_affect_key() -> None:
    orch, drv = _make_orch(affect=None)
    await orch._face_controller.start()
    await orch._update_face(safety_ctx=SafetyContext(is_emergency=False), action=None)
    # affect missing → (0, 0); first idle tick still under idle_sleepy_after_s,
    # so the controller stays NEUTRAL until the threshold elapses.
    assert drv.current is Expression.NEUTRAL


async def test_update_face_no_cognitive_core_defaults_to_neutral() -> None:
    orch, drv = _make_orch(affect=None, cognitive_core_present=False)
    await orch._face_controller.start()
    await orch._update_face(
        safety_ctx=SafetyContext(is_emergency=False),
        action=torch.tensor([0.1, 0.0, 0.0]),
    )
    # Neutral because affect is zero and the action keeps us non-idle.
    assert drv.current is Expression.NEUTRAL


@pytest.mark.parametrize(
    "bad_affect",
    [
        np.array([0.0, 0.0, 0.0], dtype=np.float32),  # wrong rank
        np.array([[0.0, 0.0]], dtype=np.float32),  # wrong shape
        [0.0, 0.0],  # not an ndarray at all
    ],
)
async def test_update_face_rejects_malformed_affect(bad_affect: object) -> None:
    orch, drv = _make_orch(affect=bad_affect)
    await orch._face_controller.start()
    await orch._update_face(
        safety_ctx=SafetyContext(is_emergency=False),
        action=torch.tensor([0.1, 0.0, 0.0]),
    )
    assert drv.current is Expression.NEUTRAL


async def test_update_face_swallows_driver_exceptions() -> None:
    """A flaky display must never crash the orchestrator loop."""
    cfg = Settings(
        mock_hardware=True,
        face_display=FaceDisplayConfig(enabled=True, min_dwell_s=0.0),
    )
    drv = MockFaceDriver(cfg.face_display)
    fc = FaceController(drv, cfg.face_display, clock=lambda: 1000.0)
    fc.update = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

    orch = MouseDroidOrchestrator(
        world_model=MagicMock(),
        agents=[MagicMock()],
        safety_monitor=MagicMock(),
        esp32=AsyncMock(),
        sensor_manager=AsyncMock(),
        cfg=cfg,
        face_controller=fc,
    )
    # Must not raise.
    await orch._update_face(safety_ctx=SafetyContext(is_emergency=False), action=None)
    fc.update.assert_awaited_once()


async def test_update_face_emergency_passes_through() -> None:
    affect = np.array([0.7, 0.1], dtype=np.float32)
    orch, drv = _make_orch(affect=affect)
    await orch._face_controller.start()
    await orch._update_face(safety_ctx=SafetyContext(is_emergency=True), action=None)
    assert drv.current is Expression.EMERGENCY


async def test_near_zero_action_counts_as_idle() -> None:
    """NN float noise below 1e-3 on an otherwise-zero action must map to is_idle=True.

    Tests the idle-flag computation directly by capturing the keyword arguments
    forwarded to FaceController.update, avoiding the need to advance a fake
    clock past idle_sleepy_after_s.
    """
    orch, _ = _make_orch(affect=None)
    await orch._face_controller.start()

    captured: list[dict[str, object]] = []
    original_update = orch._face_controller.update

    async def _spy(**kwargs: object) -> None:
        captured.append(dict(kwargs))
        await original_update(**kwargs)

    orch._face_controller.update = _spy  # type: ignore[method-assign]

    noisy_action = torch.tensor([5e-4, -2e-4, 8e-4])
    await orch._update_face(safety_ctx=SafetyContext(is_emergency=False), action=noisy_action)

    assert len(captured) == 1, "update must have been called exactly once"
    assert captured[0]["is_idle"] is True


async def test_above_threshold_action_is_not_idle() -> None:
    """Action components ≥ 1e-3 must map to is_idle=False."""
    orch, _ = _make_orch(affect=None)
    await orch._face_controller.start()

    captured: list[dict[str, object]] = []
    original_update = orch._face_controller.update

    async def _spy(**kwargs: object) -> None:
        captured.append(dict(kwargs))
        await original_update(**kwargs)

    orch._face_controller.update = _spy  # type: ignore[method-assign]

    real_action = torch.tensor([0.1, 0.0, 0.0])
    await orch._update_face(safety_ctx=SafetyContext(is_emergency=False), action=real_action)

    assert len(captured) == 1, "update must have been called exactly once"
    assert captured[0]["is_idle"] is False


async def test_update_face_no_face_controller_is_noop() -> None:
    cfg = Settings(mock_hardware=True)
    orch = MouseDroidOrchestrator(
        world_model=MagicMock(),
        agents=[MagicMock()],
        safety_monitor=MagicMock(),
        esp32=AsyncMock(),
        sensor_manager=AsyncMock(),
        cfg=cfg,
        face_controller=None,
    )
    # Must complete without raising even with no face controller present.
    await orch._update_face(safety_ctx=SafetyContext(is_emergency=False), action=None)
