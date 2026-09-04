"""AQA pins for the orchestrator shutdown contract: the motors always stop.

``_lifecycle_mixin.stop()`` used to issue ``esp32.emergency_stop()`` roughly
twenty statements downstream of ``_drain_background_tasks()``, in a flat
sequence with no ``try``. Any raise in between — a drain timeout, a wedged
voice engine, a telemetry server that would not close — skipped the emergency
stop entirely and left the last commanded velocity latched on the controller.

The drain timeout was not hypothetical: ``cancel_and_drain`` caught a bare
``TimeoutError``, which on the Python 3.10 floor the Jetson image ships is a
*distinct* class from the ``asyncio.TimeoutError`` that ``asyncio.wait_for``
actually raises, so the handler was dead code and the timeout propagated.

These tests pin the structural guarantee rather than that one bug: whatever
fails during software teardown, the actuators are halted.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.config.schema import Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle


def _make_orchestrator(cfg: Settings) -> MouseDroidOrchestrator:
    """Build an orchestrator whose collaborators are all controllable mocks."""
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

    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(1024, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )
    sensor_manager.recovery_attempt.return_value = 0

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=AsyncMock(),
        sensor_manager=sensor_manager,
        cfg=cfg,
    )


async def test_emergency_stop_still_issued_when_software_teardown_raises() -> None:
    """A raise during software teardown must not skip the emergency stop.

    Fails against the pre-fix flat sequence: the exception propagated out of
    ``stop()`` and ``esp32.emergency_stop`` was never awaited.
    """
    orch = _make_orchestrator(Settings(mock_hardware=True))

    # Any subsystem that raises on the way down; the LLM gateway is the last
    # software step before the actuators in the pre-fix ordering, so it is the
    # tightest possible reproduction of the original gap.
    gateway = AsyncMock()
    gateway.stop.side_effect = RuntimeError("gateway wedged on shutdown")
    orch._llm_gateway = gateway

    with pytest.raises(RuntimeError, match="gateway wedged"):
        await orch.stop()

    orch._esp32.emergency_stop.assert_awaited_once()
    orch._esp32.disconnect.assert_awaited_once()


async def test_emergency_stop_still_issued_when_the_drain_raises() -> None:
    """A raise from ``_drain_background_tasks`` must not skip the stop either.

    This is the original 3.10 failure mode, reproduced at the seam rather than
    by pinning interpreter-specific exception identity.
    """
    orch = _make_orchestrator(Settings(mock_hardware=True))
    orch._drain_background_tasks = AsyncMock(
        side_effect=TimeoutError("drain exceeded its deadline")
    )

    with pytest.raises(TimeoutError):
        await orch.stop()

    orch._esp32.emergency_stop.assert_awaited_once()


async def test_actuator_halt_continues_past_an_emergency_stop_failure() -> None:
    """A failing emergency stop must not prevent the transports being released.

    ``_halt_actuators`` is deliberately best-effort per step: a controller that
    will not accept the stop frame is exactly the situation in which closing
    the serial port still matters.
    """
    orch = _make_orchestrator(Settings(mock_hardware=True))
    orch._esp32.emergency_stop.side_effect = OSError("serial port vanished")

    await orch.stop()

    orch._esp32.emergency_stop.assert_awaited_once()
    orch._sensor_manager.stop.assert_awaited_once()
    orch._esp32.disconnect.assert_awaited_once()


async def test_clean_shutdown_still_halts_actuators_exactly_once() -> None:
    """The happy path is unchanged: one stop, one sensor stop, one disconnect.

    Backwards-compatibility half of the pin — the restructure must not change
    observable behaviour when nothing fails.
    """
    orch = _make_orchestrator(Settings(mock_hardware=True))

    await orch.stop()

    orch._esp32.emergency_stop.assert_awaited_once()
    orch._sensor_manager.stop.assert_awaited_once()
    orch._esp32.disconnect.assert_awaited_once()
    assert orch._running is False
