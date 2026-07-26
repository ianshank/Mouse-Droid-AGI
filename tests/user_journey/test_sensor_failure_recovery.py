"""Test user journey for sensor failure and recovery."""

from __future__ import annotations

import contextlib

import pytest

from mousedroid.factory import build_orchestrator


@pytest.mark.slow
@pytest.mark.asyncio
async def test_sensor_failure_recovery(user_journey_settings) -> None:
    """Test system degraded mode and recovery when a sensor fails.

    Steps:
    1. System starts normally, runs 3 ticks
    2. Sensor read starts returning errors
    3. System detects failure — tick may bail early or catch the error
    4. Sensor recovers
    5. System exits degraded mode, resumes normal operation with tick_count advancing
    """
    cfg = user_journey_settings
    orch = build_orchestrator(cfg)

    def _mock_imagine(action, h, z):
        import torch

        if action.dim() == 1:
            action = action.unsqueeze(0)
        return (
            torch.zeros(1, cfg.model.hidden_dim),
            torch.zeros(1, cfg.model.latent_dim),
            torch.tensor([[0.0]]),
        )

    orch._world_model.imagine_step = _mock_imagine

    await orch.start()
    try:
        # 1. Normal start and ticks
        for _ in range(3):
            await orch.tick()
        assert orch._tick_count == 3

        # Save original read method for recovery
        original_read = orch._sensor_manager.read_all

        async def failing_read(*args, **kwargs):
            raise RuntimeError("LiDAR connection lost")

        # 2. Sensor starts returning errors
        orch._sensor_manager.read_all = failing_read

        # 3. The orchestrator tick may either:
        #    (a) catch the error internally and increment tick_count, or
        #    (b) let it bubble up without incrementing tick_count.
        #    Either way, the system must not crash permanently.
        tick_before_failure = orch._tick_count
        with contextlib.suppress(RuntimeError):
            await orch.tick()

        # 4. Sensor recovers
        orch._sensor_manager.read_all = original_read

        # 5. After recovery, normal ticks resume successfully
        await orch.tick()
        await orch.tick()

        # Verify the system recovered: at least 2 successful ticks after
        # recovery means tick_count advanced past the pre-failure value.
        assert orch._tick_count >= tick_before_failure + 2, (
            f"Expected tick_count >= {tick_before_failure + 2}, "
            f"got {orch._tick_count}. System did not recover from sensor failure."
        )

    finally:
        await orch.stop()
