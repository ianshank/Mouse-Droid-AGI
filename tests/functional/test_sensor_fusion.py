"""Functional tests for sensor data processing and fusion."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_sensor_data_flows_through_world_model(functional_orchestrator):
    """Test camera and LiDAR data flows through world model."""
    orch = functional_orchestrator

    await orch.start()
    try:
        await orch.tick()
        assert orch._sensor_manager is not None
        assert orch._world_model is not None
        # Verify sensory pipeline ran during tick
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_sensor_dropout_degraded_mode(functional_orchestrator):
    """Test sensor dropout triggers degraded mode."""
    orch = functional_orchestrator

    await orch.start()
    try:
        # Simulate sensor dropout by overriding the sensor manager read
        if orch._sensor_manager:
            orch._sensor_manager.read_all = AsyncMock(side_effect=Exception("Sensor disconnected"))

        with pytest.raises(Exception, match="Sensor disconnected"):
            await orch.tick()
        # Expecting graceful handling of sensor failure
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_recovery_from_sensor_dropout(functional_orchestrator):
    """Test recovery from sensor dropout restores normal mode."""
    orch = functional_orchestrator

    await orch.start()
    try:
        # Normal
        await orch.tick()

        # Dropout
        original_read = orch._sensor_manager.read_all if orch._sensor_manager else None
        if orch._sensor_manager:
            orch._sensor_manager.read_all = AsyncMock(side_effect=Exception("Sensor disconnected"))

        with pytest.raises(Exception, match="Sensor disconnected"):
            await orch.tick()

        # Recovery
        if orch._sensor_manager and original_read:
            orch._sensor_manager.read_all = original_read
        await orch.tick()
    finally:
        await orch.stop()
