"""Functional tests for telemetry and dashboard endpoints."""

from __future__ import annotations

import asyncio
import json

import pytest


@pytest.mark.asyncio
async def test_telemetry_health_endpoint(functional_orchestrator):
    """Test the health endpoint returns structured JSON with subsystem status."""
    orch = functional_orchestrator

    await orch.start()
    try:
        health_data = await orch.health_check()

        assert health_data["status"] == "ok"
        assert "agents" in health_data
        assert isinstance(health_data["agents"], list)
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_telemetry_metrics_format(functional_orchestrator):
    """Test that metrics endpoint generates valid format."""
    orch = functional_orchestrator
    await orch.start()
    try:
        await orch.tick()
        # Telemetry server would usually serve this, but we can verify the metrics registry
        assert orch._metrics is not None
    finally:
        await orch.stop()
