"""Tests for MouseDroidOrchestrator's cloud subsystem lifecycle wiring.

Covers ``_start_cloud_subsystems``/``_stop_cloud_subsystems`` directly —
narrower and more robust than driving the full ``start()``/``stop()``
sequence, which would require mocking every other collaborator the
orchestrator owns. No prior test exercised these two methods at all (not
even for the pre-existing ``cloud_sink``/``cloud_experience_exporter``
pair), so this file covers all four cloud collaborators together.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from mousedroid.config.schema import Settings


def _make_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")
    return Settings(mock_hardware=True)


def _make_orchestrator(monkeypatch: pytest.MonkeyPatch, **cloud_kwargs: object):
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    cfg = _make_settings(monkeypatch)
    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        None,
        None,
    )
    agent = MagicMock()
    agent.name = "test_agent"
    safety = MagicMock()

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety,
        esp32=AsyncMock(),
        sensor_manager=AsyncMock(),
        cfg=cfg,
        **cloud_kwargs,
    )


async def test_start_cloud_subsystems_noop_when_all_unwired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None of the four collaborators wired -- start() touches nothing."""
    orch = _make_orchestrator(monkeypatch)
    await orch._start_cloud_subsystems()  # must not raise


async def test_stop_cloud_subsystems_noop_when_all_unwired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch = _make_orchestrator(monkeypatch)
    await orch._stop_cloud_subsystems()  # must not raise


async def test_start_cloud_subsystems_starts_all_four_when_wired_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []

    cloud_sink = AsyncMock()
    cloud_sink.start.side_effect = lambda: call_order.append("sink")
    cloud_experience_exporter = AsyncMock()
    cloud_experience_exporter.start.side_effect = lambda: call_order.append("experience_exporter")
    cloud_metrics_exporter = AsyncMock()
    cloud_metrics_exporter.start.side_effect = lambda: call_order.append("metrics_exporter")
    cloud_firestore_sync = AsyncMock()
    cloud_firestore_sync.start.side_effect = lambda: call_order.append("firestore_sync")
    orch = _make_orchestrator(
        monkeypatch,
        cloud_sink=cloud_sink,
        cloud_experience_exporter=cloud_experience_exporter,
        cloud_metrics_exporter=cloud_metrics_exporter,
        cloud_firestore_sync=cloud_firestore_sync,
    )

    await orch._start_cloud_subsystems()

    cloud_sink.start.assert_called_once()
    cloud_experience_exporter.start.assert_called_once()
    cloud_metrics_exporter.start.assert_called_once()
    cloud_firestore_sync.start.assert_called_once()
    assert call_order == ["sink", "experience_exporter", "metrics_exporter", "firestore_sync"]


async def test_stop_cloud_subsystems_stops_all_four_when_wired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each collaborator's real teardown method is called -- not a generic .close()."""
    cloud_sink = AsyncMock()
    cloud_experience_exporter = AsyncMock()
    cloud_metrics_exporter = AsyncMock()
    cloud_firestore_sync = AsyncMock()
    orch = _make_orchestrator(
        monkeypatch,
        cloud_sink=cloud_sink,
        cloud_experience_exporter=cloud_experience_exporter,
        cloud_metrics_exporter=cloud_metrics_exporter,
        cloud_firestore_sync=cloud_firestore_sync,
    )

    await orch._stop_cloud_subsystems()

    cloud_sink.flush.assert_called_once()
    cloud_sink.close.assert_called_once()
    cloud_experience_exporter.close.assert_called_once()
    # CloudMetricsExporterProtocol's teardown method is stop(), not close() --
    # the one asymmetry among the four cloud collaborators (design.md D-6).
    cloud_metrics_exporter.stop.assert_called_once()
    cloud_metrics_exporter.close.assert_not_called()
    cloud_firestore_sync.close.assert_called_once()


async def test_stop_cloud_subsystems_lifo_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Teardown order is the reverse of start order: firestore, then metrics,

    then experience_exporter, then sink.
    """
    call_order: list[str] = []

    cloud_sink = AsyncMock()
    cloud_sink.close.side_effect = lambda: call_order.append("sink")
    cloud_experience_exporter = AsyncMock()
    cloud_experience_exporter.close.side_effect = lambda: call_order.append("experience_exporter")
    cloud_metrics_exporter = AsyncMock()
    cloud_metrics_exporter.stop.side_effect = lambda: call_order.append("metrics_exporter")
    cloud_firestore_sync = AsyncMock()
    cloud_firestore_sync.close.side_effect = lambda: call_order.append("firestore_sync")

    orch = _make_orchestrator(
        monkeypatch,
        cloud_sink=cloud_sink,
        cloud_experience_exporter=cloud_experience_exporter,
        cloud_metrics_exporter=cloud_metrics_exporter,
        cloud_firestore_sync=cloud_firestore_sync,
    )

    await orch._stop_cloud_subsystems()

    assert call_order == [
        "firestore_sync",
        "metrics_exporter",
        "experience_exporter",
        "sink",
    ]
