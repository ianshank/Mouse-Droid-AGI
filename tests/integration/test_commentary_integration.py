"""Integration tests: commentary factory wiring + orchestrator lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import torch

from mousedroid.commentary.facts import extract_commentary_facts
from mousedroid.commentary.protocol import CommentaryEngineProtocol
from mousedroid.config.schema import CommentaryConfig, MetricsConfig, Settings
from mousedroid.constants import DEFAULT_AUDIO_CHUNK_SIZE, DEFAULT_BATTERY_VOLTAGE
from mousedroid.factory import build_commentary
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.voice.protocol import VoiceEngineProtocol


# --------------------------------------------------------------------------- #
# Factory -> engine -> voice fire (full factory wiring through a mock voice)
# --------------------------------------------------------------------------- #
async def test_factory_engine_fires_through_voice() -> None:
    cfg = Settings(
        mock_hardware=True,
        commentary=CommentaryConfig(
            enabled=True, composer="template", allow_without_novelty=True, min_interval_s=0.0
        ),
    )
    reg = MetricsRegistry(MetricsConfig())
    voice = AsyncMock(spec=VoiceEngineProtocol)
    voice.play_phrase = AsyncMock(return_value=(100, 0.5))
    engine = build_commentary(cfg, voice_engine=voice, metrics=reg)
    assert isinstance(engine, CommentaryEngineProtocol)

    obs = MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=5.0,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(DEFAULT_AUDIO_CHUNK_SIZE, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        _lidar_features=np.array([5.0, 6.0, 7.0], dtype=np.float32),
    )
    facts = extract_commentary_facts(obs, novelty=None, is_emergency=False)
    engine.observe(None, facts)
    await engine._evaluate_and_speak()  # type: ignore[attr-defined]

    assert voice.play_phrase.await_count == 1
    assert "commentary_emitted_total 1" in reg.render_prometheus()


# --------------------------------------------------------------------------- #
# Orchestrator wiring: spawn on start, observe per tick, drain on stop
# --------------------------------------------------------------------------- #
def _orchestrator_with_commentary(commentary: object) -> MouseDroidOrchestrator:
    cfg = Settings(
        mock_hardware=True,
        commentary=CommentaryConfig(enabled=True, composer="template", observe_stride=1),
    )
    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim),
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
        _distance_m=5.0,
        _motor_state=np.array([0.0, 0.0, 0.0, DEFAULT_BATTERY_VOLTAGE], dtype=np.float32),
        _audio_chunk=np.zeros(DEFAULT_AUDIO_CHUNK_SIZE, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        _lidar_features=np.array([5.0, 6.0, 7.0], dtype=np.float32),
    )
    sensor_manager.recovery_attempt.return_value = 0

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=AsyncMock(),
        sensor_manager=sensor_manager,
        cfg=cfg,
        commentary=commentary,  # type: ignore[arg-type]
    )


async def test_orchestrator_spawns_and_drains_commentary_task() -> None:
    commentary = AsyncMock(spec=CommentaryEngineProtocol)
    orch = _orchestrator_with_commentary(commentary)
    await orch.start()
    commentary.run.assert_called_once()  # background loop spawned
    await orch.stop()
    commentary.stop.assert_awaited_once()  # drained on shutdown


async def test_orchestrator_tick_feeds_commentary() -> None:
    commentary = MagicMock(spec=CommentaryEngineProtocol)
    orch = _orchestrator_with_commentary(commentary)
    await orch.tick()
    # Every-tick emergency stamp + strided observe (stride=1, tick 0).
    commentary.observe_emergency.assert_called_with(False)
    commentary.observe.assert_called_once()
    novelty_arg, facts_arg = commentary.observe.call_args.args
    assert novelty_arg is None  # no curiosity module wired
    assert facts_arg.min_clearance_m == 5.0
