"""Integration tests for memory/curiosity pipeline through orchestrator.

Tests the full path: tick → experience logged → episodic → consolidation → semantic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import torch

from mousedroid.config.schema import MemoryConfig, Settings
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.memory.consolidation import MemoryConsolidation
from mousedroid.memory.episodic import EpisodicReplay
from mousedroid.memory.semantic import SemanticIndex
from mousedroid.memory.tier import MemoryTier
from mousedroid.memory.working import WorkingMemory
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle


def _make_memory_tier(
    *,
    episodic_capacity: int = 100,
    semantic_dim: int = 256,
    working_context: int = 16,
    consolidation_batch: int = 4,
) -> MemoryTier:
    """Build a real MemoryTier for integration testing."""
    cfg = MemoryConfig(
        enabled=True,
        episodic_capacity=episodic_capacity,
        semantic_dim=semantic_dim,
        working_context_size=working_context,
        consolidation_batch_size=consolidation_batch,
    )
    episodic = EpisodicReplay(cfg)
    semantic = SemanticIndex(cfg)
    working = WorkingMemory(cfg, embed_dim=semantic_dim)
    consolidation = MemoryConsolidation(cfg, episodic, semantic)
    return MemoryTier(
        episodic=episodic,
        semantic=semantic,
        working=working,
        consolidation=consolidation,
    )


def _make_observation(cfg: Settings) -> MouseDroidObservationBundle:
    """Create a default observation bundle for testing."""
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.random.default_rng(42)
        .standard_normal(cfg.camera.feature_dim)
        .astype(np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(1024, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def _make_orchestrator_with_memory(
    *,
    memory_tier: MemoryTier | None = None,
    experience_logger: object | None = None,
    curiosity_module: object | None = None,
) -> MouseDroidOrchestrator:
    """Create orchestrator with real memory tier."""
    cfg = Settings(mock_hardware=True)

    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )

    agent = MagicMock()
    agent.name = "test_agent"
    agent.act.return_value = torch.tensor([0.1, 0.0, 0.0])

    safety_ctx = SafetyContext()
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = safety_ctx

    esp32 = AsyncMock()
    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _make_observation(cfg)
    sensor_manager.recovery_attempt.return_value = 0

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        memory_tier=memory_tier,
        experience_logger=experience_logger,
        curiosity_module=curiosity_module,
    )


# ---------------------------------------------------------------------------
# Experience logging integration
# ---------------------------------------------------------------------------


async def test_tick_logs_to_episodic_replay() -> None:
    """After a tick, episodic replay has at least one experience."""
    tier = _make_memory_tier()
    orch = _make_orchestrator_with_memory(memory_tier=tier)
    assert len(tier.episodic) == 0

    await orch.tick()

    assert len(tier.episodic) >= 1


async def test_multiple_ticks_accumulate_experiences() -> None:
    """Multiple ticks accumulate experiences in episodic replay."""
    tier = _make_memory_tier()
    orch = _make_orchestrator_with_memory(memory_tier=tier)

    for _ in range(10):
        await orch.tick()

    assert len(tier.episodic) == 10


async def test_tick_pushes_to_working_memory() -> None:
    """After a tick, working memory has a latent state."""
    tier = _make_memory_tier()
    orch = _make_orchestrator_with_memory(memory_tier=tier)

    await orch.tick()

    assert len(tier.working) >= 1


async def test_tick_with_experience_logger() -> None:
    """Tick calls experience logger.log() when logger is present."""
    logger = MagicMock()
    orch = _make_orchestrator_with_memory(experience_logger=logger)

    await orch.tick()

    logger.log.assert_called_once()
    record = logger.log.call_args[0][0]
    assert isinstance(record, MouseDroidExperienceRecord)


async def test_tick_with_curiosity_computes_surprise() -> None:
    """Tick with curiosity module sets surprise on experience record."""
    tier = _make_memory_tier()
    curiosity = MagicMock()
    curiosity.intrinsic_reward.return_value = torch.tensor([0.75])

    orch = _make_orchestrator_with_memory(
        memory_tier=tier,
        curiosity_module=curiosity,
    )

    await orch.tick()

    # Curiosity module should have been called
    curiosity.intrinsic_reward.assert_called()
    # Episodic should have experience with non-zero priority
    assert len(tier.episodic) == 1


# ---------------------------------------------------------------------------
# Consolidation pipeline integration
# ---------------------------------------------------------------------------


def test_consolidation_episodic_to_semantic() -> None:
    """Consolidation moves episodic experiences into semantic index."""
    tier = _make_memory_tier(consolidation_batch=3, semantic_dim=256)

    # Add records with embeddings to episodic
    for i in range(5):
        record = MouseDroidExperienceRecord(
            vision_features=np.random.default_rng(i).standard_normal(256).astype(np.float32),
        )
        tier.episodic.push(record, priority=1.0)

    assert tier.semantic.size == 0

    count = tier.consolidation.consolidate()

    assert count > 0
    assert tier.semantic.size > 0


def test_consolidation_respects_batch_size() -> None:
    """Consolidation processes at most batch_size records per cycle."""
    tier = _make_memory_tier(consolidation_batch=2, semantic_dim=256)

    for i in range(10):
        record = MouseDroidExperienceRecord(
            vision_features=np.random.default_rng(i).standard_normal(256).astype(np.float32),
        )
        tier.episodic.push(record, priority=1.0)

    count = tier.consolidation.consolidate()
    assert count <= 2


def test_semantic_retrieval_after_consolidation() -> None:
    """After consolidation, semantic index can retrieve similar experiences."""
    tier = _make_memory_tier(semantic_dim=256)

    # Add similar records
    rng = np.random.default_rng(42)
    base_features = rng.standard_normal(256).astype(np.float32)
    for _i in range(5):
        noise = rng.standard_normal(256).astype(np.float32) * 0.01
        record = MouseDroidExperienceRecord(
            vision_features=base_features + noise,
        )
        tier.episodic.push(record, priority=1.0)

    tier.consolidation.consolidate()

    # Query with similar vector should find results
    results = tier.semantic.retrieve(base_features, k=1)
    assert len(results) > 0


# ---------------------------------------------------------------------------
# Consolidation loop lifecycle
# ---------------------------------------------------------------------------


async def test_consolidation_loop_starts_on_start() -> None:
    """start() creates consolidation background task when memory tier is present."""
    tier = _make_memory_tier()
    orch = _make_orchestrator_with_memory(memory_tier=tier)

    await orch.start()
    assert orch._consolidation_task is not None
    assert not orch._consolidation_task.done()

    await orch.stop()
    assert orch._consolidation_task is None


async def test_consolidation_loop_not_started_without_memory() -> None:
    """start() does not create consolidation task when memory tier is None."""
    orch = _make_orchestrator_with_memory()

    await orch.start()
    assert orch._consolidation_task is None

    await orch.stop()


async def test_stop_cancels_consolidation_task() -> None:
    """stop() cancels the consolidation background task cleanly."""
    tier = _make_memory_tier()
    orch = _make_orchestrator_with_memory(memory_tier=tier)

    await orch.start()
    task = orch._consolidation_task
    assert task is not None

    await orch.stop()
    assert task.cancelled() or task.done()
    assert orch._consolidation_task is None


# ---------------------------------------------------------------------------
# Experience logger lifecycle
# ---------------------------------------------------------------------------


async def test_experience_logger_opened_on_start() -> None:
    """start() opens the experience logger."""
    logger = MagicMock()
    orch = _make_orchestrator_with_memory(experience_logger=logger)

    await orch.start()
    logger.open.assert_called_once()

    await orch.stop()


async def test_experience_logger_closed_on_stop() -> None:
    """stop() closes the experience logger."""
    logger = MagicMock()
    orch = _make_orchestrator_with_memory(experience_logger=logger)

    await orch.start()
    await orch.stop()
    logger.close.assert_called_once()


# ---------------------------------------------------------------------------
# No-memory backward compatibility
# ---------------------------------------------------------------------------


async def test_tick_without_memory_or_logger() -> None:
    """Tick works normally without memory tier or experience logger."""
    orch = _make_orchestrator_with_memory()

    await orch.tick()

    # Normal tick behavior preserved
    orch._esp32.send_velocity.assert_awaited_once()
