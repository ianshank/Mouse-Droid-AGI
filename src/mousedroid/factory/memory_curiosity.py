"""Factory builders — layered memory tier, experience logger, curiosity module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import (
        Settings,
    )
    from mousedroid.curiosity.protocol import CuriosityProtocol
    from mousedroid.experience.logger import ExperienceLogger
    from mousedroid.memory.tier import MemoryTier

_log = get_logger(__name__)


def build_memory_tier(cfg: Settings) -> MemoryTier | None:
    """Build all four memory subsystems if memory is enabled.

    Args:
        cfg: Root settings.

    Returns:
        ``MemoryTier`` dataclass or ``None`` if ``cfg.memory.enabled`` is False.
    """
    if not cfg.memory.enabled:
        return None

    from mousedroid.memory.consolidation import MemoryConsolidation
    from mousedroid.memory.episodic import EpisodicReplay
    from mousedroid.memory.semantic import SemanticIndex
    from mousedroid.memory.tier import MemoryTier
    from mousedroid.memory.working import WorkingMemory

    episodic = EpisodicReplay(cfg.memory, seed=cfg.memory.replay_seed)
    semantic = SemanticIndex(cfg.memory)
    working = WorkingMemory(cfg.memory, embed_dim=cfg.memory.semantic_dim)
    consolidation = MemoryConsolidation(cfg.memory, episodic, semantic)

    _log.info(
        "memory_tier_built",
        episodic_capacity=cfg.memory.episodic_capacity,
        semantic_dim=cfg.memory.semantic_dim,
        working_context=cfg.memory.working_context_size,
    )
    return MemoryTier(
        episodic=episodic,
        semantic=semantic,
        working=working,
        consolidation=consolidation,
    )


def build_experience_logger(cfg: Settings) -> ExperienceLogger | None:
    """Build LMDB experience logger if memory is enabled and experience config is present.

    Args:
        cfg: Root settings.

    Returns:
        ``ExperienceLogger`` or ``None`` if memory is disabled or experience config is absent.
    """
    if not cfg.memory.enabled:
        return None

    experience_cfg = getattr(cfg, "experience", None)
    if experience_cfg is None:
        return None

    from mousedroid.experience.logger import ExperienceLogger

    logger = ExperienceLogger(experience_cfg)
    _log.info("experience_logger_built", path=experience_cfg.path)
    return logger


def build_curiosity_module(cfg: Settings) -> CuriosityProtocol | None:
    """Build intrinsic curiosity module if memory is enabled.

    Args:
        cfg: Root settings.

    Returns:
        ``IntrinsicCuriosityModule`` or ``None`` if memory is disabled.
    """
    if not cfg.memory.enabled:
        return None

    from mousedroid.curiosity.icm import IntrinsicCuriosityModule

    try:
        module = IntrinsicCuriosityModule(cfg.model, cfg.curiosity)
        _log.info("curiosity_module_built", scale=cfg.curiosity.intrinsic_reward_scale)
        return module
    except Exception:
        _log.warning("curiosity_module_build_failed", exc_info=True)
        return None
