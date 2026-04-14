"""Tests for MemoryTier dataclass and build_memory_tier factory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema import MemoryConfig, Settings
from mousedroid.memory.tier import MemoryTier

# ---------------------------------------------------------------------------
# MemoryTier dataclass tests
# ---------------------------------------------------------------------------


def test_memory_tier_fields() -> None:
    """MemoryTier has all four required subsystems."""
    tier = MemoryTier(
        episodic=MagicMock(),
        semantic=MagicMock(),
        working=MagicMock(),
        consolidation=MagicMock(),
    )
    assert tier.episodic is not None
    assert tier.semantic is not None
    assert tier.working is not None
    assert tier.consolidation is not None


def test_memory_tier_is_frozen() -> None:
    """MemoryTier is immutable (frozen dataclass)."""
    tier = MemoryTier(
        episodic=MagicMock(),
        semantic=MagicMock(),
        working=MagicMock(),
        consolidation=MagicMock(),
    )
    with pytest.raises(AttributeError):
        tier.episodic = MagicMock()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MemoryConfig.enabled backward-compat tests
# ---------------------------------------------------------------------------


def test_memory_config_enabled_defaults_false() -> None:
    """MemoryConfig.enabled defaults to False for backward compatibility."""
    cfg = MemoryConfig()
    assert cfg.enabled is False


def test_memory_config_enabled_true() -> None:
    """MemoryConfig.enabled can be set to True."""
    cfg = MemoryConfig(enabled=True)
    assert cfg.enabled is True


def test_settings_default_memory_disabled() -> None:
    """Default Settings has memory.enabled=False."""
    cfg = Settings(mock_hardware=True)
    assert cfg.memory.enabled is False


# ---------------------------------------------------------------------------
# build_memory_tier factory tests
# ---------------------------------------------------------------------------


def test_build_memory_tier_disabled_returns_none() -> None:
    """build_memory_tier returns None when memory.enabled is False."""
    from mousedroid.factory import build_memory_tier

    cfg = Settings(mock_hardware=True)
    assert cfg.memory.enabled is False
    result = build_memory_tier(cfg)
    assert result is None


def test_build_memory_tier_enabled_returns_tier() -> None:
    """build_memory_tier returns MemoryTier when memory.enabled is True."""
    from mousedroid.factory import build_memory_tier

    cfg = Settings(mock_hardware=True, memory=MemoryConfig(enabled=True))
    result = build_memory_tier(cfg)
    assert result is not None
    assert isinstance(result, MemoryTier)


def test_build_memory_tier_components_configured() -> None:
    """build_memory_tier components respect config values."""
    from mousedroid.factory import build_memory_tier
    from mousedroid.memory.episodic import EpisodicReplay
    from mousedroid.memory.semantic import SemanticIndex
    from mousedroid.memory.working import WorkingMemory

    cfg = Settings(
        mock_hardware=True,
        memory=MemoryConfig(
            enabled=True,
            episodic_capacity=100,
            semantic_dim=64,
            working_context_size=16,
        ),
    )
    tier = build_memory_tier(cfg)
    assert isinstance(tier, MemoryTier)
    assert isinstance(tier.episodic, EpisodicReplay)
    assert isinstance(tier.semantic, SemanticIndex)
    assert isinstance(tier.working, WorkingMemory)


# ---------------------------------------------------------------------------
# build_experience_logger factory tests
# ---------------------------------------------------------------------------


def test_build_experience_logger_disabled_returns_none() -> None:
    """build_experience_logger returns None when memory.enabled is False."""
    from mousedroid.factory import build_experience_logger

    cfg = Settings(mock_hardware=True)
    assert cfg.memory.enabled is False
    result = build_experience_logger(cfg)
    assert result is None


def test_build_experience_logger_enabled_returns_logger() -> None:
    """build_experience_logger returns ExperienceLogger when memory.enabled is True."""
    from mousedroid.experience.logger import ExperienceLogger
    from mousedroid.factory import build_experience_logger

    cfg = Settings(mock_hardware=True, memory=MemoryConfig(enabled=True))
    result = build_experience_logger(cfg)
    assert isinstance(result, ExperienceLogger)


# ---------------------------------------------------------------------------
# build_curiosity_module factory tests
# ---------------------------------------------------------------------------


def test_build_curiosity_module_disabled_returns_none() -> None:
    """build_curiosity_module returns None when memory.enabled is False."""
    from mousedroid.factory import build_curiosity_module

    cfg = Settings(mock_hardware=True)
    assert cfg.memory.enabled is False
    result = build_curiosity_module(cfg)
    assert result is None


def test_build_curiosity_module_enabled_returns_icm() -> None:
    """build_curiosity_module returns ICM module when memory.enabled is True."""
    from mousedroid.curiosity.icm import IntrinsicCuriosityModule
    from mousedroid.factory import build_curiosity_module

    cfg = Settings(mock_hardware=True, memory=MemoryConfig(enabled=True))
    result = build_curiosity_module(cfg)
    assert isinstance(result, IntrinsicCuriosityModule)
