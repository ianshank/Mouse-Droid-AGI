"""Factory builders — external agent adapters.

Builds mission decomposers, memory mirrors, and cloud tool adapters
from the config.agents section. All return None defensively when
disabled or when SDKs are missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings
    from mousedroid.harness.journal.protocol import JournalProtocol
    from mousedroid.interfaces.cloud_tool_adapter import CloudToolAdapterProtocol
    from mousedroid.interfaces.memory_mirror import MemoryMirrorProtocol
    from mousedroid.interfaces.mission_decomposer import MissionDecomposerProtocol
    from mousedroid.security.injection_filter import PromptInjectionFilterProtocol

_log = get_logger(__name__)


def build_mission_decomposer(
    cfg: Settings,
    *,
    injection_filter: PromptInjectionFilterProtocol | None = None,
) -> MissionDecomposerProtocol | None:
    """Build the ADK-backed mission decomposer.

    Returns None if `cfg.agents` is None, or if `cfg.agents.adk.enabled` is False.
    """
    if not cfg.agents or not cfg.agents.adk.enabled:
        _log.debug("adk_decomposer_disabled")
        return None

    from mousedroid.agents.adk_adapter import ADKMissionAdapter

    adapter = ADKMissionAdapter(cfg.agents.adk, injection_filter=injection_filter)
    _log.info("mission_decomposer_built", backend="adk")
    return adapter


def build_memory_mirror(
    cfg: Settings,
    *,
    journal: JournalProtocol,
    injection_filter: PromptInjectionFilterProtocol | None = None,
) -> MemoryMirrorProtocol | None:
    """Build the Honcho-backed memory mirror.

    Returns None if `cfg.agents` is None, or if `cfg.agents.honcho.enabled` is False.
    """
    if not cfg.agents or not cfg.agents.honcho.enabled:
        _log.debug("honcho_mirror_disabled")
        return None

    from mousedroid.memory.honcho_mirror import HonchoMemoryMirror

    mirror = HonchoMemoryMirror(
        cfg.agents.honcho,
        journal=journal,
        injection_filter=injection_filter,
    )
    _log.info("memory_mirror_built", backend="honcho")
    return mirror


def build_cloud_tool_adapter(
    cfg: Settings,
) -> CloudToolAdapterProtocol | None:
    """Build the Composio-backed cloud tool adapter.

    Returns None if `cfg.agents` is None, or if `cfg.agents.composio.enabled` is False.
    """
    if not cfg.agents or not cfg.agents.composio.enabled:
        _log.debug("composio_adapter_disabled")
        return None

    from mousedroid.agents.composio_adapter import ComposioToolAdapter

    adapter = ComposioToolAdapter(cfg.agents.composio)
    _log.info("cloud_tool_adapter_built", backend="composio")
    return adapter
