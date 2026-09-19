"""Factory builders — safety monitor and action projector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger
from mousedroid.safety.projector_protocol import SafetyActionProjectorProtocol
from mousedroid.safety.protocol import SafetyMonitorProtocol

if TYPE_CHECKING:
    from mousedroid.config.schema import (
        Settings,
    )
    from mousedroid.safety.latch import EmergencyLatchProtocol
    from mousedroid.sensing.human_presence import HumanPresenceProtocol
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)


def build_human_presence_detector(cfg: Settings) -> HumanPresenceProtocol:
    """Resolve the human-presence source for the safety interlocks (D-1).

    There is exactly one implementation today, and it detects nobody. The
    builder exists anyway so the choice is made here, behind a Protocol,
    rather than by a ``getattr`` on the observation -- which is what let
    the gap go unnoticed. Deliberately no ``Literal`` selector field until
    a second implementation exists, matching the discipline
    ``SafetyActionProjectorProtocol`` documents for ``projector.kind``.

    Args:
        cfg: Root settings. Unused today; taken so adding a real detector
            is a change to this function and nothing else.

    Returns:
        A source conforming to ``HumanPresenceProtocol``.
    """
    del cfg  # no selector yet -- see docstring
    from mousedroid.sensing.human_presence import NullHumanPresenceDetector

    detector = NullHumanPresenceDetector()
    _log.info(
        "human_presence_detector_built",
        source=detector.source_name,
        can_detect=detector.can_detect,
    )
    return detector


def build_emergency_latch(cfg: Settings) -> EmergencyLatchProtocol | None:
    """Build the emergency-stop latch, or ``None`` when disabled (D-5).

    Returns ``None`` unless ``cfg.safety.emergency_latch.enabled``, so the
    default deployment is byte-identical to pre-latch behaviour: no latch
    object, no directory, no file. Structurally the same shape as
    :func:`build_safety_projector`.

    The record lives under the experience root rather than an absolute
    path, because that root is a persistent Docker named volume on the
    container path and a real directory on bare metal -- so the latch
    survives a systemd restart and a ``docker compose`` restart alike,
    which are the two ways the defect reproduced.

    Args:
        cfg: Root settings.

    Returns:
        A latch conforming to ``EmergencyLatchProtocol``, or ``None``.
    """
    latch_cfg = cfg.safety.emergency_latch
    if not latch_cfg.enabled:
        _log.debug("emergency_latch_disabled")
        return None

    from pathlib import Path

    from mousedroid.safety.latch import FileEmergencyLatch

    state_dir = Path(cfg.experience.path) / latch_cfg.state_dir
    latch = FileEmergencyLatch(state_dir, fsync=latch_cfg.fsync)
    _log.info("emergency_latch_built", path=str(latch.path), fsync=latch_cfg.fsync)
    return latch


def build_safety_monitor(
    cfg: Settings,
    *,
    latch: EmergencyLatchProtocol | None = None,
) -> SafetyMonitorProtocol:
    """Build safety monitor for configured platform.

    Args:
        cfg: Root settings.
        latch: A prebuilt emergency latch to share. ``build_orchestrator``
            passes one so the *same* object reaches both the monitor (which
            trips it) and the orchestrator (which loads and persists it) --
            two latches would mean a trip that is never written. ``None``
            builds one from config, so standalone callers keep working.

    Returns:
        Safety monitor conforming to ``SafetyMonitorProtocol``.
    """
    from mousedroid.safety.monitor import MouseDroidSafetyMonitor

    return MouseDroidSafetyMonitor(
        cfg.safety,
        human_presence=build_human_presence_detector(cfg),
        latch=latch if latch is not None else build_emergency_latch(cfg),
    )


def build_safety_projector(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> SafetyActionProjectorProtocol | None:
    """Build the optional geometric safety action projector (Tier C2 / C2.1).

    Returns ``None`` when ``cfg.safety.projector.enabled`` is ``False`` —
    the orchestrator skips the projection seam entirely in that case, so
    pre-C2 deployments produce byte-identical actions.

    Args:
        cfg: Root settings.
        metrics: Optional shared metrics registry. When supplied, the
            projector increments ``mousedroid_safety_action_clamps_total``
            with one of ``forward_velocity`` / ``human_proximity`` /
            ``tight_quarters`` on every materially different clamp.

    Returns:
        :class:`SafetyActionProjectorProtocol` implementation when enabled,
        ``None`` otherwise.
    """
    if not cfg.safety.projector.enabled:
        _log.debug("safety_projector_disabled")
        return None

    from mousedroid.safety.projector import GeometricSafetyProjector

    _log.info("safety_projector_built", backend="geometric")
    return GeometricSafetyProjector(cfg.safety.projector, metrics=metrics)
