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


def build_safety_monitor(cfg: Settings) -> SafetyMonitorProtocol:
    """Build safety monitor for configured platform.

    Args:
        cfg: Root settings.

    Returns:
        Safety monitor conforming to ``SafetyMonitorProtocol``.
    """
    from mousedroid.safety.monitor import MouseDroidSafetyMonitor

    return MouseDroidSafetyMonitor(
        cfg.safety,
        human_presence=build_human_presence_detector(cfg),
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
