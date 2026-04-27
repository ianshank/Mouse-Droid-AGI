"""Vision-Language-Action policy package (Phase 3a).

Provides a minimal ``VLAPolicyProtocol`` and a ``MockVLA`` implementation.
The orchestrator selects between the legacy navigation-agent path and the
VLA path via :attr:`mousedroid.config.schema.LoopConfig.policy_selector`.

Defaults preserve the existing nav-agent behavior (``policy_selector =
"nav_agent"``); the VLA branch is fully opt-in.
"""

from __future__ import annotations

from mousedroid.vla.policy import (
    MockVLA,
    VLAAction,
    VLAObservation,
    VLAPolicyProtocol,
)

__all__ = [
    "MockVLA",
    "VLAAction",
    "VLAObservation",
    "VLAPolicyProtocol",
]
