"""On-device incremental learning (Phase 6 WS2).

Exports the lightweight, torch-free DI surface (the :class:`OnDeviceLearner`
protocol and the :class:`OnDeviceUpdateResult` result type). Concrete learner
implementations (e.g. :class:`~mousedroid.learning.on_device.rssm_refiner.RSSMRefiner`)
are intentionally NOT re-exported here so importing the protocol never drags
torch into the process — import them directly from their own modules (mirrors
the factory import-decoupling norm).
"""

from __future__ import annotations

from mousedroid.learning.on_device.protocol import (
    OnDeviceLearner,
    OnDeviceUpdateResult,
)

__all__ = ["OnDeviceLearner", "OnDeviceUpdateResult"]
