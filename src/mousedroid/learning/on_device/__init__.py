"""On-device incremental learning (Phase 6 WS2).

Exports the lightweight, torch-free DI surface (the :class:`OnDeviceLearner`
protocol and the :class:`OnDeviceUpdateResult` result type). The concrete
``EWCOnlineLearner`` implementation is intentionally NOT re-exported here so
importing the protocol never drags torch into the process — import it directly
from :mod:`mousedroid.learning.on_device.ewc_online` (mirrors the factory
import-decoupling norm).
"""

from __future__ import annotations

from mousedroid.learning.on_device.protocol import (
    OnDeviceLearner,
    OnDeviceUpdateResult,
)

__all__ = ["OnDeviceLearner", "OnDeviceUpdateResult"]
