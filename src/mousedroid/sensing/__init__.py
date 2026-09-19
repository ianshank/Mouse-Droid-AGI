"""Sensor management and observation bundles."""

from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.sensing.human_presence import (
    HumanPresence,
    HumanPresenceProtocol,
    NullHumanPresenceDetector,
)
from mousedroid.sensing.manager import SensorManager
from mousedroid.sensing.protocol import ObservationProtocol

__all__ = [
    "HumanPresence",
    "HumanPresenceProtocol",
    "MouseDroidObservationBundle",
    "NullHumanPresenceDetector",
    "ObservationProtocol",
    "SensorManager",
]
