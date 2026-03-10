"""Continual learning — EWC and progressive networks."""

from mousedroid.learning.ewc import EWCAgent
from mousedroid.learning.progressive import ProgressiveNetwork
from mousedroid.learning.protocol import ContinualLearnerProtocol

__all__ = [
    "ContinualLearnerProtocol",
    "EWCAgent",
    "ProgressiveNetwork",
]
