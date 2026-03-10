"""Scaling — Mixture of Experts and adaptive compute."""

from mousedroid.scaling.adaptive import AdaptiveCompute
from mousedroid.scaling.moe import SparseMoELayer
from mousedroid.scaling.protocol import ScalingProtocol

__all__ = [
    "AdaptiveCompute",
    "ScalingProtocol",
    "SparseMoELayer",
]
