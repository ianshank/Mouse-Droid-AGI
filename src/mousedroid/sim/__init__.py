"""Simulation backends for 4WD rover sim-to-real training.

This package hosts the Isaac Lab environment stub plus a NumPy-only mock
backend used by CI and unit tests. Concrete env classes are imported
lazily from factory builders so the package loads without GPU / Isaac
dependencies.
"""

from __future__ import annotations

from mousedroid.sim.protocols import (
    ROVER_CHASSIS_POSE_DIM,
    ROVER_IMU_DIM,
    ROVER_NUM_WHEELS,
    RoverEnvProtocol,
)

__all__ = [
    "ROVER_CHASSIS_POSE_DIM",
    "ROVER_IMU_DIM",
    "ROVER_NUM_WHEELS",
    "RoverEnvProtocol",
]
