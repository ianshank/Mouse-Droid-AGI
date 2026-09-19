"""Safety context — frozen, immutable safety state passed to agents.

SafetyContext is created fresh each step by the safety monitor.
Agents extract relevant fields only. All fields have safe defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyContext:
    """Platform-agnostic safety context passed from monitors to agents.

    All fields have safe defaults. Agents extract relevant fields only.
    """

    # Shared
    surprise: float = 0.0
    valid_sensor_count: int = 0
    loop_time_ms: float = 0.0

    # Mouse Droid specific
    ultrasonic_dist_m: float = math.inf
    forward_clearance_ok: bool = True
    battery_voltage: float = 12.0
    gpu_temp_c: float = 0.0
    esp32_connected: bool = True

    # Three Laws of Robotics context
    human_detected: bool = False
    human_dist_m: float = math.inf
    human_needs_help: bool = False
    commanded_action: tuple[float, ...] | None = None
    law_violations: tuple[str, ...] = ()

    # LiDAR 360-degree obstacle awareness
    lidar_min_dist_m: float = math.inf
    lidar_clearance_ok: bool = True

    # Computed
    is_emergency: bool = False

    # Emergency latch (peer review D-5). ``is_emergency`` stays the single
    # field every consumer gates motion on and is now
    # ``live_emergency or latched``. This one answers a different question --
    # "does clearing this need a human?" -- for the dashboard, the face
    # controller and the boot log.
    emergency_latched: bool = False
