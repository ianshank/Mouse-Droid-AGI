"""Sub-second sanity smoke for the S-2 LiDAR fail-closed path.

Deliberately import-and-parse only -- no torch, no factory, no orchestrator --
so this stays a fast canary that the schema fields exist, the shipped rover
config still parses with them, and the monitor's absent-LiDAR branch is
reachable. The behavioural depth lives in the unit, property, integration, e2e
and regression tiers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from mousedroid.config.schema import SafetyConfig, Settings
from mousedroid.safety.monitor import LIDAR_UNAVAILABLE_DIST_M, MouseDroidSafetyMonitor
from mousedroid.sensing.bundle import MouseDroidObservationBundle

_JETSON_PRODUCTION = Path(__file__).resolve().parents[2] / "config" / "jetson_production.yaml"


def test_schema_exposes_the_fail_closed_knobs() -> None:
    assert "lidar_unavailable_policy" in SafetyConfig.model_fields
    assert "lidar_unavailable_grace_s" in SafetyConfig.model_fields


def test_jetson_production_still_parses() -> None:
    """The rover's own overlay must keep loading after the schema grew."""
    with _JETSON_PRODUCTION.open(encoding="utf-8") as fh:
        cfg = Settings.model_validate(yaml.safe_load(fh))
    assert cfg.safety.lidar_unavailable_policy == "ignore"
    # The override that makes S-2 sharp on this rig: one healthy sensor is
    # enough to keep driving, so the valid-mask backstop cannot be relied on.
    assert cfg.safety.min_valid_sensors == 1


def test_absent_lidar_branch_is_reachable() -> None:
    """One armed evaluate, to catch an import- or wiring-level break early."""
    monitor = MouseDroidSafetyMonitor(SafetyConfig(lidar_unavailable_policy="emergency"))
    obs = MouseDroidObservationBundle(
        _timestamp=0.0,
        _distance_m=2.0,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _lidar_features=None,
        _valid_mask=np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )
    ctx = monitor.evaluate(obs, loop_time_ms=10.0)
    assert ctx.lidar_min_dist_m == LIDAR_UNAVAILABLE_DIST_M
    assert ctx.is_emergency is True
