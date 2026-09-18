"""S-2 LiDAR fail-closed backwards-compatibility regression tests.

Pins CLAUDE.md invariant 6/9: new config fields MUST have defaults, and
existing YAML files must load unchanged after a ``git pull``.

The S-2 fix is deliberately split so that only the *truthfulness* half is
unconditional:

* ``SensorManager._safe_lidar_read`` no longer substitutes
  ``np.ones(feature_dim)`` for a failed read. That is a correctness fix, not a
  behaviour toggle -- the fabricated vector was already gated to zero in the
  world-model encoder by its valid-mask slot, and both 12.0 m and infinity sit
  above every projector clamp threshold, so no motion changes.
* Whether an absent LiDAR should *stop the rover* is gated behind
  ``SafetyConfig.lidar_unavailable_policy``, whose ``ignore`` default
  reproduces the pre-S-2 read-through exactly. These tests pin that default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from mousedroid.config.schema import SafetyConfig, Settings
from mousedroid.safety.monitor import MouseDroidSafetyMonitor
from mousedroid.sensing.bundle import MouseDroidObservationBundle

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

#: Every shipped overlay must still resolve to the pre-S-2 posture. If a future
#: change opts one of these in, that is a rover-behaviour decision and this
#: test is the place it has to be argued.
_SHIPPED_CONFIGS = [
    "default.yaml",
    "jetson_production.yaml",
    "jetson_lidar_only.yaml",
    "local_lidar_validation.yaml",
    "mock_hardware.yaml",
]


def test_policy_defaults_to_ignore_when_absent() -> None:
    """A pre-S-2 YAML has no ``lidar_unavailable_policy`` key at all."""
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.safety.lidar_unavailable_policy == "ignore"
    assert cfg.safety.lidar_unavailable_grace_s == 0.0


def test_legacy_safety_block_loads_unchanged() -> None:
    """A legacy ``safety:`` block still loads, with its own fields untouched.

    ``min_valid_sensors: 1`` is the production override that made S-2 sharp --
    one healthy sensor was enough to keep driving on a fabricated LiDAR
    all-clear. It must survive this change byte-for-byte.
    """
    legacy_yaml = """
    mock_hardware: true
    safety:
      min_valid_sensors: 1
      min_forward_clearance_m: 0.25
      lidar_max_range_m: 8.0
    """
    cfg = Settings.model_validate(yaml.safe_load(legacy_yaml))
    assert cfg.safety.lidar_unavailable_policy == "ignore"
    assert cfg.safety.lidar_unavailable_grace_s == 0.0
    assert cfg.safety.min_valid_sensors == 1
    assert cfg.safety.min_forward_clearance_m == 0.25
    assert cfg.safety.lidar_max_range_m == 8.0


def test_policy_round_trips_when_present() -> None:
    legacy_yaml = """
    mock_hardware: true
    safety:
      lidar_unavailable_policy: emergency
      lidar_unavailable_grace_s: 0.1
    """
    cfg = Settings.model_validate(yaml.safe_load(legacy_yaml))
    assert cfg.safety.lidar_unavailable_policy == "emergency"
    assert cfg.safety.lidar_unavailable_grace_s == pytest.approx(0.1)


@pytest.mark.parametrize("name", _SHIPPED_CONFIGS)
def test_shipped_config_still_loads_and_stays_fail_open(name: str) -> None:
    """Every committed overlay parses and keeps the pre-S-2 posture."""
    path = _CONFIG_DIR / name
    if not path.exists():  # pragma: no cover - safety net for a moved fixture
        pytest.skip(f"{name} not present")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    cfg = Settings.model_validate(data)
    assert cfg.safety.lidar_unavailable_policy == "ignore"


def _obs(lidar_features: np.ndarray | None) -> MouseDroidObservationBundle:
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _distance_m=2.0,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _lidar_features=lidar_features,
        _valid_mask=np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def test_default_policy_raises_no_new_emergency_stop() -> None:
    """Behavioural back-compat: the default must not e-stop an existing rig.

    A config-level default check is not enough on its own -- the field could
    default correctly while the monitor ignored it. This drives the monitor.
    """
    monitor = MouseDroidSafetyMonitor(SafetyConfig())
    for tick in range(100):
        ctx = monitor.evaluate(_obs(None), loop_time_ms=10.0)
        assert ctx.is_emergency is False, f"new emergency stop at tick {tick}"
        assert ctx.lidar_clearance_ok is True


def test_default_policy_leaves_a_measured_scan_untouched() -> None:
    """The present-features arithmetic is unchanged: min(features) * max_range."""
    monitor = MouseDroidSafetyMonitor(SafetyConfig())
    feats = np.full(36, 0.5, dtype=np.float32)
    ctx = monitor.evaluate(_obs(feats), loop_time_ms=10.0)
    assert ctx.lidar_min_dist_m == pytest.approx(0.5 * 12.0)
    assert ctx.lidar_clearance_ok is True
