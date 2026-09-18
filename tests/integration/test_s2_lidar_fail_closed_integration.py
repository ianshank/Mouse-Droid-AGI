"""Integration: a failed LiDAR read fails CLOSED from sensing through safety.

Finding S-2 of the autonomy baseline peer review. The unit tiers cover the two
halves separately -- ``SensorManager._safe_lidar_read`` no longer fabricating an
all-ones feature vector, and ``MouseDroidSafetyMonitor`` honouring
``SafetyConfig.lidar_unavailable_policy``. This tier wires a real
``SensorManager`` to a **factory-built** monitor so the whole path runs
together, and so the YAML key is proved to reach the monitor: direct
construction of ``SafetyConfig`` cannot show that ``safety:
lidar_unavailable_policy:`` in a config file lands on the object the
orchestrator actually uses.

The pre-fix behaviour this pins against: features are normalised range
fractions (``min_in_sector / max_range``), so the all-ones substitute meant
"maximum range in every sector". A dead LiDAR reported ``lidar_max_range_m``
(12.0 m by default) of clearance in every direction, ``lidar_clearance_ok``
stayed ``True``, and with ``min_valid_sensors: 1`` in
``config/jetson_production.yaml`` one surviving sensor was enough to keep
driving.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest
import yaml

from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.schema import Settings
from mousedroid.factory.safety import build_safety_monitor
from mousedroid.hardware.lidar.feature_extractor import LidarFeatureExtractor
from mousedroid.hardware.lidar.mock_lidar import MockLidar
from mousedroid.sensing.manager import SensorManager

#: One tick is enough. ``lidar_unavailable_grace_s`` defaults to 0.0 and the
#: window is exclusive, so a zero grace grants no free tick -- which also keeps
#: this test independent of the host monotonic clock's granularity.
_TICKS = 1


def _settings_from_yaml(policy: str | None) -> Settings:
    """Load ``Settings`` the way a deployment does -- through YAML text."""
    safety_block = "" if policy is None else f"\n  lidar_unavailable_policy: {policy}"
    raw = f"""
mock_hardware: true
lidar:
  enabled: true
  n_sectors: 36
  feature_dim: 36
safety:
  sensor_stale_s: 100.0{safety_block}
"""
    return Settings.model_validate(yaml.safe_load(raw))


def _make_manager(cfg: Settings, *, lidar_raises: bool) -> tuple[SensorManager, MockLidar]:
    """Real ``SensorManager`` + real ``MockLidar`` + real feature extractor."""
    assert cfg.lidar is not None
    lidar = MockLidar(cfg.lidar)
    if lidar_raises:
        lidar.read_scan = AsyncMock(side_effect=RuntimeError("LD19 serial gone"))  # type: ignore[method-assign]

    vision = AsyncMock()
    vision.capture_features = AsyncMock(
        return_value=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
    )
    distance = AsyncMock()
    distance.read_distance_m = AsyncMock(return_value=2.0)
    distance.max_range_m = 4.0

    esp32 = AsyncMock()
    esp32.read_encoders = AsyncMock(return_value=EncoderReading())
    esp32.get_battery_voltage = AsyncMock(return_value=12.0)

    manager = SensorManager(
        vision,
        distance,
        esp32,
        cfg,
        lidar=lidar,
        lidar_feature_extractor=LidarFeatureExtractor(cfg.lidar),
    )
    return manager, lidar


async def _drive(cfg: Settings, *, lidar_raises: bool, ticks: int = _TICKS):
    """Run ``ticks`` sense-then-evaluate cycles, returning the last context."""
    manager, _lidar = _make_manager(cfg, lidar_raises=lidar_raises)
    monitor = build_safety_monitor(cfg)
    ctx = None
    for _ in range(ticks):
        bundle = await manager.read_all()
        ctx = monitor.evaluate(bundle, loop_time_ms=10.0)
    assert ctx is not None
    return ctx


async def test_dead_lidar_no_longer_fabricates_full_clearance() -> None:
    """Default policy: the 12 m fabrication is gone even without opting in.

    ``ignore`` keeps the read-through *behaviour* (no new emergency stop, so
    existing deployments are unchanged), but the sensing layer no longer
    manufactures the distance that made the read-through dangerous. Infinity
    here is the monitor's own "not measured" sentinel, not a measurement.
    """
    cfg = _settings_from_yaml(None)
    ctx = await _drive(cfg, lidar_raises=True)

    assert ctx.lidar_min_dist_m != 12.0, "the fabricated all-clear distance is back"
    assert ctx.lidar_min_dist_m == float("inf")
    assert ctx.is_emergency is False


async def test_dead_lidar_emergency_policy_stops_the_rover() -> None:
    """``emergency`` in YAML reaches the factory-built monitor and halts."""
    cfg = _settings_from_yaml("emergency")
    assert cfg.safety.lidar_unavailable_policy == "emergency"

    ctx = await _drive(cfg, lidar_raises=True)

    assert ctx.lidar_clearance_ok is False
    assert ctx.lidar_min_dist_m == 0.0
    assert ctx.is_emergency is True


async def test_dead_lidar_degrade_policy_brakes_without_stopping() -> None:
    """``degrade`` hands the projector a worst-case clearance, no e-stop."""
    cfg = _settings_from_yaml("degrade")
    ctx = await _drive(cfg, lidar_raises=True)

    assert ctx.lidar_clearance_ok is False
    assert ctx.lidar_min_dist_m == 0.0
    assert ctx.is_emergency is False
    # Below both projector clamp thresholds, so motion is throttled to a crawl.
    assert ctx.lidar_min_dist_m < cfg.safety.projector.lidar_brake_distance_m
    assert ctx.lidar_min_dist_m < cfg.safety.projector.tight_quarters_dist_m


@pytest.mark.parametrize("policy", ["ignore", "degrade", "emergency"])
async def test_healthy_lidar_is_unaffected_by_any_policy(policy: str) -> None:
    """A working LD19 produces real features, so the policy never engages."""
    cfg = _settings_from_yaml(policy)
    ctx = await _drive(cfg, lidar_raises=False)

    assert ctx.lidar_clearance_ok is True
    assert ctx.is_emergency is False
    assert 0.0 < ctx.lidar_min_dist_m < float("inf")


async def test_valid_mask_still_marks_the_lidar_slot_invalid() -> None:
    """The mask contract is unchanged -- this fix is additive to it.

    The mask always reported the failure correctly; the bug was that the
    *features* contradicted it and the monitor read the features.
    """
    cfg = _settings_from_yaml("emergency")
    manager, _lidar = _make_manager(cfg, lidar_raises=True)

    bundle = await manager.read_all()

    assert bundle.valid_mask.shape == (5,)
    assert bundle.valid_mask[4] == 0.0
    assert bundle.lidar_features is None
