"""The armed interlock behaves, on the stack that arms it (D-3/D-4/D-24).

``tests/regression/test_lidar_interlock_overlay_aqa.py`` proves the YAML keys
resolve. That is not the same as proving the interlock *works*: this tier
loads the real stacked configuration through :func:`load_settings`, builds the
monitor and projector through the **factory** (so the YAML keys are shown to
reach the objects the orchestrator actually uses), and drives them.

The grace window is exercised against ``observation.timestamp`` rather than
wall time -- that is the clock ``MouseDroidSafetyMonitor.evaluate`` reads, so
the test is deterministic and needs no sleeping.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings
from mousedroid.factory.safety import build_safety_monitor, build_safety_projector
from mousedroid.hardware.lidar.feature_extractor import LidarFeatureExtractor
from mousedroid.hardware.lidar.ld19_driver import LD19LidarDriver
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.sensing.manager import SensorManager

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

#: LiDAR occupies slot 4 of ``valid_mask``; see ``MouseDroidObservationBundle``.
_LIDAR_SLOT = 4
_N_MODALITIES = 6

#: The tick at which the LiDAR goes silent. Timestamps advance from here.
_T0 = 1000.0


def _stacked() -> Settings:
    """Exactly the load ``config/jetson_lidar_only.yaml``'s header prescribes."""
    return load_settings(
        _CONFIG_DIR / "jetson_production.yaml",
        _CONFIG_DIR / "jetson_lidar_only.yaml",
        config_dir=_CONFIG_DIR,
    )


def _observation(timestamp: float, *, lidar_features: np.ndarray | None) -> object:
    """A concrete bundle, never a ``MagicMock``.

    A mock would auto-create whatever attribute the monitor happened to read,
    which is how a green test once vouched for a structurally broken interlock
    (peer review D-1). ``MouseDroidObservationBundle`` is the type the sensing
    layer really produces, so it cannot silently satisfy a field that does not
    exist.
    """
    mask = np.ones(_N_MODALITIES, dtype=np.float32)
    if lidar_features is None:
        mask[_LIDAR_SLOT] = 0.0
    return MouseDroidObservationBundle(
        _timestamp=timestamp,
        _distance_m=5.0,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _valid_mask=mask,
        _lidar_features=lidar_features,
    )


def _clear_features(cfg: Settings) -> np.ndarray:
    """All-ones: normalised range fractions, i.e. max range in every sector."""
    return np.ones(cfg.model.lidar_dim, dtype=np.float32)


# -- D-3: the grace window, which is the part most likely to be got wrong ---


def test_a_dead_lidar_does_not_trip_before_the_grace_elapses() -> None:
    """Tripping instantly would make the configured grace a lie."""
    cfg = _stacked()
    monitor = build_safety_monitor(cfg)
    grace = cfg.safety.lidar_unavailable_grace_s

    monitor.evaluate(_observation(_T0, lidar_features=_clear_features(cfg)), 10.0)

    ctx = monitor.evaluate(_observation(_T0 + grace / 2.0, lidar_features=None), 10.0)
    assert ctx.is_emergency is False
    assert ctx.lidar_clearance_ok is True


def test_a_dead_lidar_trips_once_the_grace_elapses() -> None:
    """The interlock the overlay claims to arm, actually arming."""
    cfg = _stacked()
    monitor = build_safety_monitor(cfg)
    grace = cfg.safety.lidar_unavailable_grace_s

    monitor.evaluate(_observation(_T0, lidar_features=_clear_features(cfg)), 10.0)

    ctx = monitor.evaluate(_observation(_T0 + grace, lidar_features=None), 10.0)
    assert ctx.is_emergency is True
    assert ctx.lidar_clearance_ok is False


def test_a_lidar_dead_from_boot_still_trips() -> None:
    """No good scan ever arrives, so there is no "last valid" time to wait from.

    The generic ``sensor_stale_s`` sweep is not a backstop here -- it only
    fires for a mask slot that was valid at least once.
    """
    cfg = _stacked()
    monitor = build_safety_monitor(cfg)
    grace = cfg.safety.lidar_unavailable_grace_s

    assert monitor.evaluate(_observation(_T0, lidar_features=None), 10.0).is_emergency is False
    ctx = monitor.evaluate(_observation(_T0 + grace, lidar_features=None), 10.0)
    assert ctx.is_emergency is True


def test_production_alone_never_trips_on_the_same_ticks() -> None:
    """The permanent-e-stop this batch's first draft would have shipped.

    Same monitor, same observations, production config only: ``lidar_dim`` is
    0 there, so every tick looks LiDAR-less and an armed policy would fire
    forever. It must stay inert.
    """
    cfg = load_settings(_CONFIG_DIR / "jetson_production.yaml", config_dir=_CONFIG_DIR)
    monitor = build_safety_monitor(cfg)
    for i in range(10):
        ctx = monitor.evaluate(_observation(_T0 + i, lidar_features=None), 10.0)
        assert ctx.is_emergency is False, f"production e-stopped itself on tick {i}"


# -- D-4: the projector changes commanded actions, deliberately -------------


def test_the_projector_is_built_on_the_stack_and_not_on_production() -> None:
    assert build_safety_projector(_stacked()) is not None
    assert (
        build_safety_projector(
            load_settings(_CONFIG_DIR / "jetson_production.yaml", config_dir=_CONFIG_DIR)
        )
        is None
    )


def test_the_projector_clamps_a_commanded_action_near_an_obstacle() -> None:
    """This changes actions in NORMAL operation, not only on a failure path.

    Pinned so the risk class is explicit: enabling the projector is not a
    strictly-fail-safer toggle like the rest of this batch.
    """
    cfg = _stacked()
    projector = build_safety_projector(cfg)
    assert projector is not None
    pcfg = cfg.safety.projector

    action = np.array([cfg.safety.max_velocity_mps, 0.0, 1.0], dtype=np.float32)
    near = SafetyContext(lidar_min_dist_m=pcfg.lidar_brake_distance_m / 2.0)

    clamped = projector.project(action, near)
    assert clamped[0] == pytest.approx(pcfg.crawl_velocity_mps)
    assert abs(clamped[2]) == pytest.approx(pcfg.tight_quarters_omega_max_rads)
    assert action[0] == pytest.approx(cfg.safety.max_velocity_mps), "input was mutated"


def test_the_projector_leaves_a_clear_path_alone() -> None:
    """Proves the clamp is conditional, not a blanket speed cap."""
    cfg = _stacked()
    projector = build_safety_projector(cfg)
    assert projector is not None

    action = np.array([cfg.safety.max_velocity_mps, 0.0, 0.1], dtype=np.float32)
    clear = SafetyContext(lidar_min_dist_m=cfg.safety.lidar_max_range_m)
    assert np.array_equal(projector.project(action, clear), action)


# -- D-24: the whole path, with a real driver and a real sensor manager -----


async def test_an_unplugged_lidar_reaches_the_policy_through_the_real_stack() -> None:
    """Without the D-24 fix this is the test that stays green while the rover
    drives blind: the driver returns a zero-point scan without raising, the
    extractor turns it into an all-clear ring, and the policy is never asked.
    """
    cfg = _stacked()
    assert cfg.lidar is not None

    vision = AsyncMock()
    vision.capture_features = AsyncMock(
        return_value=np.zeros(cfg.camera.feature_dim, dtype=np.float32)
    )
    distance = AsyncMock()
    distance.read_distance_m = AsyncMock(return_value=5.0)
    distance.max_range_m = 6.0
    esp32 = AsyncMock()
    esp32.read_encoders = AsyncMock(return_value=EncoderReading())
    esp32.get_battery_voltage = AsyncMock(return_value=12.0)

    # A real driver whose serial port was never opened — an unplugged cable.
    manager = SensorManager(
        vision,
        distance,
        esp32,
        cfg,
        lidar=LD19LidarDriver(cfg.lidar),
        lidar_feature_extractor=LidarFeatureExtractor(cfg.lidar),
    )
    monitor = build_safety_monitor(cfg)

    observation = await manager.read_all()
    assert observation.lidar_features is None, "a fabricated ring reached the monitor"

    ctx: SafetyContext = monitor.evaluate(observation, 10.0)
    # First tick seeds the grace clock; the policy fires once it elapses.
    assert ctx.is_emergency is False
    later = MouseDroidObservationBundle(
        _timestamp=observation.timestamp + cfg.safety.lidar_unavailable_grace_s,
        _distance_m=5.0,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _valid_mask=observation.valid_mask,
        _lidar_features=None,
    )
    assert monitor.evaluate(later, 10.0).is_emergency is True
