"""D-24 changes what a *dead* LiDAR looks like, and nothing else.

The fix adds ``LidarScan.sensor_responding`` and makes
``SensorManager._safe_lidar_read`` report the modality absent when it is
``False``. CLAUDE.md invariant 6 applies to the new field; the rest of this
file pins the blast radius, including the two observables that genuinely do
change so they are recorded rather than discovered.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings
from mousedroid.hardware.lidar.feature_extractor import LidarFeatureExtractor
from mousedroid.hardware.lidar.mock_lidar import MockLidar
from mousedroid.safety.monitor import MouseDroidSafetyMonitor
from mousedroid.sensing.lidar_scan import LidarScan, empty_scan
from mousedroid.sensing.manager import SensorManager

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_NOT_AN_OVERLAY = {"baselines.yaml", "default.yaml"}
_N_SECTORS = 36
_LIDAR_SLOT = 4


def _settings() -> Settings:
    return Settings.model_validate(
        {
            "mock_hardware": True,
            "lidar": {"enabled": True, "n_sectors": _N_SECTORS, "feature_dim": _N_SECTORS},
            "safety": {"sensor_stale_s": 100.0},
        }
    )


def _manager(cfg: Settings, lidar: object) -> SensorManager:
    vision = AsyncMock()
    vision.capture_features = AsyncMock(
        return_value=np.zeros(cfg.camera.feature_dim, dtype=np.float32)
    )
    distance = AsyncMock()
    distance.read_distance_m = AsyncMock(return_value=2.0)
    distance.max_range_m = 4.0
    esp32 = AsyncMock()
    esp32.read_encoders = AsyncMock(return_value=EncoderReading())
    esp32.get_battery_voltage = AsyncMock(return_value=12.0)
    if cfg.lidar is None:  # pragma: no cover
        raise AssertionError("lidar config missing")
    return SensorManager(
        vision,
        distance,
        esp32,
        cfg,
        lidar=lidar,  # type: ignore[arg-type]
        lidar_feature_extractor=LidarFeatureExtractor(cfg.lidar),
    )


# -- Invariant 6: the new field is additive --------------------------------


def test_the_field_has_a_default_so_existing_call_sites_compile() -> None:
    scan = LidarScan(
        angles_deg=np.array([1.0], dtype=np.float32),
        distances_mm=np.array([500.0], dtype=np.float32),
        confidences=np.array([200], dtype=np.uint8),
        timestamp=0.0,
        n_points=1,
    )
    assert scan.sensor_responding is True


def test_the_field_is_last_so_positional_construction_is_unchanged() -> None:
    """A new field inserted mid-signature would silently shift every argument."""
    scan = LidarScan(
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.uint8),
        0.0,
        0,
    )
    assert scan.n_points == 0
    assert scan.sensor_responding is True


def test_the_scan_is_still_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        empty_scan().sensor_responding = False  # type: ignore[misc]


def test_empty_scan_default_is_unchanged() -> None:
    assert empty_scan().sensor_responding is True


# -- The mock path, which every non-hardware test rides --------------------


async def test_the_mock_lidar_is_unaffected() -> None:
    """Mock scans carry real points, so nothing here should newly fail closed."""
    cfg = _settings()
    if cfg.lidar is None:  # pragma: no cover
        raise AssertionError("lidar config missing")
    lidar = MockLidar(cfg.lidar)
    scan = await lidar.read_scan()
    assert scan.n_points > 0
    assert scan.sensor_responding is True

    features, ok = await _manager(cfg, lidar)._safe_lidar_read()
    assert ok is True
    assert features is not None


# -- No shipped deployment changes its safety decision ---------------------


@pytest.mark.parametrize(
    "overlay",
    sorted(p.name for p in _CONFIG_DIR.glob("*.yaml") if p.name not in _NOT_AN_OVERLAY),
)
def test_every_overlay_but_the_lidar_rig_still_ignores_an_absent_lidar(overlay: str) -> None:
    """D-24 makes ``lidar_unavailable_policy`` *reachable*; it does not arm it.

    On an ``ignore`` stack a dead LiDAR produced ``clearance_ok=True,
    is_emergency=False`` before this change and must still produce it.
    """
    cfg = load_settings(_CONFIG_DIR / overlay, config_dir=_CONFIG_DIR)
    if cfg.safety.lidar_unavailable_policy != "ignore":
        pytest.skip(f"{overlay} deliberately arms the policy")
    monitor = MouseDroidSafetyMonitor(cfg.safety)
    _dist, clearance_ok, emergency = monitor._evaluate_lidar_clearance(None, 1.0)
    assert clearance_ok is True
    assert emergency is False


# -- The two observables that DO change, recorded rather than hidden -------


async def test_a_dead_lidar_now_reports_an_absent_modality_not_full_range() -> None:
    """Telemetry sees ``inf`` and a zeroed mask slot where it used to see
    ``lidar_max_range_m`` and a valid slot.

    Both are more truthful, and the ``read_all`` docstring, the
    ``_safe_lidar_read`` docstring and ``frame_builder`` already describe this
    as the intended behaviour -- the code simply did not do it. Pinned so the
    dashboard change is a decision on the record.
    """
    cfg = _settings()
    lidar = AsyncMock()
    lidar.read_scan = AsyncMock(return_value=empty_scan(sensor_responding=False))
    observation = await _manager(cfg, lidar).read_all()

    assert observation.lidar_features is None
    assert observation.valid_mask[_LIDAR_SLOT] == pytest.approx(0.0)

    ctx = MouseDroidSafetyMonitor(cfg.safety).evaluate(observation, 10.0)
    assert ctx.lidar_min_dist_m == float("inf")
    assert ctx.lidar_clearance_ok is True, "default policy is ignore; no new stop"
    assert ctx.is_emergency is False
