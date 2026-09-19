"""A LiDAR that produces no frames must not read as an all-clear ring (D-24).

S-2 removed the ``np.ones(feature_dim)`` substitute from
``SensorManager._safe_lidar_read``'s *exception* path. The same vector was
still manufactured one layer down, on a path that never raises:
:meth:`LD19LidarDriver.read_scan` returns ``empty_scan()`` when the serial
port was never opened or no valid frame arrived, and
:meth:`LidarFeatureExtractor.extract` maps a zero-point scan to
``np.ones(n_sectors)``. Features are normalised range fractions, so that is
*maximum range in every sector* -- and because the features were "present",
``SafetyConfig.lidar_unavailable_policy`` was never consulted at all.

The distinction this file pins is the one that makes the fix safe rather than
merely strict: a zero-point scan from a *responding* sensor means nothing lies
inside ``[min_range_m, max_range_m]``, which all-ones states correctly. Only a
non-responding sensor is absent data.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.schema import Settings
from mousedroid.hardware.lidar.feature_extractor import LidarFeatureExtractor
from mousedroid.hardware.lidar.ld19_driver import LD19LidarDriver
from mousedroid.hardware.lidar.ld19_protocol import LD19Frame, LD19Point
from mousedroid.sensing.lidar_scan import LidarScan, empty_scan
from mousedroid.sensing.manager import SensorManager

_N_SECTORS = 36


def _settings() -> Settings:
    return Settings.model_validate(
        {
            "mock_hardware": True,
            "lidar": {"enabled": True, "n_sectors": _N_SECTORS, "feature_dim": _N_SECTORS},
        }
    )


def _manager(cfg: Settings, lidar: object) -> SensorManager:
    """A real ``SensorManager`` and a real extractor -- only the driver is a double."""
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
    if cfg.lidar is None:  # pragma: no cover - guarded by _settings()
        raise AssertionError("lidar config missing")
    return SensorManager(
        vision,
        distance,
        esp32,
        cfg,
        lidar=lidar,  # type: ignore[arg-type]
        lidar_feature_extractor=LidarFeatureExtractor(cfg.lidar),
    )


def _driver_returning(scan: LidarScan) -> AsyncMock:
    lidar = AsyncMock()
    lidar.read_scan = AsyncMock(return_value=scan)
    return lidar


def _open_room_scan(cfg: Settings) -> LidarScan:
    """Frames DID arrive; every beam is a no-return (0 mm), i.e. an open room.

    Built through the real ``_assemble_scan`` rather than hand-rolled, so the
    test cannot drift from what the driver actually produces.
    """
    if cfg.lidar is None:  # pragma: no cover
        raise AssertionError("lidar config missing")
    frames = [
        LD19Frame(
            speed_deg_s=2000.0,
            start_angle_deg=float(angle),
            end_angle_deg=float(angle + 10.0),
            points=tuple(LD19Point(distance_mm=0, confidence=200) for _ in range(12)),
            timestamp_ms=0,
        )
        for angle in range(0, 360, 10)
    ]
    return LD19LidarDriver._assemble_scan(frames, cfg.lidar)


# -- The LidarScan field ----------------------------------------------------


def test_sensor_responding_defaults_true() -> None:
    """Invariant 6: every existing construction site keeps its meaning."""
    assert empty_scan().sensor_responding is True
    assert (
        LidarScan(
            angles_deg=np.array([], dtype=np.float32),
            distances_mm=np.array([], dtype=np.float32),
            confidences=np.array([], dtype=np.uint8),
            timestamp=0.0,
            n_points=0,
        ).sensor_responding
        is True
    )


def test_sensor_responding_is_keyword_only_on_the_factory() -> None:
    """Positional ``empty_scan(False)`` would read as an unlabelled boolean."""
    with pytest.raises(TypeError):
        empty_scan(False)  # type: ignore[misc]


# -- The driver's three empty paths, told apart -----------------------------


async def test_unopened_serial_port_reports_not_responding() -> None:
    """An unplugged USB cable, or a permission error on the by-id symlink."""
    cfg = _settings()
    if cfg.lidar is None:  # pragma: no cover
        raise AssertionError("lidar config missing")
    scan = await LD19LidarDriver(cfg.lidar).read_scan()
    assert scan.n_points == 0
    assert scan.sensor_responding is False


async def test_diagnostics_path_agrees_with_the_plain_read() -> None:
    """Two code paths, one meaning -- the dashboard must not disagree."""
    cfg = _settings()
    if cfg.lidar is None:  # pragma: no cover
        raise AssertionError("lidar config missing")
    scan, _stats = await LD19LidarDriver(cfg.lidar).read_scan_with_diagnostics()
    assert scan.sensor_responding is False


def test_an_open_room_still_counts_as_responding() -> None:
    """The counter-case that stops this fix from braking a rover in a clear space.

    On the LD19 a no-return beam reports 0 mm, which is below ``min_range_m``,
    so a room with nothing inside ``max_range_m`` legitimately yields zero
    points. The sensor is working perfectly.
    """
    cfg = _settings()
    scan = _open_room_scan(cfg)
    assert scan.n_points == 0
    assert scan.sensor_responding is True


# -- The sensing boundary ---------------------------------------------------


async def test_a_non_responding_sensor_reports_the_modality_absent() -> None:
    """The D-24 assertion: no fabricated ring reaches the safety monitor."""
    cfg = _settings()
    manager = _manager(cfg, _driver_returning(empty_scan(sensor_responding=False)))
    features, ok = await manager._safe_lidar_read()
    assert features is None
    assert ok is False


async def test_an_open_room_still_yields_a_real_all_clear_vector() -> None:
    """Proves the guard is not simply "zero points means absent"."""
    cfg = _settings()
    manager = _manager(cfg, _driver_returning(_open_room_scan(cfg)))
    features, ok = await manager._safe_lidar_read()
    assert ok is True
    assert features is not None
    assert np.array_equal(features, np.ones(_N_SECTORS, dtype=np.float32))


async def test_the_raw_scan_is_still_cached_for_telemetry() -> None:
    """Cached *before* the guard, so the dashboard shows "0 points", not a
    stale ring frozen in place at the moment the sensor died."""
    cfg = _settings()
    scan = empty_scan(sensor_responding=False)
    manager = _manager(cfg, _driver_returning(scan))
    await manager._safe_lidar_read()
    assert manager.last_lidar_scan is scan


async def test_a_healthy_scan_is_unaffected() -> None:
    """The regression net: normal operation must be byte-identical."""
    cfg = _settings()
    angles = np.linspace(0.0, 350.0, num=_N_SECTORS, dtype=np.float32)
    scan = LidarScan(
        angles_deg=angles,
        distances_mm=np.full(_N_SECTORS, 6000.0, dtype=np.float32),
        confidences=np.full(_N_SECTORS, 200, dtype=np.uint8),
        timestamp=0.0,
        n_points=_N_SECTORS,
    )
    manager = _manager(cfg, _driver_returning(scan))
    features, ok = await manager._safe_lidar_read()
    assert ok is True
    assert features is not None
    assert float(np.min(features)) == pytest.approx(0.5)
