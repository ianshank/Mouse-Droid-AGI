from __future__ import annotations

import pytest

from mousedroid.validation.runtime import (
    lidar_scan_coverage_deg,
    lidar_scan_validation_coverage_deg,
)


@pytest.mark.hardware
def test_pyserial_available() -> None:
    try:
        import serial  # noqa: F401
    except ImportError:
        pytest.skip("pyserial not installed")


@pytest.mark.hardware
def test_ld19_serial_open(jetson_settings) -> None:
    try:
        import serial as pyserial
    except ImportError:
        pytest.skip("pyserial not installed")

    cfg = jetson_settings.lidar
    if cfg is None or not cfg.enabled:
        pytest.skip("LiDAR disabled in config")

    try:
        ser = pyserial.Serial(
            port=cfg.serial_port,
            baudrate=cfg.baud_rate,
            timeout=cfg.read_timeout_s,
        )
    except Exception as exc:
        pytest.skip(f"LD19 serial port {cfg.serial_port} unavailable: {exc}")

    try:
        assert ser.is_open, "serial port failed to report open state"
    finally:
        ser.close()


@pytest.mark.hardware
async def test_ld19_read_scan(jetson_settings) -> None:
    try:
        import serial  # noqa: F401
    except ImportError:
        pytest.skip("pyserial not installed")

    from mousedroid.hardware.lidar.ld19_driver import LD19LidarDriver

    cfg = jetson_settings.lidar
    if cfg is None or not cfg.enabled:
        pytest.skip("LiDAR disabled in config")

    driver = LD19LidarDriver(cfg)
    read_stats = None
    try:
        try:
            await driver.start()
        except Exception as exc:
            pytest.skip(f"LD19 not available on {cfg.serial_port}: {exc}")

        scan_reader = getattr(driver, "read_scan_with_diagnostics", None)
        if callable(scan_reader):
            scan, read_stats = await scan_reader()
        else:
            scan = await driver.read_scan()
    finally:
        await driver.stop()

    assert scan.n_points > 0, "no LiDAR points returned — device may be unpowered"

    distances_m = scan.distances_mm / 1000.0

    point_angle_span = lidar_scan_coverage_deg(scan)
    angle_span = lidar_scan_validation_coverage_deg(
        scan,
        driver_covered_angle_deg=(
            read_stats.covered_angle_deg if read_stats is not None else None
        ),
    )
    assert angle_span >= cfg.min_scan_coverage_deg, (
        f"validation coverage {angle_span:.1f}° below {cfg.min_scan_coverage_deg:.1f}° threshold "
        f"(point coverage {point_angle_span:.1f}°) — "
        "LiDAR likely blocked or misaligned"
    )

    in_range = (distances_m >= cfg.min_range_m) & (distances_m <= cfg.max_range_m)
    assert bool(in_range.all()), (
        f"distances outside [{cfg.min_range_m}, {cfg.max_range_m}] m: "
        f"min={float(distances_m.min()):.3f}, max={float(distances_m.max()):.3f}"
    )
