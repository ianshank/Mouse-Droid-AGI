from __future__ import annotations

import pytest


@pytest.mark.hardware
def test_pyserial_available() -> None:
    try:
        import serial  # noqa: F401
    except ImportError:
        pytest.skip("pyserial not installed")


@pytest.mark.hardware
def test_ld19_serial_open() -> None:
    try:
        import serial as pyserial
    except ImportError:
        pytest.skip("pyserial not installed")

    from mousedroid.config.schema import LidarConfig

    cfg = LidarConfig()
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
async def test_ld19_read_scan() -> None:
    try:
        import serial  # noqa: F401
    except ImportError:
        pytest.skip("pyserial not installed")

    from mousedroid.config.schema import LidarConfig
    from mousedroid.hardware.lidar.ld19_driver import LD19LidarDriver

    cfg = LidarConfig()
    driver = LD19LidarDriver(cfg)
    try:
        try:
            await driver.start()
        except Exception as exc:
            pytest.skip(f"LD19 not available on {cfg.serial_port}: {exc}")

        scan = await driver.read_scan()
    finally:
        await driver.stop()

    assert scan.n_points > 0, "no LiDAR points returned — device may be unpowered"

    angles = scan.angles_deg
    distances_m = scan.distances_mm / 1000.0

    angle_span = float(angles.max() - angles.min())
    assert angle_span >= 270.0, (
        f"angular coverage {angle_span:.1f}° below 270° threshold — "
        "LiDAR likely blocked or misaligned"
    )

    in_range = (distances_m >= cfg.min_range_m) & (distances_m <= cfg.max_range_m)
    assert bool(in_range.all()), (
        f"distances outside [{cfg.min_range_m}, {cfg.max_range_m}] m: "
        f"min={float(distances_m.min()):.3f}, max={float(distances_m.max()):.3f}"
    )
