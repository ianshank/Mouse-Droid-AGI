"""Tests for the SSD1306 face driver with luma + smbus2 patched.

These tests run with no real hardware. They verify:

* the I²C probe uses the configured bus + address;
* concurrent ``show_expression`` calls are serialised under the lock;
* the blink task is cancelled cleanly on ``stop()``;
* lifecycle is idempotent.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema import FaceDisplayConfig
from mousedroid.hardware.display.expressions import Expression
from mousedroid.hardware.protocols import FaceDisplayProtocol


class _FakeSMBus:
    """Minimal smbus2.SMBus stand-in usable as a context manager."""

    instances: ClassVar[list[_FakeSMBus]] = []

    def __init__(self, bus: int) -> None:
        self.bus = bus
        self.read_calls: list[int] = []
        _FakeSMBus.instances.append(self)

    def __enter__(self) -> _FakeSMBus:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def read_byte(self, address: int) -> int:
        self.read_calls.append(address)
        return 0


class _FakeDevice:
    """Stand-in for a luma.oled.device.ssd1306 instance."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.frames: list[Any] = []
        self.cleared = 0
        self.lock = asyncio.Lock()

    def display(self, image: Any) -> None:
        self.frames.append(image)

    def clear(self) -> None:
        self.cleared += 1


@pytest.fixture(autouse=True)
def patch_luma_smbus(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject fake luma + smbus2 modules so the driver can import them."""
    _FakeSMBus.instances.clear()

    smbus2_module = types.ModuleType("smbus2")
    smbus2_module.SMBus = _FakeSMBus  # type: ignore[attr-defined]

    serial_factory = MagicMock(return_value=object())
    luma_core_module = types.ModuleType("luma.core")
    luma_core_iface_module = types.ModuleType("luma.core.interface")
    luma_core_serial_module = types.ModuleType("luma.core.interface.serial")
    luma_core_serial_module.i2c = serial_factory  # type: ignore[attr-defined]

    fake_devices: list[_FakeDevice] = []

    def _ssd1306(serial: Any, **kwargs: Any) -> _FakeDevice:
        dev = _FakeDevice(serial=serial, **kwargs)
        fake_devices.append(dev)
        return dev

    luma_oled_module = types.ModuleType("luma.oled")
    luma_oled_device_module = types.ModuleType("luma.oled.device")
    luma_oled_device_module.ssd1306 = _ssd1306  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "smbus2", smbus2_module)
    monkeypatch.setitem(sys.modules, "luma", types.ModuleType("luma"))
    monkeypatch.setitem(sys.modules, "luma.core", luma_core_module)
    monkeypatch.setitem(sys.modules, "luma.core.interface", luma_core_iface_module)
    monkeypatch.setitem(sys.modules, "luma.core.interface.serial", luma_core_serial_module)
    monkeypatch.setitem(sys.modules, "luma.oled", luma_oled_module)
    monkeypatch.setitem(sys.modules, "luma.oled.device", luma_oled_device_module)

    return {"serial_factory": serial_factory, "fake_devices": fake_devices}


@pytest.fixture
def cfg() -> FaceDisplayConfig:
    return FaceDisplayConfig(
        enabled=True,
        i2c_bus=7,
        i2c_address=0x3C,
        idle_blink_interval_s=0.0,  # disable blink for deterministic tests
    )


def test_ssd1306_satisfies_protocol(cfg: FaceDisplayConfig) -> None:
    from mousedroid.hardware.display.ssd1306_face_driver import SSD1306FaceDriver

    drv = SSD1306FaceDriver(cfg)
    assert isinstance(drv, FaceDisplayProtocol)


async def test_start_probes_configured_bus_and_address(
    cfg: FaceDisplayConfig, patch_luma_smbus: dict[str, Any]
) -> None:
    from mousedroid.hardware.display.ssd1306_face_driver import SSD1306FaceDriver

    drv = SSD1306FaceDriver(cfg)
    await drv.start()
    try:
        assert len(_FakeSMBus.instances) == 1
        bus = _FakeSMBus.instances[0]
        assert bus.bus == cfg.i2c_bus
        assert bus.read_calls == [cfg.i2c_address]
        # Boot banner rendered.
        device = patch_luma_smbus["fake_devices"][-1]
        assert len(device.frames) == 1
        # Serial constructed with the configured port + address.
        patch_luma_smbus["serial_factory"].assert_called_once_with(
            port=cfg.i2c_bus,
            address=cfg.i2c_address,
        )
    finally:
        await drv.stop()


async def test_show_expression_calls_under_lock_serialise(
    cfg: FaceDisplayConfig, patch_luma_smbus: dict[str, Any]
) -> None:
    from mousedroid.hardware.display.ssd1306_face_driver import SSD1306FaceDriver

    drv = SSD1306FaceDriver(cfg)
    await drv.start()
    try:
        device = patch_luma_smbus["fake_devices"][-1]
        before = len(device.frames)
        await asyncio.gather(
            drv.show_expression(Expression.HAPPY),
            drv.show_expression(Expression.ALERT),
            drv.show_expression(Expression.SAD),
        )
        # Three expression frames after the boot banner.
        assert len(device.frames) - before == 3
    finally:
        await drv.stop()


async def test_stop_cancels_blink_task() -> None:
    from mousedroid.hardware.display.ssd1306_face_driver import SSD1306FaceDriver

    cfg = FaceDisplayConfig(enabled=True, idle_blink_interval_s=0.05)
    drv = SSD1306FaceDriver(cfg)
    await drv.start()
    # Yield a few times so the blink task definitely starts.
    await asyncio.sleep(0.01)
    assert drv._blink_task is not None
    await drv.stop()
    assert drv._blink_task is None
    assert drv.started is False


async def test_stop_idempotent(cfg: FaceDisplayConfig) -> None:
    from mousedroid.hardware.display.ssd1306_face_driver import SSD1306FaceDriver

    drv = SSD1306FaceDriver(cfg)
    await drv.start()
    await drv.stop()
    await drv.stop()
    assert drv.started is False


async def test_blink_does_not_overwrite_concurrent_show_expression(
    patch_luma_smbus: dict[str, Any],
) -> None:
    """show_expression during blink must not be clobbered when eyes reopen.

    Sequence:
    1. start() — panel up, blink task spawned with very short interval.
    2. Blink fires: eyes close (BLINK rendered).
    3. Before eyes reopen, show_expression(HAPPY) is called explicitly.
    4. Eyes reopen — the blink loop must NOT restore the pre-blink expression
       because _current is now HAPPY (no longer BLINK).
    5. Final expression must be HAPPY.
    """
    from mousedroid.hardware.display.expressions import Expression
    from mousedroid.hardware.display.ssd1306_face_driver import SSD1306FaceDriver

    # Interval must be long enough that a *second* blink cannot fire during
    # the observation window (asyncio.sleep(0.05) below).
    # Rule: blink_close_duration_s << idle_blink_interval_s > observation_sleep
    cfg = FaceDisplayConfig(
        enabled=True,
        idle_blink_interval_s=0.03,
        blink_close_duration_s=0.01,
    )
    drv = SSD1306FaceDriver(cfg)
    await drv.start()
    try:
        # The blink loop only fires when _current is not None/EMERGENCY.
        # Render an initial expression so the loop has a "previous" to work with.
        await drv.show_expression(Expression.NEUTRAL)

        # Wait until the blink loop renders BLINK (eyes closed).
        for _ in range(20):
            await asyncio.sleep(0.01)
            if drv._current is Expression.BLINK:
                break

        # While eyes are closed, push an explicit new expression.
        await drv.show_expression(Expression.HAPPY)
        assert drv._current is Expression.HAPPY

        # Let the blink close-duration elapse so the restore branch runs,
        # but stay well below the next blink interval so a second blink
        # cannot fire and overwrite our assertion.
        await asyncio.sleep(0.02)

        # The fix: HAPPY must survive because _current is no longer BLINK.
        assert drv._current is Expression.HAPPY
    finally:
        await drv.stop()


async def test_probe_failure_propagates_when_fallback_disabled() -> None:
    from mousedroid.hardware.display.ssd1306_face_driver import SSD1306FaceDriver

    class _ErrSMBus(_FakeSMBus):
        def read_byte(self, address: int) -> int:
            raise OSError("no device")

    sys.modules["smbus2"].SMBus = _ErrSMBus  # type: ignore[attr-defined]

    cfg = FaceDisplayConfig(
        enabled=True,
        idle_blink_interval_s=0.0,
        fallback_to_mock_on_error=False,
    )
    drv = SSD1306FaceDriver(cfg)
    with pytest.raises(OSError, match="no device"):
        await drv.start()
