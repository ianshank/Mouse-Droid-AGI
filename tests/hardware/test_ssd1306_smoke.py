"""Real-hardware smoke test for the SSD1306 face display.

Skipped by default; only runs when ``pytest -m hardware`` is invoked on a
Jetson Orin Nano with the SSD1306 panel wired to header I²C bus 7. The
test exercises the full happy path: probe, render BOOT, swap through a
few expressions, then stop cleanly.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from mousedroid.config.schema import FaceDisplayConfig
from mousedroid.hardware.display.expressions import Expression


@pytest.mark.hardware
def test_ssd1306_real_smoke() -> None:
    if os.environ.get("MOUSEDROID_FACE_DISPLAY_SMOKE", "0") != "1":
        pytest.skip("set MOUSEDROID_FACE_DISPLAY_SMOKE=1 to run real-HW smoke test")

    from mousedroid.hardware.display.ssd1306_face_driver import SSD1306FaceDriver

    cfg = FaceDisplayConfig(
        enabled=True,
        i2c_bus=int(os.environ.get("MOUSEDROID_FACE_DISPLAY_BUS", "7")),
        idle_blink_interval_s=2.0,
    )
    drv = SSD1306FaceDriver(cfg)

    async def go() -> None:
        await drv.start()
        try:
            for expr in (
                Expression.NEUTRAL,
                Expression.HAPPY,
                Expression.ALERT,
                Expression.SLEEPY,
            ):
                await drv.show_expression(expr)
                await asyncio.sleep(1.0)
        finally:
            await drv.stop()

    asyncio.run(go())
