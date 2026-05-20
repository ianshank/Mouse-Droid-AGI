"""Hardware-bound PR #104 tests — exercise the live rover.

These tests are gated by :data:`pytest.mark.hardware` so CI (which always
sees ``MOUSEDROID_MOCK_HARDWARE=true`` and runs ``-m "not hardware"``) skips
them. On the Jetson the rover-side runner picks them up via
``pytest -m hardware``.

What this covers (the rover-side mirror of the integration / E2E tests):

1. **Live factory wiring** — ``cfg.esp32.enabled=False`` resolves to a
   ``MockESP32Driver`` even on the Jetson, while every other subsystem
   (camera, LiDAR) keeps using the real driver.
2. **JetsonCSI camera ``capture_raw_jpeg``** — produces a non-empty JPEG via
   either the GStreamer or V4L2 fallback path, depending on what the
   container has installed.
3. **Orchestrator boots + ticks** with ESP32 disabled — no monkey-patches
   on ``orchestrator.start()`` required.

If any of these breaks on the Jetson, the dashboard verification path
silently regresses to "show solid green" or "service crash-loops" — the
exact failure modes PR #104 was created to close. That makes these tests
the canonical *operator-facing* acceptance check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings

pytestmark = [pytest.mark.hardware]


# ---------------------------------------------------------------------------
# Skip-gate helper — we want a *graceful* skip on a non-Jetson host
# ---------------------------------------------------------------------------


def _require_jetson() -> None:
    from tests._jetson_hardware import is_jetson_host

    if not is_jetson_host():
        pytest.skip("Not running on Jetson host — hardware test deferred.")


# ---------------------------------------------------------------------------
# Factory wiring on real hardware
# ---------------------------------------------------------------------------


def test_factory_resolves_mock_esp32_when_disabled_on_jetson(
    jetson_settings: Settings,
) -> None:
    """On the Jetson, ``esp32.enabled=False`` resolves to MockESP32Driver.

    Confirms the deployed schema-driven escape hatch (vs the prior monkey-
    patches at ``/opt/mousedroid/src``). If the operator's ``docker.env``
    sets ``MOUSEDROID_ESP32__ENABLED=false`` but the container has a
    stale factory build, the orchestrator boots with a real
    ``SerialESP32Driver`` and starts crashing on ``connect()``.
    """
    _require_jetson()
    from mousedroid.comms.mock_driver import MockESP32Driver
    from mousedroid.factory import build_esp32_driver

    # Apply the dashboard-mode override to a copy of the production
    # settings so we don't accidentally mutate the session-scoped fixture.
    cfg = jetson_settings.model_copy(deep=True)
    cfg.esp32.enabled = False

    drv = build_esp32_driver(cfg)
    inner = getattr(drv, "_inner", None)
    assert isinstance(inner, MockESP32Driver), (
        f"Jetson factory still resolved a {type(inner).__name__} when "
        f"esp32.enabled=False — schema-driven escape hatch is broken."
    )


@pytest.mark.asyncio
async def test_jetson_csi_capture_raw_jpeg_produces_jpeg(
    jetson_settings: Settings,
) -> None:
    """Live JetsonCSI camera's ``capture_raw_jpeg`` yields a real JPEG.

    Acceptance criterion: the response body starts with the JPEG SOI marker
    (``\\xff\\xd8\\xff``) and decodes to the configured resolution via
    Pillow. The test tolerates both the GStreamer + V4L2 backends — only
    the live Jetson knows which one the container has, and the operator
    contract is "I get a snapshot either way".
    """
    _require_jetson()
    pytest.importorskip("PIL")
    from io import BytesIO

    from PIL import Image

    from mousedroid.factory import build_camera

    cam = build_camera(jetson_settings)
    await cam.start()
    try:
        # Not every VisionProtocol implementation is also a
        # RawFrameSourceProtocol — the test asserts the live deployment
        # IS a RawFrameSourceProtocol since /camera/frame.jpg is the
        # operator-facing surface.
        from mousedroid.hardware.protocols import RawFrameSourceProtocol

        assert isinstance(cam, RawFrameSourceProtocol), (
            f"{type(cam).__name__} is not a RawFrameSourceProtocol — the "
            f"dashboard's /camera/frame.jpg will return 404."
        )
        jpeg = await cam.capture_raw_jpeg()
        assert jpeg is not None, "Live JetsonCSI returned None — camera not warm?"
        assert jpeg.startswith(b"\xff\xd8\xff"), (
            "Bytes returned from capture_raw_jpeg() are not a JPEG."
        )
        im = Image.open(BytesIO(jpeg))
        im.verify()
        im2 = Image.open(BytesIO(jpeg))
        assert im2.size == (
            jetson_settings.camera.resolution_width,
            jetson_settings.camera.resolution_height,
        )
        assert im2.mode == "RGB"
        assert im2.format == "JPEG"
    finally:
        await cam.stop()


@pytest.mark.asyncio
async def test_orchestrator_starts_with_esp32_disabled(
    jetson_settings: Settings,
) -> None:
    """Live orchestrator start with ``esp32.enabled=False`` doesn't raise.

    The canonical PR #104 acceptance: a Jetson with a disconnected ESP32
    must boot the orchestrator cleanly. We start it, let one tick happen,
    then stop it — confirming both halves of the lifecycle work.
    """
    _require_jetson()
    import asyncio

    from mousedroid.factory import build_orchestrator

    cfg = jetson_settings.model_copy(deep=True)
    cfg.esp32.enabled = False

    orch = build_orchestrator(cfg)
    # The orchestrator's start() schedules a long-running task; we await it
    # only long enough to confirm it's running, then signal stop().
    await orch.start()
    try:
        await asyncio.sleep(0.05)
    finally:
        await orch.stop()
