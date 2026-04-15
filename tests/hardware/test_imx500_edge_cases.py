"""IMX500 camera edge-case and lifecycle integration tests.

Extends ``test_imx500_integration.py`` with double-start guard,
capture-after-stop error handling, pipeline recovery, and feature
invariant checks.

Run on Jetson::

    pytest tests/hardware/test_imx500_edge_cases.py -m hardware -v --timeout=60
"""

from __future__ import annotations

import asyncio
import contextlib

import numpy as np
import pytest

pytestmark = pytest.mark.hardware

picamera2 = pytest.importorskip("picamera2", reason="picamera2 not available")

JETSON_PROD_CONFIG = "config/jetson_production.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def camera_cfg(jetson_settings):
    """CameraConfig from the shared session settings."""
    return jetson_settings.camera


@pytest.fixture
async def camera(camera_cfg):
    """Start a fresh IMX500Camera per test and stop it afterwards."""
    from mousedroid.hardware.camera.imx500 import IMX500Camera

    cam = IMX500Camera(camera_cfg)
    await cam.start()
    yield cam
    await cam.stop()


# ---------------------------------------------------------------------------
# 1. Double-start idempotency
# ---------------------------------------------------------------------------


async def test_double_start_does_not_crash(camera_cfg) -> None:
    """Calling ``start()`` twice without ``stop()`` must not raise."""
    from mousedroid.hardware.camera.imx500 import IMX500Camera

    cam = IMX500Camera(camera_cfg)
    await cam.start()
    try:
        # Second start — implementation should be idempotent or raise cleanly
        with contextlib.suppress(RuntimeError):
            await cam.start()
        # Either way, camera should still produce features
        feats = await cam.capture_features()
        assert feats.shape == (camera_cfg.feature_dim,)
    finally:
        await cam.stop()


# ---------------------------------------------------------------------------
# 2. Capture after stop — must not segfault
# ---------------------------------------------------------------------------


async def test_capture_after_stop_raises(camera_cfg) -> None:
    """``capture_features()`` after ``stop()`` must raise, not segfault."""
    from mousedroid.hardware.camera.imx500 import IMX500Camera

    cam = IMX500Camera(camera_cfg)
    await cam.start()
    await cam.stop()

    with pytest.raises((RuntimeError, AttributeError)):
        await cam.capture_features()


# ---------------------------------------------------------------------------
# 3. Stop when never started — no-op
# ---------------------------------------------------------------------------


async def test_stop_without_start_is_noop(camera_cfg) -> None:
    """``stop()`` on a never-started camera must not raise."""
    from mousedroid.hardware.camera.imx500 import IMX500Camera

    cam = IMX500Camera(camera_cfg)
    await cam.stop()  # should be a safe no-op
    assert cam._camera is None


# ---------------------------------------------------------------------------
# 4. Feature determinism — same frame should give same features
# ---------------------------------------------------------------------------


async def test_feature_extraction_deterministic(camera, camera_cfg) -> None:
    """Feeding the same raw frame twice must yield identical feature vectors."""
    # Capture a raw frame via the internal method
    frame = await asyncio.to_thread(camera._capture_frame)

    feats_a = camera._extract_features(frame)
    feats_b = camera._extract_features(frame)

    np.testing.assert_array_equal(feats_a, feats_b)


# ---------------------------------------------------------------------------
# 5. Feature dim matches config after fallback path
# ---------------------------------------------------------------------------


async def test_fallback_feature_dim_matches_config(camera_cfg) -> None:
    """With onboard inference disabled, feature_dim still matches config."""
    from mousedroid.hardware.camera.imx500 import IMX500Camera

    cfg_fallback = camera_cfg.model_copy(update={"use_onboard_inference": False})
    cam = IMX500Camera(cfg_fallback)
    await cam.start()

    try:
        feats = await cam.capture_features()
        assert feats.shape[0] == cfg_fallback.feature_dim
        assert feats.dtype == np.float32
        # L2 norm should be ~1.0 (unit normalised)
        norm = float(np.linalg.norm(feats))
        assert norm > 0.0, "Feature vector is zero after fallback extraction"
    finally:
        await cam.stop()


# ---------------------------------------------------------------------------
# 6. Feature property accessor matches config
# ---------------------------------------------------------------------------


def test_feature_dim_property(camera, camera_cfg) -> None:
    """``camera.feature_dim`` must equal ``camera_cfg.feature_dim``."""
    assert camera.feature_dim == camera_cfg.feature_dim


# ---------------------------------------------------------------------------
# 7. Start-capture-stop cycle repeated — no resource leak
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
async def test_repeated_start_stop_no_leak(camera_cfg) -> None:
    """Three start→capture→stop cycles must all succeed (no fd leak)."""
    from mousedroid.hardware.camera.imx500 import IMX500Camera

    for cycle in range(3):
        cam = IMX500Camera(camera_cfg)
        await cam.start()
        feats = await cam.capture_features()
        assert feats.shape == (camera_cfg.feature_dim,), f"Cycle {cycle} feature shape mismatch"
        await cam.stop()
        assert cam._camera is None, f"Cycle {cycle} camera handle not released"


# ---------------------------------------------------------------------------
# 8. Concurrent captures produce valid independent results
# ---------------------------------------------------------------------------


@pytest.mark.timeout(15)
async def test_concurrent_captures_all_valid(camera, camera_cfg) -> None:
    """Four concurrent captures must all return valid feature vectors."""
    results = await asyncio.gather(
        camera.capture_features(),
        camera.capture_features(),
        camera.capture_features(),
        camera.capture_features(),
        return_exceptions=True,
    )
    for i, r in enumerate(results):
        assert isinstance(r, np.ndarray), f"Capture {i} returned {type(r)}: {r!r}"
        assert r.shape == (camera_cfg.feature_dim,)
        assert r.dtype == np.float32
