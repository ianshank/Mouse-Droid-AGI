"""IMX500 camera integration tests for Jetson hardware.

These tests require:
    - ``picamera2`` installed on the Jetson
    - IMX500 camera module connected and enabled in the Linux device tree
    - Camera not already opened by another process

Run on Jetson::

    pytest tests/hardware/test_imx500_integration.py -m hardware -v --timeout=60

All config values (feature_dim, fps, resolution) are read from
``CameraConfig`` — no hardcoded numbers appear in test assertions.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from mousedroid.config.schema import Settings

pytestmark = pytest.mark.hardware

# Skip entire module if picamera2 is unavailable (Windows / x86 CI)
picamera2 = pytest.importorskip("picamera2", reason="picamera2 not available")

JETSON_PROD_CONFIG = "config/jetson_production.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def camera_cfg():
    """Load CameraConfig from jetson_production.yaml."""
    import yaml

    with open(JETSON_PROD_CONFIG) as fh:
        raw = yaml.safe_load(fh)

    settings = Settings(**raw)
    return settings.camera


@pytest.fixture(scope="module")
async def camera(camera_cfg):
    """Start a real IMX500Camera and stop it after all module tests."""
    from mousedroid.hardware.camera.imx500 import IMX500Camera

    cam = IMX500Camera(camera_cfg)
    await cam.start()
    yield cam
    await cam.stop()


# ---------------------------------------------------------------------------
# 1. Feature output shape and dtype
# ---------------------------------------------------------------------------


async def test_capture_features_returns_ndarray(camera, camera_cfg) -> None:
    """capture_features() must return a numpy ndarray."""
    feats = await camera.capture_features()
    assert isinstance(feats, np.ndarray), f"Expected ndarray, got {type(feats)}"


async def test_feature_shape_matches_config(camera, camera_cfg) -> None:
    """Feature vector must be 1-D with length == cfg.feature_dim."""
    feats = await camera.capture_features()
    assert feats.ndim == 1, f"Expected 1-D features, got shape {feats.shape}"
    assert (
        feats.shape[0] == camera_cfg.feature_dim
    ), f"Expected feature_dim={camera_cfg.feature_dim}, got {feats.shape[0]}"


async def test_feature_dtype_is_float32(camera, camera_cfg) -> None:
    """Features must be float32 (L2-normalised output)."""
    feats = await camera.capture_features()
    assert feats.dtype == np.float32, f"Expected float32, got {feats.dtype}"


async def test_features_are_non_zero(camera, camera_cfg) -> None:
    """Feature vector should not be the zero vector (camera is capturing)."""
    feats = await camera.capture_features()
    assert np.linalg.norm(feats) > 0.0, "Feature vector is all zeros — camera may be black"


async def test_features_l2_norm_approx_one(camera, camera_cfg) -> None:
    """L2-normalised features should have unit norm (within floating-point tolerance)."""
    feats = await camera.capture_features()
    norm = float(np.linalg.norm(feats))
    assert abs(norm - 1.0) < 0.01, f"Expected unit norm, got {norm:.4f}"


# ---------------------------------------------------------------------------
# 2. No-onboard-inference fallback path
# ---------------------------------------------------------------------------


async def test_fallback_mean_pooling(camera_cfg) -> None:
    """With use_onboard_inference=False the mean-pooling path must produce valid features."""
    from mousedroid.hardware.camera.imx500 import IMX500Camera

    # Override config to disable onboard inference
    cfg_no_onboard = camera_cfg.model_copy(update={"use_onboard_inference": False})
    cam = IMX500Camera(cfg_no_onboard)
    await cam.start()

    try:
        feats = await cam.capture_features()
        assert feats.shape[0] == cfg_no_onboard.feature_dim
        assert feats.dtype == np.float32
    finally:
        await cam.stop()


# ---------------------------------------------------------------------------
# 3. Frame rate
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
async def test_capture_frame_rate(camera, camera_cfg) -> None:
    """100 consecutive captures must average ≥ 80% of target frame rate."""
    n_frames = 100
    t0 = time.monotonic()

    for _ in range(n_frames):
        await camera.capture_features()

    elapsed = time.monotonic() - t0
    achieved_fps = n_frames / elapsed
    target_fps = float(camera_cfg.fps)
    # Allow ±20% jitter from the configured FPS
    lower_bound = target_fps * 0.80

    assert (
        achieved_fps >= lower_bound
    ), f"Frame rate {achieved_fps:.1f} fps is below 80% of target {target_fps:.1f} fps"


# ---------------------------------------------------------------------------
# 4. Concurrent captures (async isolation)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(15)
async def test_concurrent_captures_no_error(camera, camera_cfg) -> None:
    """Two concurrent capture calls must both return valid feature vectors."""
    for _ in range(5):
        results = await asyncio.gather(
            camera.capture_features(),
            camera.capture_features(),
            return_exceptions=True,
        )
        for r in results:
            assert isinstance(r, np.ndarray), f"Concurrent capture returned: {r!r}"
            assert r.shape == (camera_cfg.feature_dim,)
            assert r.dtype == np.float32


# ---------------------------------------------------------------------------
# 5. Start / stop lifecycle
# ---------------------------------------------------------------------------


async def test_camera_start_stop_cycle(camera_cfg) -> None:
    """Camera should start and stop cleanly without resource errors."""
    from mousedroid.hardware.camera.imx500 import IMX500Camera

    cam = IMX500Camera(camera_cfg)
    await cam.start()
    feats = await cam.capture_features()
    assert feats.shape[0] == camera_cfg.feature_dim
    await cam.stop()
    # After stop, camera object should have released picamera2 handle
    assert cam._camera is None, "Camera handle not released after stop()"
