"""Camera pipeline integration tests using mock jetson_utils.

Tests CSI capture -> preprocess -> tensor pipeline, frame shape validation,
multiple capture cycles, graceful disconnect handling, and frame rate measurement.

All config values derived from ``CameraConfig`` — no hardcoded numbers.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mousedroid.config.schema import CameraConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def camera_cfg() -> CameraConfig:
    """Provide a CameraConfig with known test values."""
    return CameraConfig(
        resolution_width=640,
        resolution_height=480,
        fps=30,
        feature_dim=256,
        use_onboard_inference=False,
        backend="auto",
        feature_extractor="mean_pool",
        l2_normalize=True,
    )


@pytest.fixture
def camera_cfg_hd() -> CameraConfig:
    """Provide a high-resolution CameraConfig for shape validation."""
    return CameraConfig(
        resolution_width=1280,
        resolution_height=720,
        fps=15,
        feature_dim=512,
        use_onboard_inference=False,
        backend="auto",
        feature_extractor="mean_pool",
        l2_normalize=True,
    )


# ---------------------------------------------------------------------------
# 1. CSI capture -> preprocess -> tensor pipeline
# ---------------------------------------------------------------------------


class TestCSICapturePreprocessPipeline:
    """Verify the full capture -> preprocess -> feature extraction pipeline."""

    @pytest.mark.asyncio
    async def test_mock_camera_returns_features(self, camera_cfg: CameraConfig) -> None:
        """MockCamera.capture_features should return correct-shape array."""
        from mousedroid.hardware.camera.mock_camera import MockCamera

        cam = MockCamera(camera_cfg)
        await cam.start()
        features = await cam.capture_features()
        await cam.stop()

        assert isinstance(features, np.ndarray)
        assert features.shape == (camera_cfg.feature_dim,)
        assert features.dtype == np.float32

    @pytest.mark.asyncio
    async def test_feature_vector_non_zero(self, camera_cfg: CameraConfig) -> None:
        """Feature vectors should not be all zeros."""
        from mousedroid.hardware.camera.mock_camera import MockCamera

        cam = MockCamera(camera_cfg)
        await cam.start()
        features = await cam.capture_features()
        await cam.stop()

        assert np.linalg.norm(features) > 0.0

    @pytest.mark.asyncio
    async def test_successive_features_differ(self, camera_cfg: CameraConfig) -> None:
        """Successive captures should return different features (random mock)."""
        from mousedroid.hardware.camera.mock_camera import MockCamera

        cam = MockCamera(camera_cfg)
        await cam.start()
        feat_a = await cam.capture_features()
        feat_b = await cam.capture_features()
        await cam.stop()

        assert not np.array_equal(feat_a, feat_b)

    @pytest.mark.asyncio
    async def test_feature_extractor_mean_pool(self, camera_cfg: CameraConfig) -> None:
        """Mean-pool feature extractor should produce configured dim output."""
        from mousedroid.hardware.camera.feature_extractor import build_feature_extractor

        extractor = build_feature_extractor(camera_cfg)
        frame = np.random.randint(
            0,
            255,
            (camera_cfg.resolution_height, camera_cfg.resolution_width, 3),
            dtype=np.uint8,
        )
        features = extractor.extract(frame)

        assert features.shape == (camera_cfg.feature_dim,)
        assert features.dtype == np.float32


# ---------------------------------------------------------------------------
# 2. Frame shape validation (configurable resolution from config)
# ---------------------------------------------------------------------------


class TestFrameShapeValidation:
    """Verify frame dimensions match configuration."""

    @pytest.mark.asyncio
    async def test_default_resolution_shape(self, camera_cfg: CameraConfig) -> None:
        """Feature dim should match config for default 640x480."""
        from mousedroid.hardware.camera.mock_camera import MockCamera

        cam = MockCamera(camera_cfg)
        await cam.start()
        features = await cam.capture_features()
        await cam.stop()

        assert features.shape[0] == camera_cfg.feature_dim

    @pytest.mark.asyncio
    async def test_hd_resolution_shape(self, camera_cfg_hd: CameraConfig) -> None:
        """Feature dim should match config for 1280x720 with dim=512."""
        from mousedroid.hardware.camera.mock_camera import MockCamera

        cam = MockCamera(camera_cfg_hd)
        await cam.start()
        features = await cam.capture_features()
        await cam.stop()

        assert features.shape[0] == camera_cfg_hd.feature_dim

    @pytest.mark.asyncio
    async def test_feature_dim_property(self, camera_cfg: CameraConfig) -> None:
        """Camera feature_dim property should match config."""
        from mousedroid.hardware.camera.mock_camera import MockCamera

        cam = MockCamera(camera_cfg)
        assert cam.feature_dim == camera_cfg.feature_dim

    def test_mean_pool_output_shape_various_inputs(self, camera_cfg: CameraConfig) -> None:
        """Mean-pool should produce correct dim regardless of input resolution."""
        from mousedroid.hardware.camera.feature_extractor import build_feature_extractor

        extractor = build_feature_extractor(camera_cfg)

        # Various input sizes should all produce feature_dim output
        for h, w in [(480, 640), (240, 320), (720, 1280)]:
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            features = extractor.extract(frame)
            assert features.shape == (camera_cfg.feature_dim,)


# ---------------------------------------------------------------------------
# 3. Multiple capture cycles
# ---------------------------------------------------------------------------


class TestMultipleCaptureCycles:
    """Verify camera handles multiple capture cycles without error."""

    @pytest.mark.asyncio
    async def test_100_captures_no_error(self, camera_cfg: CameraConfig) -> None:
        """100 consecutive captures must all succeed."""
        from mousedroid.hardware.camera.mock_camera import MockCamera

        cam = MockCamera(camera_cfg)
        await cam.start()

        for i in range(100):
            features = await cam.capture_features()
            assert features.shape == (
                camera_cfg.feature_dim,
            ), f"Capture {i} returned wrong shape: {features.shape}"

        await cam.stop()

    @pytest.mark.asyncio
    async def test_start_stop_restart_cycle(self, camera_cfg: CameraConfig) -> None:
        """Camera should handle start/stop/restart cycles cleanly."""
        from mousedroid.hardware.camera.mock_camera import MockCamera

        cam = MockCamera(camera_cfg)

        for _ in range(3):
            await cam.start()
            features = await cam.capture_features()
            assert features.shape == (camera_cfg.feature_dim,)
            await cam.stop()

    @pytest.mark.asyncio
    async def test_concurrent_captures(self, camera_cfg: CameraConfig) -> None:
        """Concurrent capture calls should all return valid features."""
        from mousedroid.hardware.camera.mock_camera import MockCamera

        cam = MockCamera(camera_cfg)
        await cam.start()

        results = await asyncio.gather(
            cam.capture_features(),
            cam.capture_features(),
            cam.capture_features(),
        )

        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.shape == (camera_cfg.feature_dim,)

        await cam.stop()


# ---------------------------------------------------------------------------
# 4. Graceful handling of camera disconnect
# ---------------------------------------------------------------------------


class TestCameraDisconnectHandling:
    """Verify graceful behaviour when camera connection is lost."""

    @pytest.mark.asyncio
    async def test_mock_camera_stop_is_idempotent(self, camera_cfg: CameraConfig) -> None:
        """Calling stop() multiple times should not raise."""
        from mousedroid.hardware.camera.mock_camera import MockCamera

        cam = MockCamera(camera_cfg)
        await cam.start()
        await cam.stop()
        await cam.stop()  # Second stop should be safe

    @pytest.mark.asyncio
    async def test_capture_after_stop_still_works_mock(self, camera_cfg: CameraConfig) -> None:
        """MockCamera capture after stop should still return valid data.

        In mock mode the camera is stateless, so this verifies the mock
        does not raise on captures without an explicit start.
        """
        from mousedroid.hardware.camera.mock_camera import MockCamera

        cam = MockCamera(camera_cfg)
        # Capture without start — mock should handle gracefully
        features = await cam.capture_features()
        assert features.shape == (camera_cfg.feature_dim,)

    @pytest.mark.asyncio
    async def test_jetson_csi_stop_releases_camera(self, camera_cfg: CameraConfig) -> None:
        """JetsonCSICamera.stop() should set _camera to None."""
        mock_jutils = MagicMock()
        mock_source = MagicMock()
        mock_jutils.videoSource.return_value = mock_source
        mock_jutils.cudaToNumpy.return_value = np.zeros(
            (camera_cfg.resolution_height, camera_cfg.resolution_width, 3),
            dtype=np.uint8,
        )

        with patch.dict("sys.modules", {"jetson_utils": mock_jutils}):
            # Need to reimport after patching
            import importlib

            import mousedroid.hardware.camera.jetson_csi as jcsi_mod

            importlib.reload(jcsi_mod)

            cam = jcsi_mod.JetsonCSICamera(camera_cfg)
            # Manually set up to simulate started state
            cam._camera = mock_source
            cam._backend = "jetson_utils"

            await cam.stop()
            assert cam._camera is None


# ---------------------------------------------------------------------------
# 5. Frame rate measurement (mock timing)
# ---------------------------------------------------------------------------


class TestFrameRateMeasurement:
    """Verify capture rate measurement using mock timing."""

    @pytest.mark.asyncio
    async def test_capture_rate_achievable(self, camera_cfg: CameraConfig) -> None:
        """Mock camera should achieve at least target FPS (no real I/O)."""
        from mousedroid.hardware.camera.mock_camera import MockCamera

        cam = MockCamera(camera_cfg)
        await cam.start()

        n_frames = 50
        t0 = time.monotonic()

        for _ in range(n_frames):
            await cam.capture_features()

        elapsed = time.monotonic() - t0
        achieved_fps = n_frames / elapsed if elapsed > 0 else float("inf")

        await cam.stop()

        # Mock should easily exceed target since there's no real I/O
        # We just verify it doesn't hang or take unreasonably long
        target_fps = float(camera_cfg.fps)
        assert achieved_fps >= target_fps * 0.5, (
            f"Mock capture rate {achieved_fps:.1f} fps is less than 50% of "
            f"target {target_fps:.1f} fps — indicates performance issue"
        )

    @pytest.mark.asyncio
    async def test_capture_timing_consistent(self, camera_cfg: CameraConfig) -> None:
        """Capture times should be relatively consistent (low variance)."""
        from mousedroid.hardware.camera.mock_camera import MockCamera

        cam = MockCamera(camera_cfg)
        await cam.start()

        times: list[float] = []
        for _ in range(20):
            t0 = time.monotonic()
            await cam.capture_features()
            times.append(time.monotonic() - t0)

        await cam.stop()

        mean_t = sum(times) / len(times)
        # All captures should complete within 10x the mean (no outliers)
        for t in times:
            assert t < mean_t * 10, f"Capture time {t:.4f}s is >10x mean {mean_t:.4f}s"
