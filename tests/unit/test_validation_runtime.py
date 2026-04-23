"""Unit tests for reusable runtime validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.config.schema import Settings
from mousedroid.sensing.lidar_scan import LidarScan
from mousedroid.validation import runtime


def test_resolve_runtime_config_paths_prefers_explicit_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit config arguments override environment-derived paths."""
    monkeypatch.setenv(
        "MOUSEDROID_JETSON_CONFIGS",
        "config/jetson_production.yaml,config/jetson_lidar_only.yaml",
    )

    resolved = runtime.resolve_runtime_config_paths(["config/mock_hardware.yaml"])

    assert resolved == (runtime.Path("config/mock_hardware.yaml"),)


def test_resolve_runtime_config_paths_uses_env_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CSV env vars are resolved in order when no explicit args are passed."""
    monkeypatch.delenv("MOUSEDROID_CONFIG", raising=False)
    monkeypatch.setenv(
        "MOUSEDROID_JETSON_CONFIGS",
        "config/jetson_production.yaml, config/jetson_lidar_only.yaml",
    )

    resolved = runtime.resolve_runtime_config_paths()

    assert resolved == (
        runtime.Path("config/jetson_production.yaml"),
        runtime.Path("config/jetson_lidar_only.yaml"),
    )


@pytest.mark.asyncio
async def test_capture_camera_frame_uses_runtime_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera helper should start, capture, and stop via the factory-built driver."""

    class StubCamera:
        def __init__(self) -> None:
            self._backend = "gstreamer"
            self.started = False
            self.stopped = False

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

        def _capture_frame(self) -> np.ndarray:
            return np.ones((4, 5, 3), dtype=np.uint8)

    stub = StubCamera()
    monkeypatch.setattr(runtime, "build_camera", lambda cfg: stub)

    frame, backend_name = await runtime.capture_camera_frame(Settings(mock_hardware=True))

    assert frame.shape == (4, 5, 3)
    assert backend_name == "gstreamer"
    assert stub.started is True
    assert stub.stopped is True


@pytest.mark.asyncio
async def test_capture_microphone_chunk_requires_live_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Microphone helper should fail when the runtime stream never opens."""

    class StubMicrophone:
        def __init__(self) -> None:
            self._stream = None
            self.stopped = False

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            self.stopped = True

        async def read_chunk(self) -> np.ndarray:
            return np.zeros(4, dtype=np.float32)

    stub = StubMicrophone()
    monkeypatch.setattr(runtime, "build_microphone", lambda cfg: stub)

    with pytest.raises(RuntimeError, match="configured microphone device unavailable"):
        await runtime.capture_microphone_chunk(
            Settings(mock_hardware=True, microphone={"enabled": True, "chunk_size": 4}),
        )

    assert stub.stopped is True


@pytest.mark.asyncio
async def test_play_speaker_tone_writes_chunked_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speaker helper should chunk and write audio using the runtime driver."""

    class StubSpeaker:
        def __init__(self) -> None:
            self._stream = object()
            self.sample_rate = 1000
            self.chunk_size = 4
            self.channels = 1
            self.writes: list[np.ndarray] = []
            self.stopped = False

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            self.stopped = True

        async def write_chunk(self, samples: np.ndarray) -> None:
            self.writes.append(samples.copy())

    stub = StubSpeaker()
    monkeypatch.setattr(runtime, "build_speaker", lambda cfg: stub)

    written = await runtime.play_speaker_tone(
        Settings(mock_hardware=True, speaker={"enabled": True, "sample_rate": 1000}),
        duration_s=0.01,
        frequency_hz=110.0,
    )

    assert written == 12
    assert len(stub.writes) == 3
    assert all(chunk.shape == (4,) for chunk in stub.writes)
    assert stub.stopped is True


def test_lidar_scan_coverage_deg_measures_partial_span() -> None:
    scan = LidarScan(
        angles_deg=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        distances_mm=np.array([1000.0, 1000.0, 1000.0], dtype=np.float32),
        confidences=np.array([100, 100, 100], dtype=np.uint8),
        timestamp=0.0,
        n_points=3,
    )

    coverage_deg = runtime.lidar_scan_coverage_deg(scan)

    assert coverage_deg == pytest.approx(20.0)


def test_lidar_scan_coverage_deg_handles_wraparound() -> None:
    scan = LidarScan(
        angles_deg=np.array([350.0, 355.0, 0.0, 5.0], dtype=np.float32),
        distances_mm=np.array([1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float32),
        confidences=np.array([100, 100, 100, 100], dtype=np.uint8),
        timestamp=0.0,
        n_points=4,
    )

    coverage_deg = runtime.lidar_scan_coverage_deg(scan)

    assert coverage_deg == pytest.approx(15.0)
